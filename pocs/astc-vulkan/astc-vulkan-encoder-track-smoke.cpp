#include <astcenc.h>

#include "astc-vulkan-contract.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

struct track_result {
    double mse = 0.0;
    uint32_t blocks = 0;
    uint32_t dual_plane_blocks = 0;
    uint32_t alpha_plane_blocks = 0;
    size_t bytes = 0;
};

std::vector<float> make_latent_signal(uint32_t width, uint32_t height) {
    std::vector<float> result(static_cast<size_t>(width) * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t offset = (static_cast<size_t>(y) * width + x) * 4;
            const float luminance = 0.5f + 0.35f * std::sin(0.11f * x + 0.07f * y);
            const float residual = 0.5f + 0.30f * std::cos(0.37f * x - 0.19f * y);
            result[offset + 0] = luminance;
            result[offset + 1] = luminance;
            result[offset + 2] = luminance;
            result[offset + 3] = residual;
        }
    }
    return result;
}

void make_block_constant(std::vector<float> & image, uint32_t width, uint32_t height,
                         uint32_t block_x, uint32_t block_y) {
    for (uint32_t y0 = 0; y0 < height; y0 += block_y) {
        for (uint32_t x0 = 0; x0 < width; x0 += block_x) {
            double sum[4]{};
            uint32_t count = 0;
            for (uint32_t y = y0; y < std::min(y0 + block_y, height); ++y) {
                for (uint32_t x = x0; x < std::min(x0 + block_x, width); ++x) {
                    const float * texel = image.data() + (static_cast<size_t>(y) * width + x) * 4;
                    for (uint32_t channel = 0; channel < 4; ++channel) sum[channel] += texel[channel];
                    ++count;
                }
            }
            for (uint32_t channel = 0; channel < 4; ++channel) sum[channel] /= count;
            for (uint32_t y = y0; y < std::min(y0 + block_y, height); ++y) {
                for (uint32_t x = x0; x < std::min(x0 + block_x, width); ++x) {
                    float * texel = image.data() + (static_cast<size_t>(y) * width + x) * 4;
                    for (uint32_t channel = 0; channel < 4; ++channel) {
                        texel[channel] = static_cast<float>(sum[channel]);
                    }
                }
            }
        }
    }
}

bool run_track(const char * name, const ggml_vk_astc_format_contract & format,
               const std::vector<float> & source, const std::vector<float> & input,
               uint32_t width, uint32_t height, bool constrained) {
    astcenc_config config{};
    astcenc_error status = astcenc_config_init(
        ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
        ASTCENC_PRE_THOROUGH, 0, &config);
    if (status != ASTCENC_SUCCESS) return false;
    if (constrained) {
        // These are public astcenc search controls, not a new bitstream. They
        // bias the candidate objective toward an independent alpha component
        // and remove multi-partition candidates for a stable L+A experiment.
        config.cw_r_weight = 1.0f;
        config.cw_g_weight = 1.0f;
        config.cw_b_weight = 1.0f;
        config.cw_a_weight = 4.0f;
        config.tune_partition_count_limit = 1;
        config.tune_2plane_early_out_limit_correlation = 1.0f;
    }
    astcenc_context * context = nullptr;
    status = astcenc_context_alloc(&config, 1, &context);
    if (status != ASTCENC_SUCCESS) return false;
    void * input_slice = const_cast<float *>(input.data());
    astcenc_image input_image{ width, height, 1, ASTCENC_TYPE_F32, &input_slice };
    const astcenc_swizzle swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                   ASTCENC_SWZ_B, ASTCENC_SWZ_A };
    track_result result;
    result.bytes = ggml_vk_astc_image_storage_bytes(format, width, height);
    std::vector<uint8_t> compressed(result.bytes);
    status = astcenc_compress_image(context, &input_image, &swizzle,
                                    compressed.data(), compressed.size(), 0);
    if (status == ASTCENC_SUCCESS) {
        result.blocks = static_cast<uint32_t>(compressed.size() / 16);
        for (size_t offset = 0; offset < compressed.size(); offset += 16) {
            astcenc_block_info info{};
            status = astcenc_get_block_info(context, compressed.data() + offset, &info);
            if (status != ASTCENC_SUCCESS) break;
            if (info.is_dual_plane_block) {
                ++result.dual_plane_blocks;
                if (info.dual_plane_component == 3) ++result.alpha_plane_blocks;
            }
        }
    }
    std::vector<float> decoded(input.size());
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{ width, height, 1, ASTCENC_TYPE_F32, &decoded_slice };
    if (status == ASTCENC_SUCCESS) {
        status = astcenc_decompress_image(context, compressed.data(), compressed.size(),
                                           &decoded_image, &swizzle, 0);
    }
    astcenc_context_free(context);
    if (status != ASTCENC_SUCCESS) return false;
    for (size_t index = 0; index < source.size(); ++index) {
        const double delta = static_cast<double>(source[index]) - decoded[index];
        result.mse += delta * delta;
    }
    result.mse /= source.size();
    std::printf("encoder-track format=%s track=%s bytes=%zu MSE=%.8g blocks=%u "
                "dual-plane=%u alpha-plane=%u\n",
                format.name, name, result.bytes, result.mse, result.blocks,
                result.dual_plane_blocks, result.alpha_plane_blocks);
    return std::isfinite(result.mse);
}

bool run_format(const ggml_vk_astc_format_contract & format) {
    const uint32_t width = format.block_width * 2;
    const uint32_t height = format.block_height * 2;
    const std::vector<float> source = make_latent_signal(width, height);
    std::vector<float> block_constant = source;
    make_block_constant(block_constant, width, height, format.block_width, format.block_height);
    return run_track("standard", format, source, source, width, height, false) &&
        run_track("constrained-search", format, source, source, width, height, true) &&
        run_track("minimal-block-policy", format, source, block_constant, width, height, true);
}

} // namespace

int main() {
    const bool ok = run_format(ggml_vk_astc_4x4_unorm_rgba) &&
        run_format(ggml_vk_astc_5x5_unorm_rgba) &&
        run_format(ggml_vk_astc_6x6_unorm_rgba);
    if (!ok) {
        std::fprintf(stderr, "ASTC encoder track smoke failed\n");
        return 1;
    }
    std::printf("ASTC encoder track smoke passed\n");
    return 0;
}
