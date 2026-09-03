#include <astcenc.h>

#include "astc-vulkan-paired.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

struct case_result {
    double mse = 0.0;
    size_t bytes = 0;
    uint32_t unique_payloads = 0;
};

case_result run_case(uint32_t block_width, uint32_t block_height,
                     astc_vulkan_paired_layout layout, bool steering) {
    const uint32_t width = block_width * 2;
    const uint32_t logical_height = block_height * 2;
    const uint32_t texture_height = block_height;
    const size_t texels = static_cast<size_t>(width) * texture_height;
    std::vector<float> source(texels * 4);
    for (uint32_t y = 0; y < texture_height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const float q0 = 0.5f + 0.25f * std::sin(0.17f * x + 0.11f * y);
            const float q1 = 0.5f + 0.25f * std::cos(0.13f * x - 0.07f * y);
            const float alpha = steering ?
                0.5f + 0.35f * std::sin(0.19f * x - 0.23f * y) : 0.5f;
            const auto texel = astc_vulkan_make_paired_texel(q0, q1, alpha, layout);
            const size_t index = (static_cast<size_t>(y) * width + x) * 4;
            source[index + 0] = texel.r;
            source[index + 1] = texel.g;
            source[index + 2] = texel.b;
            source[index + 3] = texel.a;
        }
    }

    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, block_width, block_height, 1,
                            ASTCENC_PRE_THOROUGH, 0, &config) != ASTCENC_SUCCESS) return {};
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return {};
    void * source_slice = source.data();
    astcenc_image source_image{width, texture_height, 1, ASTCENC_TYPE_F32, &source_slice};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                  ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    const size_t bytes = static_cast<size_t>((width + block_width - 1) / block_width) *
                         ((texture_height + block_height - 1) / block_height) * 16;
    std::vector<uint8_t> payload(bytes);
    if (astcenc_compress_image(context, &source_image, &swizzle,
                               payload.data(), payload.size(), 0) != ASTCENC_SUCCESS) {
        astcenc_context_free(context);
        return {};
    }
    std::vector<float> decoded(source.size());
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{width, texture_height, 1, ASTCENC_TYPE_F32, &decoded_slice};
    if (astcenc_decompress_image(context, payload.data(), payload.size(),
                                 &decoded_image, &swizzle, 0) != ASTCENC_SUCCESS) {
        astcenc_context_free(context);
        return {};
    }
    case_result result;
    result.bytes = payload.size();
    for (uint32_t y = 0; y < texture_height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t index = (static_cast<size_t>(y) * width + x) * 4;
            const astc_vulkan_rgba_texel decoded_texel{
                decoded[index], decoded[index + 1], decoded[index + 2], decoded[index + 3]};
            const float q0 = 0.5f + 0.25f * std::sin(0.17f * x + 0.11f * y);
            const float q1 = 0.5f + 0.25f * std::cos(0.13f * x - 0.07f * y);
            const float e0 = q0 - astc_vulkan_paired_weight(decoded_texel, 0, layout);
            const float e1 = q1 - astc_vulkan_paired_weight(decoded_texel, 1, layout);
            result.mse += static_cast<double>(e0) * e0 + static_cast<double>(e1) * e1;
        }
    }
    result.mse /= static_cast<double>(texels * 2);
    result.unique_payloads = 1;
    astcenc_context_free(context);
    (void) logical_height;
    return result;
}

bool run_format(uint32_t width, uint32_t height) {
    bool ok = true;
    for (const auto layout : {astc_vulkan_paired_layout::rg_b,
                              astc_vulkan_paired_layout::r_gb}) {
        const case_result fixed = run_case(width, height, layout, false);
        const case_result steered = run_case(width, height, layout, true);
        const double bits_per_weight = 128.0 / (static_cast<double>(width) * height * 2.0);
        std::printf("paired-d2 %ux%u layout=%s rate=%.5f fixed-mse=%.8g steering-mse=%.8g\n",
                    width, height, astc_vulkan_paired_layout_name(layout), bits_per_weight,
                    fixed.mse, steered.mse);
        ok = ok && fixed.bytes != 0 && steered.bytes == fixed.bytes &&
             std::isfinite(fixed.mse) && std::isfinite(steered.mse);
    }
    return ok;
}

} // namespace

int main() {
    const bool ok = run_format(6, 6) && run_format(8, 5) && run_format(8, 6) &&
                    run_format(10, 6) && run_format(8, 8) && run_format(10, 8) &&
                    run_format(10, 10);
    if (!ok) {
        std::fprintf(stderr, "paired D2 ASTC smoke failed\n");
        return 1;
    }
    std::printf("paired D2 ASTC smoke passed\n");
    return 0;
}
