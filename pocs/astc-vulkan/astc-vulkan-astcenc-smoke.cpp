#include <astcenc.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "astc-vulkan-contract.h"

namespace {

bool run_format(const ggml_vk_astc_format_contract & format) {
    const unsigned int width = format.block_width * 2;
    const unsigned int height = format.block_height * 2;
    const size_t texel_components = static_cast<size_t>(width) * height * format.channels;
    std::vector<float> input(texel_components);
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            const size_t offset = (static_cast<size_t>(y) * width + x) * format.channels;
            input[offset + 0] = 0.15f + 0.70f * x / (width - 1);
            input[offset + 1] = 0.10f + 0.75f * y / (height - 1);
            input[offset + 2] = 0.20f + 0.45f * (x + y) / (width + height - 2);
            input[offset + 3] = 0.95f;
        }
    }

    astcenc_config config{};
    astcenc_error status = astcenc_config_init(
        ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
        ASTCENC_PRE_MEDIUM, 0, &config);
    if (status != ASTCENC_SUCCESS) {
        std::fprintf(stderr, "%s config failed: %s\n", format.name,
                     astcenc_get_error_string(status));
        return false;
    }

    astcenc_context * context = nullptr;
    status = astcenc_context_alloc(&config, 1, &context);
    if (status != ASTCENC_SUCCESS) {
        std::fprintf(stderr, "%s context allocation failed: %s\n", format.name,
                     astcenc_get_error_string(status));
        return false;
    }

    void * input_slice = input.data();
    astcenc_image input_image{ width, height, 1, ASTCENC_TYPE_F32, &input_slice };
    const astcenc_swizzle swizzle{
        ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A,
    };
    const size_t compressed_bytes = ggml_vk_astc_image_storage_bytes(format, width, height);
    std::vector<uint8_t> compressed(compressed_bytes);
    status = astcenc_compress_image(
        context, &input_image, &swizzle, compressed.data(), compressed.size(), 0);
    if (status != ASTCENC_SUCCESS) {
        std::fprintf(stderr, "%s compression failed: %s\n", format.name,
                     astcenc_get_error_string(status));
        astcenc_context_free(context);
        return false;
    }

    std::vector<float> decoded(texel_components);
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{ width, height, 1, ASTCENC_TYPE_F32, &decoded_slice };
    status = astcenc_decompress_image(
        context, compressed.data(), compressed.size(), &decoded_image, &swizzle, 0);
    astcenc_context_free(context);
    if (status != ASTCENC_SUCCESS) {
        std::fprintf(stderr, "%s decompression failed: %s\n", format.name,
                     astcenc_get_error_string(status));
        return false;
    }

    double squared_error = 0.0;
    float max_error = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        const float error = std::fabs(input[i] - decoded[i]);
        squared_error += static_cast<double>(error) * error;
        max_error = std::max(max_error, error);
    }
    const double mse = squared_error / input.size();
    std::printf("ASTC encoder %s: %zu bytes, MSE %.8f, max error %.6f\n",
                format.name, compressed_bytes, mse, max_error);
    // This is a smoke threshold for the generic image-oriented encoder, not a
    // neural-weight quality target. Keep MSE bounded while exposing the larger
    // per-texel endpoint error for later weight-aware evaluation.
    return std::isfinite(mse) && mse <= 0.005;
}

} // namespace

int main() {
    const bool format_4x4_ok = run_format(ggml_vk_astc_4x4_unorm_rgba);
    const bool format_5x5_ok = run_format(ggml_vk_astc_5x5_unorm_rgba);
    const bool format_6x6_ok = run_format(ggml_vk_astc_6x6_unorm_rgba);
    if (!format_4x4_ok || !format_5x5_ok || !format_6x6_ok) {
        std::fprintf(stderr, "ASTC encoder smoke failed\n");
        return 1;
    }
    std::printf("ASTC encoder smoke passed\n");
    return 0;
}
