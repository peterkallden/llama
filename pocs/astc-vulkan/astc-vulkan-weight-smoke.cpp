#include <astcenc.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "astc-vulkan-tensor-contract.h"

namespace {

constexpr float kWeightMin = -1.0f;
constexpr float kWeightScale = 2.0f;
constexpr uint32_t kRows = 12;
constexpr uint32_t kColumns = 48;

struct roundtrip_result {
    double mse = 0.0;
    float max_error = 0.0f;
    double dot_error = 0.0;
};

bool encode_roundtrip(const ggml_vk_astc_format_contract & format,
                      const ggml_vk_astc_weight_layout & layout,
                      const std::vector<float> & weights,
                      const std::vector<float> & activations,
                      roundtrip_result & result) {
    const unsigned int width = layout.texel_columns();
    const unsigned int height = layout.rows;
    const size_t texel_components = static_cast<size_t>(width) * height * 4;
    std::vector<float> texels(texel_components);
    for (uint32_t row = 0; row < layout.rows; ++row) {
        for (uint32_t texel = 0; texel < width; ++texel) {
            const size_t texel_offset = (static_cast<size_t>(row) * width + texel) * 4;
            const size_t weight_offset = static_cast<size_t>(row) * layout.columns + texel * 4;
            for (uint32_t channel = 0; channel < 4; ++channel) {
                texels[texel_offset + channel] =
                    (weights[weight_offset + channel] - kWeightMin) / kWeightScale;
            }
        }
    }

    astcenc_config config{};
    astcenc_error status = astcenc_config_init(
        ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
        ASTCENC_PRE_MEDIUM, 0, &config);
    if (status != ASTCENC_SUCCESS) return false;
    astcenc_context * context = nullptr;
    status = astcenc_context_alloc(&config, 1, &context);
    if (status != ASTCENC_SUCCESS) return false;

    void * input_slice = texels.data();
    astcenc_image input_image{ width, height, 1, ASTCENC_TYPE_F32, &input_slice };
    const astcenc_swizzle swizzle{
        ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A,
    };
    const size_t compressed_bytes = layout.storage_bytes(format);
    std::vector<uint8_t> compressed(compressed_bytes);
    status = astcenc_compress_image(
        context, &input_image, &swizzle, compressed.data(), compressed.size(), 0);
    if (status != ASTCENC_SUCCESS) {
        astcenc_context_free(context);
        return false;
    }

    std::vector<float> decoded(texel_components);
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{ width, height, 1, ASTCENC_TYPE_F32, &decoded_slice };
    status = astcenc_decompress_image(
        context, compressed.data(), compressed.size(), &decoded_image, &swizzle, 0);
    astcenc_context_free(context);
    if (status != ASTCENC_SUCCESS) return false;

    double squared_error = 0.0;
    for (uint32_t row = 0; row < layout.rows; ++row) {
        for (uint32_t column = 0; column < layout.columns; ++column) {
            const size_t weight_index = static_cast<size_t>(row) * layout.columns + column;
            const size_t texel_index = (static_cast<size_t>(row) * width + column / 4) * 4 + column % 4;
            const float reconstructed = decoded[texel_index] * kWeightScale + kWeightMin;
            const float error = std::fabs(weights[weight_index] - reconstructed);
            squared_error += static_cast<double>(error) * error;
            result.max_error = std::max(result.max_error, error);
        }
    }
    result.mse = squared_error / weights.size();

    double reference_dot = 0.0;
    double reconstructed_dot = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        const uint32_t column = static_cast<uint32_t>(i % layout.columns);
        const uint32_t row = static_cast<uint32_t>(i / layout.columns);
        const size_t texel_index = (static_cast<size_t>(row) * width + column / 4) * 4 + column % 4;
        const float reconstructed = decoded[texel_index] * kWeightScale + kWeightMin;
        reference_dot += weights[i] * activations[column];
        reconstructed_dot += reconstructed * activations[column];
    }
    result.dot_error = std::fabs(reference_dot - reconstructed_dot);
    std::printf("ASTC weight %s: %zu bytes, MSE %.8f, max error %.6f, dot error %.6f\n",
                format.name, compressed_bytes, result.mse, result.max_error, result.dot_error);
    return std::isfinite(result.mse) && std::isfinite(result.dot_error);
}

} // namespace

int main() {
    constexpr ggml_vk_astc_weight_layout layout{ kRows, kColumns, 4 };
    static_assert(layout.is_valid(), "weight layout must be valid");
    std::vector<float> weights(static_cast<size_t>(kRows) * kColumns);
    std::vector<float> activations(kColumns);
    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t column = 0; column < kColumns; ++column) {
            weights[static_cast<size_t>(row) * kColumns + column] =
                0.75f * std::sin(0.17f * (row + 1) * (column + 1));
        }
    }
    for (uint32_t column = 0; column < kColumns; ++column) {
        activations[column] = 0.5f * std::cos(0.11f * (column + 1));
    }

    roundtrip_result format_4x4;
    roundtrip_result format_6x6;
    if (!encode_roundtrip(ggml_vk_astc_4x4_unorm_rgba, layout,
                          weights, activations, format_4x4) ||
        !encode_roundtrip(ggml_vk_astc_6x6_unorm_rgba, layout,
                          weights, activations, format_6x6)) {
        std::fprintf(stderr, "ASTC weight smoke failed\n");
        return 1;
    }
    std::printf("ASTC weight smoke passed\n");
    return 0;
}

