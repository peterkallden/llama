#include <astcenc.h>

#include "astc-vulkan-contract.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

constexpr uint32_t kRows = 32;
constexpr uint32_t kColumns = 256;
constexpr uint32_t kCoarseLevels = 16;

enum class channel_semantics {
    independent_rgba,
    coarse_only_pairs,
    coarse_residual_pairs,
    rg16_control_pairs,
};

const char * semantics_name(channel_semantics semantics) {
    switch (semantics) {
        case channel_semantics::independent_rgba:    return "independent-rgba";
        case channel_semantics::coarse_only_pairs:   return "coarse-only-pairs";
        case channel_semantics::coarse_residual_pairs: return "coarse-residual-pairs";
        case channel_semantics::rg16_control_pairs:  return "rg16-control-pairs";
    }
    return "unknown";
}

struct encoded_representation {
    uint32_t width = 0;
    std::vector<float> texels;
    std::vector<float> coarse;
    float minimum = 0.0f;
    float range = 1.0f;
    float residual_scale = 0.0f;
};

double elementwise_mse(const std::vector<float> & reference,
                       const std::vector<float> & candidate) {
    double sum = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double error = reference[i] - candidate[i];
        sum += error * error;
    }
    return sum / reference.size();
}

double activation_relative_mse(const std::vector<float> & reference,
                               const std::vector<float> & candidate) {
    double error_sum = 0.0;
    double reference_sum = 0.0;
    for (uint32_t sample = 1; sample <= 4; ++sample) {
        for (uint32_t row = 0; row < kRows; ++row) {
            double expected = 0.0;
            double actual = 0.0;
            for (uint32_t column = 0; column < kColumns; ++column) {
                const float activation = 0.5f * std::sin(0.017f * sample * (column + 1)) +
                    0.2f * std::cos(0.031f * (sample + 1) * (column + 3));
                const size_t index = static_cast<size_t>(row) * kColumns + column;
                expected += reference[index] * activation;
                actual += candidate[index] * activation;
            }
            const double error = expected - actual;
            error_sum += error * error;
            reference_sum += expected * expected;
        }
    }
    return error_sum / std::max(reference_sum, 1e-12);
}

encoded_representation encode_semantics(const std::vector<float> & weights,
                                        channel_semantics semantics) {
    encoded_representation result;
    const auto [minimum_it, maximum_it] = std::minmax_element(weights.begin(), weights.end());
    result.minimum = *minimum_it;
    result.range = std::max(*maximum_it - result.minimum, 1e-6f);
    const uint32_t values_per_texel = semantics == channel_semantics::independent_rgba ? 4 : 2;
    result.width = (kColumns + values_per_texel - 1) / values_per_texel;
    result.texels.assign(static_cast<size_t>(result.width) * kRows * 4, 0.5f);
    if (semantics == channel_semantics::coarse_only_pairs ||
        semantics == channel_semantics::coarse_residual_pairs) {
        const float coarse_step = result.range / (kCoarseLevels - 1);
        result.residual_scale = coarse_step * 0.5f;
        result.coarse.resize(weights.size());
        for (size_t index = 0; index < weights.size(); ++index) {
            const float level = std::round((weights[index] - result.minimum) / coarse_step);
            result.coarse[index] = result.minimum +
                std::clamp(level, 0.0f, static_cast<float>(kCoarseLevels - 1)) * coarse_step;
        }
    }

    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t column = 0; column < kColumns; ++column) {
            const size_t weight_index = static_cast<size_t>(row) * kColumns + column;
            const uint32_t texel_x = column / values_per_texel;
            const uint32_t lane = column % values_per_texel;
            float * texel = result.texels.data() +
                (static_cast<size_t>(row) * result.width + texel_x) * 4;
            const float normalized = (weights[weight_index] - result.minimum) / result.range;
            if (semantics == channel_semantics::independent_rgba) {
                texel[lane] = normalized;
            } else if (semantics == channel_semantics::coarse_only_pairs ||
                       semantics == channel_semantics::coarse_residual_pairs) {
                const uint32_t coarse_channel = lane * 2;
                const float coarse_normalized =
                    (result.coarse[weight_index] - result.minimum) / result.range;
                texel[coarse_channel] = coarse_normalized;
                if (semantics == channel_semantics::coarse_residual_pairs) {
                    const float residual = weights[weight_index] - result.coarse[weight_index];
                    texel[coarse_channel + 1] = std::clamp(
                        0.5f + residual / (2.0f * result.residual_scale), 0.0f, 1.0f);
                }
            } else {
                const uint32_t quantized = static_cast<uint32_t>(std::round(
                    std::clamp(normalized, 0.0f, 1.0f) * 65535.0f));
                texel[lane * 2] = static_cast<float>(quantized >> 8) / 255.0f;
                texel[lane * 2 + 1] = static_cast<float>(quantized & 0xff) / 255.0f;
            }
        }
    }
    return result;
}

bool astc_roundtrip(const encoded_representation & encoded,
                    const ggml_vk_astc_format_contract & format,
                    std::vector<float> & decoded, size_t & compressed_bytes) {
    compressed_bytes = ggml_vk_astc_image_storage_bytes(format, encoded.width, kRows);
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
                            ASTCENC_PRE_THOROUGH, 0, &config) != ASTCENC_SUCCESS) return false;
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return false;
    void * input_slice = const_cast<float *>(encoded.texels.data());
    astcenc_image input_image{ encoded.width, kRows, 1, ASTCENC_TYPE_F32, &input_slice };
    const astcenc_swizzle swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A };
    std::vector<uint8_t> compressed(compressed_bytes);
    astcenc_error status = astcenc_compress_image(
        context, &input_image, &swizzle, compressed.data(), compressed.size(), 0);
    decoded.resize(encoded.texels.size());
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{ encoded.width, kRows, 1, ASTCENC_TYPE_F32, &decoded_slice };
    if (status == ASTCENC_SUCCESS) {
        status = astcenc_decompress_image(
            context, compressed.data(), compressed.size(), &decoded_image, &swizzle, 0);
    }
    astcenc_context_free(context);
    return status == ASTCENC_SUCCESS;
}

std::vector<float> decode_semantics(const encoded_representation & encoded,
                                    const std::vector<float> & decoded,
                                    channel_semantics semantics) {
    const uint32_t values_per_texel = semantics == channel_semantics::independent_rgba ? 4 : 2;
    std::vector<float> result(static_cast<size_t>(kRows) * kColumns);
    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t column = 0; column < kColumns; ++column) {
            const uint32_t texel_x = column / values_per_texel;
            const uint32_t lane = column % values_per_texel;
            const float * texel = decoded.data() +
                (static_cast<size_t>(row) * encoded.width + texel_x) * 4;
            float reconstructed = 0.0f;
            if (semantics == channel_semantics::independent_rgba) {
                reconstructed = texel[lane] * encoded.range + encoded.minimum;
            } else if (semantics == channel_semantics::coarse_only_pairs ||
                       semantics == channel_semantics::coarse_residual_pairs) {
                const uint32_t coarse_channel = lane * 2;
                reconstructed = texel[coarse_channel] * encoded.range + encoded.minimum;
                if (semantics == channel_semantics::coarse_residual_pairs) {
                    reconstructed += (texel[coarse_channel + 1] - 0.5f) *
                        2.0f * encoded.residual_scale;
                }
            } else {
                const uint32_t high = static_cast<uint32_t>(std::round(
                    std::clamp(texel[lane * 2], 0.0f, 1.0f) * 255.0f));
                const uint32_t low = static_cast<uint32_t>(std::round(
                    std::clamp(texel[lane * 2 + 1], 0.0f, 1.0f) * 255.0f));
                reconstructed = (static_cast<float>((high << 8) | low) / 65535.0f) *
                    encoded.range + encoded.minimum;
            }
            result[static_cast<size_t>(row) * kColumns + column] = reconstructed;
        }
    }
    return result;
}

bool run_semantics(const std::vector<float> & weights,
                   const ggml_vk_astc_format_contract & format,
                   channel_semantics semantics) {
    const encoded_representation encoded = encode_semantics(weights, semantics);
    std::vector<float> decoded;
    size_t compressed_bytes = 0;
    if (!astc_roundtrip(encoded, format, decoded, compressed_bytes)) return false;
    const std::vector<float> reconstructed = decode_semantics(encoded, decoded, semantics);
    const double mse = elementwise_mse(weights, reconstructed);
    const double activation_mse = activation_relative_mse(weights, reconstructed);
    std::printf("channel-semantics format=%s mode=%s values-per-texel=%u bytes=%zu bpw=%.5f MSE=%.8g activation-relative-MSE=%.8g\n",
                format.name, semantics_name(semantics),
                semantics == channel_semantics::independent_rgba ? 4u : 2u,
                compressed_bytes, compressed_bytes * 8.0 / weights.size(), mse, activation_mse);
    return std::isfinite(mse) && std::isfinite(activation_mse);
}

} // namespace

int main() {
    std::vector<float> weights(static_cast<size_t>(kRows) * kColumns);
    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t column = 0; column < kColumns; ++column) {
            weights[static_cast<size_t>(row) * kColumns + column] =
                0.70f * std::sin(0.037f * (row + 1) * (column + 1)) +
                0.15f * std::cos(0.113f * (row + 3) + 0.029f * column);
        }
    }
    for (const auto & format : { ggml_vk_astc_4x4_unorm_rgba,
                                 ggml_vk_astc_5x5_unorm_rgba,
                                 ggml_vk_astc_6x6_unorm_rgba }) {
        for (const auto semantics : { channel_semantics::independent_rgba,
                                      channel_semantics::coarse_only_pairs,
                                      channel_semantics::coarse_residual_pairs,
                                      channel_semantics::rg16_control_pairs }) {
            if (!run_semantics(weights, format, semantics)) {
                std::fprintf(stderr, "channel semantic smoke failed\n");
                return 1;
            }
        }
    }
    return 0;
}
