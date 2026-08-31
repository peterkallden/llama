#include <astcenc.h>

#include "astc-vulkan-contract.h"
#include "astc-vulkan-input.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
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

struct channel_pairing {
    const char * name;
    // coarse(value0), residual(value0), coarse(value1), residual(value1)
    std::array<uint32_t, 4> channels;
};

constexpr std::array<channel_pairing, 3> kCanonicalPairings{{
    { "RG+BA", { 0, 1, 2, 3 } },
    { "RB+GA", { 0, 2, 1, 3 } },
    { "RA+GB", { 0, 3, 1, 2 } },
}};

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
    uint32_t height = 0;
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
                               const std::vector<float> & candidate,
                               uint32_t rows, uint32_t columns) {
    double error_sum = 0.0;
    double reference_sum = 0.0;
    for (uint32_t sample = 1; sample <= 4; ++sample) {
        for (uint32_t row = 0; row < rows; ++row) {
            double expected = 0.0;
            double actual = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                const float activation = 0.5f * std::sin(0.017f * sample * (column + 1)) +
                    0.2f * std::cos(0.031f * (sample + 1) * (column + 3));
                const size_t index = static_cast<size_t>(row) * columns + column;
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
                                        channel_semantics semantics,
                                        const channel_pairing & pairing,
                                        uint32_t rows, uint32_t columns) {
    encoded_representation result;
    const auto [minimum_it, maximum_it] = std::minmax_element(weights.begin(), weights.end());
    result.minimum = *minimum_it;
    result.range = std::max(*maximum_it - result.minimum, 1e-6f);
    const uint32_t values_per_texel = semantics == channel_semantics::independent_rgba ? 4 : 2;
    result.width = (columns + values_per_texel - 1) / values_per_texel;
    result.height = rows;
    result.texels.assign(static_cast<size_t>(result.width) * rows * 4, 0.5f);
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

    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            const size_t weight_index = static_cast<size_t>(row) * columns + column;
            const uint32_t texel_x = column / values_per_texel;
            const uint32_t lane = column % values_per_texel;
            float * texel = result.texels.data() +
                (static_cast<size_t>(row) * result.width + texel_x) * 4;
            const float normalized = (weights[weight_index] - result.minimum) / result.range;
            if (semantics == channel_semantics::independent_rgba) {
                texel[lane] = normalized;
            } else if (semantics == channel_semantics::coarse_only_pairs ||
                       semantics == channel_semantics::coarse_residual_pairs) {
                const uint32_t coarse_channel = pairing.channels[lane * 2];
                const uint32_t residual_channel = pairing.channels[lane * 2 + 1];
                const float coarse_normalized =
                    (result.coarse[weight_index] - result.minimum) / result.range;
                texel[coarse_channel] = coarse_normalized;
                if (semantics == channel_semantics::coarse_residual_pairs) {
                    const float residual = weights[weight_index] - result.coarse[weight_index];
                    texel[residual_channel] = std::clamp(
                        0.5f + residual / (2.0f * result.residual_scale), 0.0f, 1.0f);
                }
            } else {
                const uint32_t quantized = static_cast<uint32_t>(std::round(
                    std::clamp(normalized, 0.0f, 1.0f) * 65535.0f));
                texel[pairing.channels[lane * 2]] =
                    static_cast<float>(quantized >> 8) / 255.0f;
                texel[pairing.channels[lane * 2 + 1]] =
                    static_cast<float>(quantized & 0xff) / 255.0f;
            }
        }
    }
    return result;
}

bool astc_roundtrip(const encoded_representation & encoded,
                    const ggml_vk_astc_format_contract & format,
                    std::vector<float> & decoded, size_t & compressed_bytes) {
    compressed_bytes = ggml_vk_astc_image_storage_bytes(format, encoded.width, encoded.height);
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
                            ASTCENC_PRE_THOROUGH, 0, &config) != ASTCENC_SUCCESS) return false;
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return false;
    void * input_slice = const_cast<float *>(encoded.texels.data());
    astcenc_image input_image{ encoded.width, encoded.height, 1, ASTCENC_TYPE_F32, &input_slice };
    const astcenc_swizzle swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A };
    std::vector<uint8_t> compressed(compressed_bytes);
    astcenc_error status = astcenc_compress_image(
        context, &input_image, &swizzle, compressed.data(), compressed.size(), 0);
    decoded.resize(encoded.texels.size());
    void * decoded_slice = decoded.data();
    astcenc_image decoded_image{ encoded.width, encoded.height, 1, ASTCENC_TYPE_F32, &decoded_slice };
    if (status == ASTCENC_SUCCESS) {
        status = astcenc_decompress_image(
            context, compressed.data(), compressed.size(), &decoded_image, &swizzle, 0);
    }
    astcenc_context_free(context);
    return status == ASTCENC_SUCCESS;
}

std::vector<float> decode_semantics(const encoded_representation & encoded,
                                    const std::vector<float> & decoded,
                                    channel_semantics semantics,
                                    const channel_pairing & pairing,
                                    uint32_t rows, uint32_t columns) {
    const uint32_t values_per_texel = semantics == channel_semantics::independent_rgba ? 4 : 2;
    std::vector<float> result(static_cast<size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            const uint32_t texel_x = column / values_per_texel;
            const uint32_t lane = column % values_per_texel;
            const float * texel = decoded.data() +
                (static_cast<size_t>(row) * encoded.width + texel_x) * 4;
            float reconstructed = 0.0f;
            if (semantics == channel_semantics::independent_rgba) {
                reconstructed = texel[lane] * encoded.range + encoded.minimum;
            } else if (semantics == channel_semantics::coarse_only_pairs ||
                       semantics == channel_semantics::coarse_residual_pairs) {
                const uint32_t coarse_channel = pairing.channels[lane * 2];
                const uint32_t residual_channel = pairing.channels[lane * 2 + 1];
                reconstructed = texel[coarse_channel] * encoded.range + encoded.minimum;
                if (semantics == channel_semantics::coarse_residual_pairs) {
                    reconstructed += (texel[residual_channel] - 0.5f) *
                        2.0f * encoded.residual_scale;
                }
            } else {
                const uint32_t high = static_cast<uint32_t>(std::round(
                    std::clamp(texel[pairing.channels[lane * 2]], 0.0f, 1.0f) * 255.0f));
                const uint32_t low = static_cast<uint32_t>(std::round(
                    std::clamp(texel[pairing.channels[lane * 2 + 1]], 0.0f, 1.0f) * 255.0f));
                reconstructed = (static_cast<float>((high << 8) | low) / 65535.0f) *
                    encoded.range + encoded.minimum;
            }
            result[static_cast<size_t>(row) * columns + column] = reconstructed;
        }
    }
    return result;
}

bool run_semantics(const std::vector<float> & weights,
                   const ggml_vk_astc_format_contract & format,
                   channel_semantics semantics,
                   const channel_pairing & pairing,
                   uint32_t rows, uint32_t columns) {
    const encoded_representation encoded = encode_semantics(weights, semantics, pairing, rows, columns);
    std::vector<float> decoded;
    size_t compressed_bytes = 0;
    if (!astc_roundtrip(encoded, format, decoded, compressed_bytes)) return false;
    const std::vector<float> reconstructed =
        decode_semantics(encoded, decoded, semantics, pairing, rows, columns);
    const double mse = elementwise_mse(weights, reconstructed);
    const double activation_mse = activation_relative_mse(weights, reconstructed, rows, columns);
    std::printf("channel-semantics format=%s mode=%s pairing=%s values-per-texel=%u bytes=%zu bpw=%.5f MSE=%.8g activation-relative-MSE=%.8g\n",
                format.name, semantics_name(semantics), pairing.name,
                semantics == channel_semantics::independent_rgba ? 4u : 2u,
                compressed_bytes, compressed_bytes * 8.0 / weights.size(), mse, activation_mse);
    return std::isfinite(mse) && std::isfinite(activation_mse);
}

} // namespace

int main(int argc, char ** argv) {
    bool all_pair_permutations = false;
    std::string model_path;
    std::string tensor_name;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--all-pair-permutations") {
            all_pair_permutations = true;
        } else if ((option == "--model" || option == "--tensor") && index + 1 < argc) {
            const std::string value = argv[++index];
            if (option == "--model") model_path = value;
            else tensor_name = value;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--all-pair-permutations] [--model path --tensor name]\n",
                         argv[0]);
            return 2;
        }
    }
    if (model_path.empty() != tensor_name.empty()) {
        std::fprintf(stderr, "--model and --tensor must be supplied together\n");
        return 2;
    }
    uint32_t rows = kRows;
    uint32_t columns = kColumns;
    std::vector<float> weights;
    if (model_path.empty()) {
        weights.resize(static_cast<size_t>(rows) * columns);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t column = 0; column < columns; ++column) {
                weights[static_cast<size_t>(row) * columns + column] =
                0.70f * std::sin(0.037f * (row + 1) * (column + 1)) +
                0.15f * std::cos(0.113f * (row + 3) + 0.029f * column);
            }
        }
    } else {
        ggml_vk_astc_loaded_matrix matrix;
        std::string error;
        if (!ggml_vk_astc_load_gguf_matrix(model_path, tensor_name, matrix, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        rows = matrix.rows;
        columns = matrix.columns;
        weights = std::move(matrix.values);
    }
    std::vector<channel_pairing> pairings(kCanonicalPairings.begin(), kCanonicalPairings.end());
    std::vector<std::array<char, 6>> generated_names;
    if (all_pair_permutations) {
        pairings.clear();
        pairings.reserve(24);
        generated_names.reserve(24);
        std::array<uint32_t, 4> permutation{ 0, 1, 2, 3 };
        do {
            generated_names.push_back({
                "RGBA"[permutation[0]], "RGBA"[permutation[1]], '+',
                "RGBA"[permutation[2]], "RGBA"[permutation[3]], '\0',
            });
            pairings.push_back({ generated_names.back().data(), permutation });
        } while (std::next_permutation(permutation.begin(), permutation.end()));
    }
    for (const auto & format : { ggml_vk_astc_4x4_unorm_rgba,
                                 ggml_vk_astc_5x5_unorm_rgba,
                                 ggml_vk_astc_6x6_unorm_rgba }) {
        if (!run_semantics(weights, format, channel_semantics::independent_rgba,
                           kCanonicalPairings.front(), rows, columns)) {
            std::fprintf(stderr, "channel semantic smoke failed\n");
            return 1;
        }
        for (const channel_pairing & pairing : pairings) {
            for (const auto semantics : { channel_semantics::coarse_only_pairs,
                                          channel_semantics::coarse_residual_pairs,
                                          channel_semantics::rg16_control_pairs }) {
                if (!run_semantics(weights, format, semantics, pairing, rows, columns)) {
                    std::fprintf(stderr, "channel semantic smoke failed\n");
                    return 1;
                }
            }
        }
    }
    return 0;
}
