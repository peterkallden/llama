#include <astcenc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include "astc-vulkan-tensor-contract.h"

namespace {

constexpr float kWeightMin = -1.0f;
constexpr float kWeightScale = 2.0f;
constexpr uint32_t kRows = 12;
constexpr uint32_t kColumns = 48;
constexpr uint32_t kLargeRows = 64;
constexpr uint32_t kLargeColumns = 256;
constexpr double kElementwiseLossWeight = 1.0;
constexpr double kActivationLossWeight = 1.0;
constexpr double kBlockTailLossWeight = 0.25;

struct encoder_quality {
    const char * name;
    float value;
};

const std::array<encoder_quality, 3> kEncoderQualities{{
    { "fast", ASTCENC_PRE_FAST },
    { "medium", ASTCENC_PRE_MEDIUM },
    { "thorough", ASTCENC_PRE_THOROUGH },
}};

struct roundtrip_result {
    double mse = 0.0;
    double activation_mse = 0.0;
    double objective = 0.0;
    double max_block_mse = 0.0;
    double p95_block_mse = 0.0;
    float max_error = 0.0f;
    double dot_error = 0.0;
    std::array<uint32_t, 4> channel_order{ 0, 1, 2, 3 };
    const char * layout_name = "identity";
    const char * quality_name = "medium";
};

std::vector<uint32_t> identity_texel_order(const ggml_vk_astc_weight_layout & layout) {
    std::vector<uint32_t> order(layout.texel_count());
    std::iota(order.begin(), order.end(), 0);
    return order;
}

std::vector<uint32_t> grouped_texel_order(const ggml_vk_astc_weight_layout & layout,
                                          const std::vector<float> & weights) {
    const uint32_t texel_columns = layout.texel_columns();
    std::vector<float> row_scores(layout.rows, 0.0f);
    std::vector<float> group_scores(texel_columns, 0.0f);
    for (uint32_t row = 0; row < layout.rows; ++row) {
        for (uint32_t column = 0; column < layout.columns; ++column) {
            const float magnitude = std::fabs(weights[static_cast<size_t>(row) * layout.columns + column]);
            row_scores[row] += magnitude;
            group_scores[column / 4] += magnitude;
        }
    }
    std::vector<uint32_t> row_order(layout.rows);
    std::vector<uint32_t> group_order(texel_columns);
    std::iota(row_order.begin(), row_order.end(), 0);
    std::iota(group_order.begin(), group_order.end(), 0);
    std::sort(row_order.begin(), row_order.end(), [&](uint32_t lhs, uint32_t rhs) {
        return row_scores[lhs] < row_scores[rhs];
    });
    std::sort(group_order.begin(), group_order.end(), [&](uint32_t lhs, uint32_t rhs) {
        return group_scores[lhs] < group_scores[rhs];
    });

    std::vector<uint32_t> order(layout.texel_count());
    for (uint32_t encoded_row = 0; encoded_row < layout.rows; ++encoded_row) {
        for (uint32_t encoded_group = 0; encoded_group < texel_columns; ++encoded_group) {
            const uint32_t logical_row = row_order[encoded_row];
            const uint32_t logical_group = group_order[encoded_group];
            order[static_cast<size_t>(encoded_row) * texel_columns + encoded_group] =
                logical_row * texel_columns + logical_group;
        }
    }
    return order;
}

bool encode_roundtrip(const ggml_vk_astc_format_contract & format,
                      const ggml_vk_astc_weight_layout & layout,
                      const std::vector<float> & weights,
                      const std::vector<std::vector<float>> & activation_samples,
                      const std::array<uint32_t, 4> & channel_order,
                      const std::vector<uint32_t> & texel_order,
                      const encoder_quality & quality,
                      roundtrip_result & result) {
    const unsigned int width = layout.texel_columns();
    const unsigned int height = layout.rows;
    const size_t texel_components = static_cast<size_t>(width) * height * 4;
    std::vector<float> texels(texel_components);
    for (uint32_t encoded_row = 0; encoded_row < layout.rows; ++encoded_row) {
        for (uint32_t encoded_texel = 0; encoded_texel < width; ++encoded_texel) {
            const size_t encoded_index = static_cast<size_t>(encoded_row) * width + encoded_texel;
            const uint32_t logical_index = texel_order[encoded_index];
            const uint32_t logical_row = logical_index / width;
            const uint32_t logical_texel = logical_index % width;
            const size_t texel_offset = encoded_index * 4;
            const size_t weight_offset =
                static_cast<size_t>(logical_row) * layout.columns + logical_texel * 4;
            for (uint32_t channel = 0; channel < 4; ++channel) {
                texels[texel_offset + channel] =
                    (weights[weight_offset + channel_order[channel]] - kWeightMin) /
                    kWeightScale;
            }
        }
    }

    astcenc_config config{};
    astcenc_error status = astcenc_config_init(
        ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
        quality.value, 0, &config);
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

    std::vector<uint32_t> inverse_texel_order(texel_order.size());
    for (size_t encoded_index = 0; encoded_index < texel_order.size(); ++encoded_index) {
        inverse_texel_order[texel_order[encoded_index]] = static_cast<uint32_t>(encoded_index);
    }
    std::vector<float> reconstructed_weights(weights.size());
    const uint32_t blocks_x = ggml_vk_astc_block_count(width, format.block_width);
    const uint32_t blocks_y = ggml_vk_astc_block_count(height, format.block_height);
    std::vector<double> block_squared_error(static_cast<size_t>(blocks_x) * blocks_y, 0.0);
    std::vector<uint32_t> block_value_count(block_squared_error.size(), 0);
    double squared_error = 0.0;
    for (uint32_t row = 0; row < layout.rows; ++row) {
        for (uint32_t column = 0; column < layout.columns; ++column) {
            const size_t weight_index = static_cast<size_t>(row) * layout.columns + column;
            const uint32_t logical_channel = column % 4;
            const auto decoded_channel = std::find(
                channel_order.begin(), channel_order.end(), logical_channel);
            const uint32_t logical_texel = row * width + column / 4;
            const uint32_t encoded_texel = inverse_texel_order[logical_texel];
            const size_t texel_index = static_cast<size_t>(encoded_texel) * 4 +
                static_cast<size_t>(decoded_channel - channel_order.begin());
            const float reconstructed = decoded[texel_index] * kWeightScale + kWeightMin;
            reconstructed_weights[weight_index] = reconstructed;
            const float error = std::fabs(weights[weight_index] - reconstructed);
            squared_error += static_cast<double>(error) * error;
            result.max_error = std::max(result.max_error, error);
            const uint32_t encoded_x = encoded_texel % width;
            const uint32_t encoded_y = encoded_texel / width;
            const size_t block_index = static_cast<size_t>(encoded_y / format.block_height) * blocks_x +
                encoded_x / format.block_width;
            block_squared_error[block_index] += static_cast<double>(error) * error;
            ++block_value_count[block_index];
        }
    }
    result.mse = squared_error / weights.size();
    std::vector<double> block_mse(block_squared_error.size());
    for (size_t i = 0; i < block_mse.size(); ++i) {
        block_mse[i] = block_squared_error[i] / block_value_count[i];
    }
    result.max_block_mse = *std::max_element(block_mse.begin(), block_mse.end());
    std::sort(block_mse.begin(), block_mse.end());
    const size_t p95_index = std::min(block_mse.size() - 1, (block_mse.size() * 95) / 100);
    result.p95_block_mse = block_mse[p95_index];

    double activation_squared_error = 0.0;
    for (const std::vector<float> & activations : activation_samples) {
        for (uint32_t row = 0; row < layout.rows; ++row) {
            double reference_dot = 0.0;
            double reconstructed_dot = 0.0;
            for (uint32_t column = 0; column < layout.columns; ++column) {
                const size_t index = static_cast<size_t>(row) * layout.columns + column;
                reference_dot += weights[index] * activations[column];
                reconstructed_dot += reconstructed_weights[index] * activations[column];
            }
            const double error = reference_dot - reconstructed_dot;
            activation_squared_error += error * error;
        }
    }
    result.activation_mse = activation_squared_error /
        static_cast<double>(activation_samples.size() * layout.rows);
    result.objective = kElementwiseLossWeight * result.mse +
        kActivationLossWeight * result.activation_mse +
        kBlockTailLossWeight * result.max_block_mse;

    double reference_dot = 0.0;
    double reconstructed_dot = 0.0;
    const std::vector<float> & first_activations = activation_samples.front();
    for (size_t i = 0; i < weights.size(); ++i) {
        reference_dot += weights[i] * first_activations[i % layout.columns];
        reconstructed_dot += reconstructed_weights[i] * first_activations[i % layout.columns];
    }
    result.dot_error = std::fabs(reference_dot - reconstructed_dot);
    return std::isfinite(result.mse) && std::isfinite(result.max_block_mse) &&
        std::isfinite(result.p95_block_mse) && std::isfinite(result.activation_mse) &&
        std::isfinite(result.objective) && std::isfinite(result.dot_error);
}

bool search_channel_orders(const ggml_vk_astc_format_contract & format,
                           const ggml_vk_astc_weight_layout & layout,
                           const std::vector<float> & weights,
                           const std::vector<std::vector<float>> & activation_samples,
                           const char * layout_name,
                           const std::vector<uint32_t> & texel_order,
                           const encoder_quality & quality,
                           roundtrip_result & best_result) {
    std::array<uint32_t, 4> channel_order{ 0, 1, 2, 3 };
    bool found_result = false;
    do {
        roundtrip_result candidate;
        if (!encode_roundtrip(format, layout, weights, activation_samples,
                              channel_order, texel_order, quality, candidate)) {
            return false;
        }
        if (!found_result || candidate.objective < best_result.objective ||
            (candidate.objective == best_result.objective && candidate.mse < best_result.mse)) {
            best_result = candidate;
            best_result.channel_order = channel_order;
            best_result.layout_name = layout_name;
            best_result.quality_name = quality.name;
            found_result = true;
        }
    } while (std::next_permutation(channel_order.begin(), channel_order.end()));

    std::printf(
        "ASTC weight %s %s/%s order %u%u%u%u: %zu bytes, MSE %.8f, block P95/max %.8f/%.8f, activation MSE %.8f, objective %.8f, max error %.6f, dot error %.6f\n",
        format.name, best_result.layout_name, best_result.quality_name,
        best_result.channel_order[0],
        best_result.channel_order[1],
        best_result.channel_order[2], best_result.channel_order[3],
        layout.storage_bytes(format), best_result.mse, best_result.p95_block_mse,
        best_result.max_block_mse, best_result.activation_mse, best_result.objective,
        best_result.max_error, best_result.dot_error);
    return true;
}

bool search_layout_qualities(const ggml_vk_astc_format_contract & format,
                             const ggml_vk_astc_weight_layout & layout,
                             const std::vector<float> & weights,
                             const std::vector<std::vector<float>> & activation_samples,
                             const char * layout_name,
                             const std::vector<uint32_t> & texel_order,
                             roundtrip_result & best_result) {
    bool found_result = false;
    for (const encoder_quality & quality : kEncoderQualities) {
        roundtrip_result candidate;
        if (!search_channel_orders(format, layout, weights, activation_samples,
                                   layout_name, texel_order, quality, candidate)) {
            return false;
        }
        if (!found_result || candidate.objective < best_result.objective ||
            (candidate.objective == best_result.objective && candidate.mse < best_result.mse)) {
            best_result = candidate;
            found_result = true;
        }
    }
    return found_result;
}

} // namespace

int main(int argc, char ** argv) {
    bool large_fixture = false;
    if (argc == 2 && std::strcmp(argv[1], "--large") == 0) {
        large_fixture = true;
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--large]\n", argv[0]);
        return 2;
    }
    const uint32_t rows = large_fixture ? kLargeRows : kRows;
    const uint32_t columns = large_fixture ? kLargeColumns : kColumns;
    const ggml_vk_astc_weight_layout layout{ rows, columns, 4 };
    if (!layout.is_valid()) {
        std::fprintf(stderr, "invalid weight layout\n");
        return 1;
    }
    std::vector<float> weights(static_cast<size_t>(rows) * columns);
    std::vector<std::vector<float>> activation_samples(6, std::vector<float>(columns));
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            weights[static_cast<size_t>(row) * columns + column] =
                0.75f * std::sin(0.17f * (row + 1) * (column + 1));
        }
    }
    for (uint32_t column = 0; column < columns; ++column) {
        activation_samples[0][column] = 0.5f * std::cos(0.11f * (column + 1));
        activation_samples[1][column] = 0.5f * std::sin(0.07f * (column + 3));
        activation_samples[2][column] = 0.25f * std::cos(0.19f * (column + 5));
        activation_samples[3][column] = (static_cast<int>(column % 7) - 3) * 0.1f;
        activation_samples[4][column] = 0.0f;
        activation_samples[5][column] = 0.0f;
    }
    uint32_t random_state = 0x9e3779b9u;
    for (uint32_t column = 0; column < columns; ++column) {
        random_state = random_state * 1664525u + 1013904223u;
        activation_samples[4][column] =
            (static_cast<float>(random_state >> 8) / 16777215.0f - 0.5f) * 0.8f;
        if (column % 13 == 0) {
            activation_samples[5][column] = (column % 26 == 0) ? 1.0f : -1.0f;
        }
    }

    const auto identity_order = identity_texel_order(layout);
    const auto grouped_order = grouped_texel_order(layout, weights);
    roundtrip_result format_4x4_identity;
    roundtrip_result format_4x4_grouped;
    roundtrip_result format_6x6_identity;
    roundtrip_result format_6x6_grouped;
    if (!search_layout_qualities(ggml_vk_astc_4x4_unorm_rgba, layout,
                                 weights, activation_samples, "identity", identity_order,
                               format_4x4_identity) ||
        !search_layout_qualities(ggml_vk_astc_4x4_unorm_rgba, layout,
                                 weights, activation_samples, "grouped", grouped_order,
                               format_4x4_grouped) ||
        !search_layout_qualities(ggml_vk_astc_6x6_unorm_rgba, layout,
                                 weights, activation_samples, "identity", identity_order,
                               format_6x6_identity) ||
        !search_layout_qualities(ggml_vk_astc_6x6_unorm_rgba, layout,
                                 weights, activation_samples, "grouped", grouped_order,
                               format_6x6_grouped)) {
        std::fprintf(stderr, "ASTC weight smoke failed\n");
        return 1;
    }
    const roundtrip_result & best_4x4 =
        format_4x4_grouped.objective < format_4x4_identity.objective ? format_4x4_grouped : format_4x4_identity;
    const roundtrip_result & best_6x6 =
        format_6x6_grouped.objective < format_6x6_identity.objective ? format_6x6_grouped : format_6x6_identity;
    std::printf("ASTC weight selected 4x4 %s/%s, 6x6 %s/%s\n",
                best_4x4.layout_name, best_4x4.quality_name,
                best_6x6.layout_name, best_6x6.quality_name);
    std::printf("ASTC weight smoke passed\n");
    return 0;
}
