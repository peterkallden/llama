#include <astcenc.h>

#include "astc-vulkan-contract.h"
#include "astc-vulkan-input.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRows = 32;
constexpr uint32_t kColumns = 256;
constexpr uint32_t kDefaultCoarseLevels = 16;
float g_astc_preset = ASTCENC_PRE_THOROUGH;

struct affine_decoder {
    double scale_l = 0.0;
    double scale_a = 0.0;
    double offset = 0.0;
};

struct latent_representation {
    std::vector<float> texels;
    affine_decoder decoder;
};

struct activations {
    uint32_t samples = 4;
    uint32_t columns = 0;
    std::vector<float> values;
};

struct astc_roundtrip_result {
    std::vector<float> texels;
    std::vector<uint8_t> compressed;
    size_t compressed_bytes = 0;
    uint32_t block_count = 0;
    uint32_t dual_plane_blocks = 0;
    uint32_t alpha_dual_plane_blocks = 0;
};

double elementwise_mse(const std::vector<float> & reference,
                       const std::vector<float> & candidate) {
    double sum = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double error = reference[i] - candidate[i];
        sum += error * error;
    }
    return sum / std::max<size_t>(reference.size(), 1);
}

double activation_relative_mse(const std::vector<float> & reference,
                               const std::vector<float> & candidate,
                               uint32_t rows, uint32_t columns,
                               const activations & inputs) {
    double error_sum = 0.0;
    double reference_sum = 0.0;
    for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
        const float * input = inputs.values.data() + static_cast<size_t>(sample) * columns;
        for (uint32_t row = 0; row < rows; ++row) {
            double expected = 0.0;
            double actual = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                const size_t index = static_cast<size_t>(row) * columns + column;
                expected += reference[index] * input[column];
                actual += candidate[index] * input[column];
            }
            const double error = expected - actual;
            error_sum += error * error;
            reference_sum += expected * expected;
        }
    }
    return error_sum / std::max(reference_sum, 1e-12);
}

activations make_default_activations(uint32_t columns) {
    activations result;
    result.columns = columns;
    result.values.resize(static_cast<size_t>(result.samples) * columns);
    for (uint32_t sample = 0; sample < result.samples; ++sample) {
        for (uint32_t column = 0; column < columns; ++column) {
            result.values[static_cast<size_t>(sample) * columns + column] =
                0.5f * std::sin(0.017f * (sample + 1) * (column + 1)) +
                0.2f * std::cos(0.031f * (sample + 2) * (column + 3));
        }
    }
    return result;
}

void limit_activation_samples(activations & inputs, uint32_t maximum_samples) {
    if (maximum_samples == 0 || inputs.samples <= maximum_samples) return;
    inputs.samples = maximum_samples;
    inputs.values.resize(static_cast<size_t>(inputs.samples) * inputs.columns);
}

void crop_activation_columns(activations & inputs, uint32_t columns) {
    if (columns >= inputs.columns) return;
    std::vector<float> cropped(static_cast<size_t>(inputs.samples) * columns);
    for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
        std::copy_n(inputs.values.data() + static_cast<size_t>(sample) * inputs.columns,
                    columns, cropped.data() + static_cast<size_t>(sample) * columns);
    }
    inputs.columns = columns;
    inputs.values = std::move(cropped);
}

template<typename T>
bool write_binary(const std::string & path, const std::vector<T> & values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    return static_cast<bool>(output);
}

std::vector<float> reconstruct(const std::vector<float> & texels,
                               const affine_decoder & decoder) {
    std::vector<float> result(texels.size() / 4);
    for (size_t index = 0; index < result.size(); ++index) {
        const float * texel = texels.data() + index * 4;
        const double luminance = (texel[0] + texel[1] + texel[2]) / 3.0;
        result[index] = static_cast<float>(decoder.scale_l * luminance +
                                           decoder.scale_a * texel[3] + decoder.offset);
    }
    return result;
}

bool astc_decode(const std::vector<uint8_t> & compressed, uint32_t rows, uint32_t columns,
                 const ggml_vk_astc_format_contract & format, std::vector<float> & texels);
std::vector<double> matvec_outputs(const std::vector<float> & weights, uint32_t rows,
                                   uint32_t columns, const activations & inputs);

struct coordinate_result {
    std::vector<float> reconstructed;
    std::vector<uint8_t> compressed;
    uint32_t forward_changes = 0;
    uint32_t reverse_changes = 0;
};

struct astc_candidate {
    const char * name = nullptr;
    std::vector<float> reconstructed;
    std::vector<uint8_t> compressed;
};

// Choose whole, independently decodable ASTC blocks from a legal stream pool.
// Calibration activations are the only selection signal; evaluation is outside.
bool coordinate_select_astc_blocks(const std::vector<float> & reference,
                                   const std::vector<astc_candidate> & candidates,
                                   uint32_t rows, uint32_t columns,
                                   const ggml_vk_astc_format_contract & format,
                                   const activations & calibration,
                                   const affine_decoder & decoder,
                                   coordinate_result & result) {
    if (candidates.empty()) return false;
    const uint32_t blocks_x = (columns + format.block_width - 1) / format.block_width;
    const uint32_t blocks_y = (rows + format.block_height - 1) / format.block_height;
    const std::vector<double> expected = matvec_outputs(reference, rows, columns, calibration);
    std::vector<double> actual = matvec_outputs(candidates.front().reconstructed, rows, columns, calibration);
    std::vector<double> residual(expected.size());
    double residual_error = 0.0;
    for (size_t index = 0; index < residual.size(); ++index) {
        residual[index] = expected[index] - actual[index];
        residual_error += residual[index] * residual[index];
    }
    result.reconstructed = candidates.front().reconstructed;
    result.compressed = candidates.front().compressed;

    auto sweep = [&](bool reverse) {
        uint32_t changes = 0;
        for (uint32_t ordinal = 0; ordinal < blocks_x * blocks_y; ++ordinal) {
            const uint32_t block = reverse ? blocks_x * blocks_y - ordinal - 1 : ordinal;
            const uint32_t row0 = (block / blocks_x) * format.block_height;
            const uint32_t column0 = (block % blocks_x) * format.block_width;
            const uint32_t row_count = std::min(row0 + format.block_height, rows) - row0;
            for (const astc_candidate & candidate : candidates) {
                std::vector<double> delta(static_cast<size_t>(calibration.samples) * row_count);
                double residual_dot_delta = 0.0;
                double delta_norm = 0.0;
                for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                    const float * input = calibration.values.data() + static_cast<size_t>(sample) * columns;
                    for (uint32_t row = row0; row < std::min(row0 + format.block_height, rows); ++row) {
                        double value = 0.0;
                        for (uint32_t column = column0; column < std::min(column0 + format.block_width, columns); ++column) {
                            const size_t weight_index = static_cast<size_t>(row) * columns + column;
                            value += (candidate.reconstructed[weight_index] - result.reconstructed[weight_index]) * input[column];
                        }
                        const size_t index = static_cast<size_t>(sample) * rows + row;
                        delta[static_cast<size_t>(sample) * row_count + row - row0] = value;
                        residual_dot_delta += residual[index] * value;
                        delta_norm += value * value;
                    }
                }
                const double candidate_error = residual_error - 2.0 * residual_dot_delta + delta_norm;
                if (candidate_error + 1e-18 >= residual_error) continue;
                for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                    for (uint32_t row = row0; row < std::min(row0 + format.block_height, rows); ++row) {
                        const size_t index = static_cast<size_t>(sample) * rows + row;
                        residual[index] -= delta[static_cast<size_t>(sample) * row_count + row - row0];
                    }
                }
                residual_error = candidate_error;
                for (uint32_t row = row0; row < std::min(row0 + format.block_height, rows); ++row) {
                    for (uint32_t column = column0; column < std::min(column0 + format.block_width, columns); ++column) {
                        const size_t index = static_cast<size_t>(row) * columns + column;
                        result.reconstructed[index] = candidate.reconstructed[index];
                    }
                }
                std::copy_n(candidate.compressed.data() + static_cast<size_t>(block) * 16, 16,
                            result.compressed.data() + static_cast<size_t>(block) * 16);
                ++changes;
            }
        }
        return changes;
    };
    result.forward_changes = sweep(false);
    result.reverse_changes = sweep(true);
    std::vector<float> texels;
    if (!astc_decode(result.compressed, rows, columns, format, texels)) return false;
    const std::vector<float> verified = reconstruct(texels, decoder);
    double maximum_error = 0.0;
    for (size_t index = 0; index < verified.size(); ++index) {
        maximum_error = std::max(maximum_error,
                                 std::abs(static_cast<double>(verified[index] - result.reconstructed[index])));
    }
    return maximum_error <= 1e-6;
}

bool astc_roundtrip(const std::vector<float> & source, uint32_t rows, uint32_t columns,
                    const ggml_vk_astc_format_contract & format,
                    const affine_decoder * ranking_decoder,
                    astc_roundtrip_result & result) {
    result.compressed_bytes = ggml_vk_astc_image_storage_bytes(format, columns, rows);
    astcenc_config config{};
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    const unsigned int flags = ranking_decoder == nullptr ? 0 : ASTCENC_FLG_MAP_NEURAL_LA;
#else
    if (ranking_decoder != nullptr) return false;
    const unsigned int flags = 0;
#endif
    if (astcenc_config_init(ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
                            g_astc_preset, flags, &config) != ASTCENC_SUCCESS) return false;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (ranking_decoder != nullptr) {
        config.neural_l_scale = static_cast<float>(ranking_decoder->scale_l);
        config.neural_a_scale = static_cast<float>(ranking_decoder->scale_a);
    }
#endif
    astcenc_context * context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (astcenc_context_alloc(&config, 1, &context, nullptr) != ASTCENC_SUCCESS) return false;
#else
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return false;
#endif
    void * source_slice = const_cast<float *>(source.data());
    astcenc_image source_image{ columns, rows, 1, ASTCENC_TYPE_F32, &source_slice };
    const astcenc_swizzle swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A };
    result.compressed.resize(result.compressed_bytes);
    astcenc_error status = astcenc_compress_image(
        context, &source_image, &swizzle, result.compressed.data(), result.compressed.size(), 0);
    if (status == ASTCENC_SUCCESS) {
        result.block_count = static_cast<uint32_t>(result.compressed.size() / 16);
        for (size_t offset = 0; offset < result.compressed.size(); offset += 16) {
            astcenc_block_info info{};
            status = astcenc_get_block_info(context, result.compressed.data() + offset, &info);
            if (status != ASTCENC_SUCCESS) break;
            if (info.is_dual_plane_block) {
                ++result.dual_plane_blocks;
                if (info.dual_plane_component == 3) ++result.alpha_dual_plane_blocks;
            }
        }
    }
    result.texels.resize(source.size());
    void * decoded_slice = result.texels.data();
    astcenc_image decoded_image{ columns, rows, 1, ASTCENC_TYPE_F32, &decoded_slice };
    if (status == ASTCENC_SUCCESS) {
        status = astcenc_decompress_image(
            context, result.compressed.data(), result.compressed.size(), &decoded_image, &swizzle, 0);
    }
    astcenc_context_free(context);
    return status == ASTCENC_SUCCESS;
}

bool astc_decode(const std::vector<uint8_t> & compressed, uint32_t rows, uint32_t columns,
                 const ggml_vk_astc_format_contract & format, std::vector<float> & texels) {
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
                            g_astc_preset, 0, &config) != ASTCENC_SUCCESS) return false;
    astcenc_context * context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (astcenc_context_alloc(&config, 1, &context, nullptr) != ASTCENC_SUCCESS) return false;
#else
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return false;
#endif
    texels.resize(static_cast<size_t>(rows) * columns * 4);
    void * decoded_slice = texels.data();
    astcenc_image image{ columns, rows, 1, ASTCENC_TYPE_F32, &decoded_slice };
    const astcenc_swizzle swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A };
    const astcenc_error status = astcenc_decompress_image(
        context, compressed.data(), compressed.size(), &image, &swizzle, 0);
    astcenc_context_free(context);
    return status == ASTCENC_SUCCESS;
}

std::vector<double> matvec_outputs(const std::vector<float> & weights, uint32_t rows,
                                   uint32_t columns, const activations & inputs) {
    std::vector<double> result(static_cast<size_t>(inputs.samples) * rows);
    for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
        const float * input = inputs.values.data() + static_cast<size_t>(sample) * columns;
        for (uint32_t row = 0; row < rows; ++row) {
            double value = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                value += weights[static_cast<size_t>(row) * columns + column] * input[column];
            }
            result[static_cast<size_t>(sample) * rows + row] = value;
        }
    }
    return result;
}

double relative_output_mse(const std::vector<double> & reference,
                           const std::vector<double> & candidate) {
    double error_sum = 0.0;
    double reference_sum = 0.0;
    for (size_t index = 0; index < reference.size(); ++index) {
        const double error = reference[index] - candidate[index];
        error_sum += error * error;
        reference_sum += reference[index] * reference[index];
    }
    return error_sum / std::max(reference_sum, 1e-12);
}

latent_representation make_scalar_latents(const std::vector<float> & weights,
                                          float minimum, float range) {
    latent_representation result;
    result.texels.resize(weights.size() * 4);
    result.decoder = { range, 0.0, minimum };
    for (size_t index = 0; index < weights.size(); ++index) {
        const float normalized = (weights[index] - minimum) / range;
        std::fill_n(result.texels.data() + index * 4, 4, normalized);
    }
    return result;
}

latent_representation make_row_column_latents(const std::vector<float> & weights,
                                              uint32_t rows, uint32_t columns) {
    std::vector<float> row_means(rows, 0.0f);
    std::vector<float> column_means(columns, 0.0f);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            const float value = weights[static_cast<size_t>(row) * columns + column];
            row_means[row] += value;
            column_means[column] += value;
        }
    }
    for (float & value : row_means) value /= columns;
    for (float & value : column_means) value /= rows;
    const auto [row_min_it, row_max_it] = std::minmax_element(row_means.begin(), row_means.end());
    const auto [column_min_it, column_max_it] =
        std::minmax_element(column_means.begin(), column_means.end());
    const float row_range = std::max(*row_max_it - *row_min_it, 1e-6f);
    const float column_range = std::max(*column_max_it - *column_min_it, 1e-6f);
    latent_representation result;
    result.texels.resize(weights.size() * 4);
    result.decoder = { row_range, column_range, *row_min_it + *column_min_it };
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            float * texel = result.texels.data() +
                (static_cast<size_t>(row) * columns + column) * 4;
            texel[0] = texel[1] = texel[2] =
                (row_means[row] - *row_min_it) / row_range;
            texel[3] = (column_means[column] - *column_min_it) / column_range;
        }
    }
    return result;
}

latent_representation make_additive_latents(const std::vector<float> & weights,
                                            float minimum, float range,
                                            uint32_t rows, uint32_t columns,
                                            uint32_t block, bool block_residual,
                                            uint32_t coarse_levels) {
    latent_representation result;
    result.texels.resize(weights.size() * 4);
    const float step = range / (coarse_levels - 1);
    const float residual_radius = std::max(step * 0.5f, 1e-6f);
    result.decoder = { range, 2.0 * residual_radius, minimum - residual_radius };
    std::vector<float> residuals(weights.size());
    for (size_t index = 0; index < weights.size(); ++index) {
        const float level = std::round((weights[index] - minimum) / step);
        const float coarse = minimum +
            std::clamp(level, 0.0f, static_cast<float>(coarse_levels - 1)) * step;
        residuals[index] = weights[index] - coarse;
    }
    if (block_residual) {
        // A block-constant residual is deliberately low frequency. This is a
        // cheap control for the hypothesis that ASTC can preserve a structured
        // second latent even when it destroys an independent per-value tail.
        for (uint32_t row0 = 0; row0 < rows; row0 += block) {
            for (uint32_t column0 = 0; column0 < columns; column0 += block) {
                double sum = 0.0;
                uint32_t count = 0;
                for (uint32_t row = row0; row < std::min(row0 + block, rows); ++row) {
                    for (uint32_t column = column0;
                         column < std::min(column0 + block, columns); ++column) {
                        sum += residuals[static_cast<size_t>(row) * columns + column];
                        ++count;
                    }
                }
                const float mean = static_cast<float>(sum / std::max(count, 1u));
                for (uint32_t row = row0; row < std::min(row0 + block, rows); ++row) {
                    for (uint32_t column = column0;
                         column < std::min(column0 + block, columns); ++column) {
                        residuals[static_cast<size_t>(row) * columns + column] = mean;
                    }
                }
            }
        }
    }
    for (size_t index = 0; index < weights.size(); ++index) {
        const float level = std::round((weights[index] - minimum) / step);
        const float coarse = minimum +
            std::clamp(level, 0.0f, static_cast<float>(coarse_levels - 1)) * step;
        const float l = (coarse - minimum) / range;
        const float a = std::clamp(0.5f + residuals[index] / (2.0f * residual_radius),
                                   0.0f, 1.0f);
        float * texel = result.texels.data() + index * 4;
        texel[0] = l;
        texel[1] = l;
        texel[2] = l;
        texel[3] = a;
    }
    return result;
}

double run_case(const char * name, const std::vector<float> & weights,
                const latent_representation & latents, uint32_t rows, uint32_t columns,
                const ggml_vk_astc_format_contract & format,
                const activations & inputs, const activations * selection_inputs = nullptr,
                bool neural_rank = false) {
    astc_roundtrip_result roundtrip;
    const affine_decoder * ranking_decoder = neural_rank ? &latents.decoder : nullptr;
    if (!astc_roundtrip(latents.texels, rows, columns, format, ranking_decoder, roundtrip)) return NAN;
    const std::vector<float> reconstructed = reconstruct(roundtrip.texels, latents.decoder);
    const double mse = elementwise_mse(weights, reconstructed);
    const double activation_mse = activation_relative_mse(
        weights, reconstructed, rows, columns, inputs);
    std::printf("latent format=%s mode=%s bytes=%zu bpw=%.5f dual-plane=%u/%u alpha-plane=%u "
                "sL=%.8g sA=%.8g b=%.8g MSE=%.8g activation-relative-MSE=%.8g\n",
                format.name, name, roundtrip.compressed_bytes,
                roundtrip.compressed_bytes * 8.0 / weights.size(),
                roundtrip.dual_plane_blocks, roundtrip.block_count,
                roundtrip.alpha_dual_plane_blocks, latents.decoder.scale_l, latents.decoder.scale_a,
                latents.decoder.offset, mse, activation_mse);
    double score = activation_mse;
    if (selection_inputs != nullptr) {
        score = activation_relative_mse(weights, reconstructed, rows, columns,
                                        *selection_inputs);
        std::printf("latent-selection mode=%s calibration-relative-MSE=%.8g\n",
                    name, score);
    }
    return std::isfinite(mse) && std::isfinite(activation_mse) ? score : NAN;
}

bool run_coordinate_case(const std::vector<float> & weights,
                         const latent_representation & latents, uint32_t rows, uint32_t columns,
                         const ggml_vk_astc_format_contract & format,
                         const activations & calibration, const activations & holdout,
                         bool include_fast_candidate) {
    astc_roundtrip_result standard;
    astc_roundtrip_result neural;
    if (!astc_roundtrip(latents.texels, rows, columns, format, nullptr, standard) ||
        !astc_roundtrip(latents.texels, rows, columns, format, &latents.decoder, neural)) return false;
    const std::vector<float> standard_weights = reconstruct(standard.texels, latents.decoder);
    const std::vector<float> neural_weights = reconstruct(neural.texels, latents.decoder);
    const double standard_calibration = activation_relative_mse(
        weights, standard_weights, rows, columns, calibration);
    const double standard_holdout = activation_relative_mse(
        weights, standard_weights, rows, columns, holdout);
    const double neural_calibration = activation_relative_mse(
        weights, neural_weights, rows, columns, calibration);
    const double neural_holdout = activation_relative_mse(
        weights, neural_weights, rows, columns, holdout);
    std::vector<astc_candidate> candidates = {
        { "standard", standard_weights, standard.compressed },
        { "neural-rank", neural_weights, neural.compressed },
    };
    if (include_fast_candidate) {
        const float saved_preset = g_astc_preset;
        g_astc_preset = ASTCENC_PRE_FAST;
        astc_roundtrip_result fast;
        const bool encoded = astc_roundtrip(latents.texels, rows, columns, format, &latents.decoder, fast);
        g_astc_preset = saved_preset;
        if (!encoded) return false;
        candidates.push_back({ "neural-rank-fast", reconstruct(fast.texels, latents.decoder),
                               std::move(fast.compressed) });
    }
    coordinate_result selected;
    if (!coordinate_select_astc_blocks(weights, candidates,
                                       rows, columns, format, calibration, latents.decoder, selected)) return false;
    const double calibration_mse = activation_relative_mse(
        weights, selected.reconstructed, rows, columns, calibration);
    const double holdout_mse = activation_relative_mse(
        weights, selected.reconstructed, rows, columns, holdout);
    std::printf("latent-coordinate format=%s mode=luminance-alpha-additive "
                "standard-calibration=%.8g standard-holdout=%.8g "
                "neural-calibration=%.8g neural-holdout=%.8g "
                "calibration-relative-MSE=%.8g holdout-relative-MSE=%.8g "
                "forward-changes=%u reverse-changes=%u candidates=%zu calibration-samples=%u holdout-samples=%u\n",
                format.name, standard_calibration, standard_holdout, neural_calibration, neural_holdout,
                calibration_mse, holdout_mse,
                selected.forward_changes, selected.reverse_changes, candidates.size(), calibration.samples, holdout.samples);
    return std::isfinite(calibration_mse) && std::isfinite(holdout_mse);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string tensor_name;
    std::string trace_path;
    std::string calibration_trace_path;
    bool search_levels = false;
    bool neural_rank = false;
    bool coordinate_select = false;
    bool coordinate_only = false;
    bool coordinate_fast_candidate = false;
    std::string footprint;
    std::string preset = "thorough";
    uint32_t maximum_samples = 0;
    uint32_t maximum_rows = 0;
    uint32_t maximum_columns = 0;
    std::string export_astc_path;
    std::string export_reference_path;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--search-levels") {
            search_levels = true;
        } else if (option == "--neural-rank") {
            neural_rank = true;
        } else if (option == "--coordinate-select") {
            coordinate_select = true;
        } else if (option == "--coordinate-only") {
            coordinate_only = true;
        } else if (option == "--coordinate-fast-candidate") {
            coordinate_fast_candidate = true;
        } else if (option == "--footprint" && index + 1 < argc) {
            footprint = argv[++index];
        } else if (option == "--preset" && index + 1 < argc) {
            preset = argv[++index];
        } else if (option == "--max-samples" && index + 1 < argc) {
            maximum_samples = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (option == "--max-rows" && index + 1 < argc) {
            maximum_rows = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (option == "--max-columns" && index + 1 < argc) {
            maximum_columns = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if ((option == "--export-astc" || option == "--export-reference") && index + 1 < argc) {
            const std::string value = argv[++index];
            if (option == "--export-astc") export_astc_path = value;
            else export_reference_path = value;
        } else if ((option == "--model" || option == "--tensor" || option == "--trace" ||
                    option == "--calibration-trace") &&
                   index + 1 < argc) {
            const std::string value = argv[++index];
            if (option == "--model") model_path = value;
            else if (option == "--tensor") tensor_name = value;
            else if (option == "--trace") trace_path = value;
            else calibration_trace_path = value;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--search-levels] [--neural-rank] [--coordinate-select] [--coordinate-only] [--coordinate-fast-candidate] "
                         "[--footprint 4x4|5x5|6x6] [--preset thorough|medium|fast] [--model path --tensor name] "
                         "[--trace path] [--calibration-trace path] [--max-samples N] [--max-rows N] [--max-columns N]\n",
                         argv[0]);
            return 2;
        }
    }
    if (model_path.empty() != tensor_name.empty()) {
        std::fprintf(stderr, "--model and --tensor must be supplied together\n");
        return 2;
    }
    if (coordinate_select && !neural_rank) {
        std::fprintf(stderr, "--coordinate-select requires --neural-rank\n");
        return 2;
    }
    if (coordinate_only && !coordinate_select) {
        std::fprintf(stderr, "--coordinate-only requires --coordinate-select\n");
        return 2;
    }
    if (coordinate_fast_candidate && !coordinate_select) {
        std::fprintf(stderr, "--coordinate-fast-candidate requires --coordinate-select\n");
        return 2;
    }
    if (export_astc_path.empty() != export_reference_path.empty() ||
        (!export_astc_path.empty() && footprint.empty())) {
        std::fprintf(stderr, "exports require --export-astc, --export-reference, and --footprint\n");
        return 2;
    }
    if (preset == "medium") {
        g_astc_preset = ASTCENC_PRE_MEDIUM;
    } else if (preset == "fast") {
        g_astc_preset = ASTCENC_PRE_FAST;
    } else if (preset != "thorough") {
        std::fprintf(stderr, "unsupported --preset value: %s\n", preset.c_str());
        return 2;
    }
#if !defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (neural_rank) {
        std::fprintf(stderr, "this build does not include the experimental neural astcenc fork\n");
        return 2;
    }
#endif

    uint32_t rows = kRows;
    uint32_t columns = kColumns;
    std::vector<float> weights;
    std::string error;
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
        if (!ggml_vk_astc_load_gguf_matrix(model_path, tensor_name, matrix, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        rows = matrix.rows;
        columns = matrix.columns;
        weights = std::move(matrix.values);
    }

    const uint32_t source_rows = rows;
    const uint32_t trace_columns = columns;
    const uint32_t selected_rows = maximum_rows == 0 ? rows : std::min(rows, maximum_rows);
    const uint32_t selected_columns = maximum_columns == 0 ? columns : std::min(columns, maximum_columns);
    if (selected_rows != rows || selected_columns != columns) {
        std::vector<float> cropped(static_cast<size_t>(selected_rows) * selected_columns);
        for (uint32_t row = 0; row < selected_rows; ++row) {
            std::copy_n(weights.data() + static_cast<size_t>(row) * columns, selected_columns,
                        cropped.data() + static_cast<size_t>(row) * selected_columns);
        }
        rows = selected_rows;
        columns = selected_columns;
        weights = std::move(cropped);
        std::printf("latent-submatrix rows=%u columns=%u source-rows=%u source-columns=%u\n",
                    rows, columns, source_rows, trace_columns);
    }

    const auto [minimum_it, maximum_it] = std::minmax_element(weights.begin(), weights.end());
    const float minimum = *minimum_it;
    const float range = std::max(*maximum_it - minimum, 1e-6f);
    const latent_representation scalar_latents = make_scalar_latents(weights, minimum, range);
    const latent_representation row_column_latents = make_row_column_latents(weights, rows, columns);
    activations inputs = make_default_activations(columns);
    activations calibration_inputs = inputs;
    if (!trace_path.empty()) {
        ggml_vk_astc_activation_trace loaded;
        if (!ggml_vk_astc_load_activation_trace(trace_path, loaded, error) ||
            loaded.columns != trace_columns) {
            std::fprintf(stderr, "invalid activation trace: %s\n", error.c_str());
            return 1;
        }
        inputs.samples = loaded.samples;
        inputs.values = std::move(loaded.values);
    }
    if (calibration_trace_path.empty()) calibration_inputs = inputs;
    if (!calibration_trace_path.empty()) {
        ggml_vk_astc_activation_trace loaded;
        if (!ggml_vk_astc_load_activation_trace(calibration_trace_path, loaded, error) ||
            loaded.columns != trace_columns) {
            std::fprintf(stderr, "invalid calibration activation trace: %s\n", error.c_str());
            return 1;
        }
        calibration_inputs.samples = loaded.samples;
        calibration_inputs.values = std::move(loaded.values);
    }
    limit_activation_samples(inputs, maximum_samples);
    limit_activation_samples(calibration_inputs, maximum_samples);
    crop_activation_columns(inputs, columns);
    crop_activation_columns(calibration_inputs, columns);
    for (const auto & format : { ggml_vk_astc_4x4_unorm_rgba,
                                 ggml_vk_astc_5x5_unorm_rgba,
                                 ggml_vk_astc_6x6_unorm_rgba }) {
        const std::string format_footprint = std::to_string(format.block_width) + "x" +
                                             std::to_string(format.block_height);
        if (!footprint.empty() && footprint != format_footprint) continue;
        const latent_representation additive_latents = make_additive_latents(
            weights, minimum, range, rows, columns, format.block_width, false,
            kDefaultCoarseLevels);
        const latent_representation block_latents = make_additive_latents(
            weights, minimum, range, rows, columns, format.block_width, true,
            kDefaultCoarseLevels);
        if (!export_astc_path.empty()) {
            astc_roundtrip_result exported;
            if (!astc_roundtrip(additive_latents.texels, rows, columns, format, nullptr, exported) ||
                !write_binary(export_astc_path, exported.compressed) ||
                !write_binary(export_reference_path, exported.texels)) {
                std::fprintf(stderr, "ASTC latent export failed\n");
                return 1;
            }
            std::printf("latent-export format=%s bytes=%zu texels=%zu astc=%s reference=%s\n",
                        format.name, exported.compressed.size(), exported.texels.size(),
                        export_astc_path.c_str(), export_reference_path.c_str());
        }
        if (!coordinate_only &&
            (!std::isfinite(run_case("scalar-rgba", weights, scalar_latents,
                                    rows, columns, format, inputs)) ||
            !std::isfinite(run_case("row-column-additive", weights, row_column_latents,
                                    rows, columns, format, inputs)) ||
            !std::isfinite(run_case("luminance-alpha-additive", weights, additive_latents,
                                    rows, columns, format, inputs)) ||
            !std::isfinite(run_case("luminance-alpha-block-residual", weights, block_latents,
                                    rows, columns, format, inputs)))) {
            std::fprintf(stderr, "ASTC latent smoke failed\n");
            return 1;
        }
        if (!coordinate_only && neural_rank &&
            (!std::isfinite(run_case("scalar-rgba-neural-rank", weights, scalar_latents,
                                     rows, columns, format, inputs, nullptr, true)) ||
             !std::isfinite(run_case("luminance-alpha-additive-neural-rank", weights,
                                     additive_latents, rows, columns, format, inputs,
                                     nullptr, true)))) {
            std::fprintf(stderr, "ASTC latent neural-rank smoke failed\n");
            return 1;
        }
        if (coordinate_select &&
            !run_coordinate_case(weights, additive_latents, rows, columns, format,
                                 calibration_inputs, inputs, coordinate_fast_candidate)) {
            std::fprintf(stderr, "ASTC coordinate selection smoke failed\n");
            return 1;
        }
        if (!coordinate_only && search_levels) {
            double best_score = INFINITY;
            uint32_t best_levels = 0;
            for (const uint32_t coarse_levels : { 3u, 5u, 8u, 16u, 32u }) {
                const latent_representation candidate = make_additive_latents(
                    weights, minimum, range, rows, columns, format.block_width, false,
                    coarse_levels);
                char name[64];
                std::snprintf(name, sizeof(name), "projection-levels-%u", coarse_levels);
                const double score = run_case(name, weights, candidate, rows, columns,
                                              format, inputs, &calibration_inputs, neural_rank);
                if (!std::isfinite(score)) {
                    std::fprintf(stderr, "ASTC latent projection search failed\n");
                    return 1;
                }
                if (score < best_score) {
                    best_score = score;
                    best_levels = coarse_levels;
                }
            }
            std::printf("latent-selection format=%s best-coarse-levels=%u "
                        "calibration-relative-MSE=%.8g\n",
                        format.name, best_levels, best_score);
        }
    }
    if (!footprint.empty() && footprint != "4x4" && footprint != "5x5" && footprint != "6x6") {
        std::fprintf(stderr, "unsupported --footprint value: %s\n", footprint.c_str());
        return 2;
    }
    return 0;
}
