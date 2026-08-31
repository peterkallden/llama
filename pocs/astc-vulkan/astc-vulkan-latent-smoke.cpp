#include <astcenc.h>

#include "astc-vulkan-contract.h"
#include "astc-vulkan-input.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRows = 32;
constexpr uint32_t kColumns = 256;
constexpr uint32_t kDefaultCoarseLevels = 16;

struct affine_decoder {
    double scale_l = 0.0;
    double scale_a = 0.0;
    double offset = 0.0;
};

struct activations {
    uint32_t samples = 4;
    uint32_t columns = 0;
    std::vector<float> values;
};

struct astc_roundtrip_result {
    std::vector<float> texels;
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

bool solve_3x3(double matrix[3][3], double rhs[3], affine_decoder & result) {
    for (uint32_t column = 0; column < 3; ++column) {
        uint32_t pivot = column;
        for (uint32_t row = column + 1; row < 3; ++row) {
            if (std::fabs(matrix[row][column]) > std::fabs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::fabs(matrix[pivot][column]) < 1e-12) return false;
        for (uint32_t entry = column; entry < 3; ++entry) {
            std::swap(matrix[column][entry], matrix[pivot][entry]);
        }
        std::swap(rhs[column], rhs[pivot]);
        const double inverse = 1.0 / matrix[column][column];
        for (uint32_t entry = column; entry < 3; ++entry) matrix[column][entry] *= inverse;
        rhs[column] *= inverse;
        for (uint32_t row = 0; row < 3; ++row) {
            if (row == column) continue;
            const double factor = matrix[row][column];
            for (uint32_t entry = column; entry < 3; ++entry) {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    result.scale_l = rhs[0];
    result.scale_a = rhs[1];
    result.offset = rhs[2];
    return true;
}

affine_decoder fit_affine_decoder(const std::vector<float> & weights,
                                  const std::vector<float> & texels) {
    double normal[3][3]{};
    double rhs[3]{};
    for (size_t index = 0; index < weights.size(); ++index) {
        const float * texel = texels.data() + index * 4;
        // RGB stores the first (luminance-like) latent. Alpha stores the
        // second latent. Averaging RGB makes the contract robust to minor
        // encoder channel asymmetry while preserving the intended semantics.
        const double features[3] = {
            (texel[0] + texel[1] + texel[2]) / 3.0,
            texel[3],
            1.0,
        };
        for (uint32_t row = 0; row < 3; ++row) {
            rhs[row] += features[row] * weights[index];
            for (uint32_t column = 0; column < 3; ++column) {
                normal[row][column] += features[row] * features[column];
            }
        }
    }
    affine_decoder result;
    return solve_3x3(normal, rhs, result) ? result : affine_decoder{};
}

affine_decoder fit_scalar_decoder(const std::vector<float> & weights,
                                  const std::vector<float> & texels) {
    double sum_l = 0.0;
    double sum_ll = 0.0;
    double sum_w = 0.0;
    double sum_lw = 0.0;
    for (size_t index = 0; index < weights.size(); ++index) {
        const float * texel = texels.data() + index * 4;
        const double luminance = (texel[0] + texel[1] + texel[2]) / 3.0;
        sum_l += luminance;
        sum_ll += luminance * luminance;
        sum_w += weights[index];
        sum_lw += luminance * weights[index];
    }
    const double count = weights.size();
    const double determinant = sum_ll * count - sum_l * sum_l;
    if (std::fabs(determinant) < 1e-12) return {};
    return {
        (sum_lw * count - sum_l * sum_w) / determinant,
        0.0,
        (sum_ll * sum_w - sum_l * sum_lw) / determinant,
    };
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

bool astc_roundtrip(const std::vector<float> & source, uint32_t rows, uint32_t columns,
                    const ggml_vk_astc_format_contract & format,
                    astc_roundtrip_result & result) {
    result.compressed_bytes = ggml_vk_astc_image_storage_bytes(format, columns, rows);
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
                            ASTCENC_PRE_THOROUGH, 0, &config) != ASTCENC_SUCCESS) return false;
    astcenc_context * context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return false;
    void * source_slice = const_cast<float *>(source.data());
    astcenc_image source_image{ columns, rows, 1, ASTCENC_TYPE_F32, &source_slice };
    const astcenc_swizzle swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A };
    std::vector<uint8_t> compressed(result.compressed_bytes);
    astcenc_error status = astcenc_compress_image(
        context, &source_image, &swizzle, compressed.data(), compressed.size(), 0);
    if (status == ASTCENC_SUCCESS) {
        result.block_count = static_cast<uint32_t>(compressed.size() / 16);
        for (size_t offset = 0; offset < compressed.size(); offset += 16) {
            astcenc_block_info info{};
            status = astcenc_get_block_info(context, compressed.data() + offset, &info);
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
            context, compressed.data(), compressed.size(), &decoded_image, &swizzle, 0);
    }
    astcenc_context_free(context);
    return status == ASTCENC_SUCCESS;
}

std::vector<float> make_scalar_latents(const std::vector<float> & weights,
                                       float minimum, float range) {
    std::vector<float> result(weights.size() * 4);
    for (size_t index = 0; index < weights.size(); ++index) {
        const float normalized = (weights[index] - minimum) / range;
        std::fill_n(result.data() + index * 4, 4, normalized);
    }
    return result;
}

std::vector<float> make_row_column_latents(const std::vector<float> & weights,
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
    std::vector<float> result(weights.size() * 4);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            float * texel = result.data() +
                (static_cast<size_t>(row) * columns + column) * 4;
            texel[0] = texel[1] = texel[2] =
                (row_means[row] - *row_min_it) / row_range;
            texel[3] = (column_means[column] - *column_min_it) / column_range;
        }
    }
    return result;
}

std::vector<float> make_additive_latents(const std::vector<float> & weights,
                                         float minimum, float range,
                                         uint32_t rows, uint32_t columns,
                                         uint32_t block, bool block_residual,
                                         uint32_t coarse_levels) {
    std::vector<float> result(weights.size() * 4);
    const float step = range / (coarse_levels - 1);
    const float residual_radius = std::max(step * 0.5f, 1e-6f);
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
        float * texel = result.data() + index * 4;
        texel[0] = l;
        texel[1] = l;
        texel[2] = l;
        texel[3] = a;
    }
    return result;
}

double run_case(const char * name, const std::vector<float> & weights,
                const std::vector<float> & latents, uint32_t rows, uint32_t columns,
                const ggml_vk_astc_format_contract & format,
                const activations & inputs, const activations * selection_inputs = nullptr) {
    astc_roundtrip_result roundtrip;
    if (!astc_roundtrip(latents, rows, columns, format, roundtrip)) return NAN;
    const affine_decoder decoder = std::string(name) == "scalar-rgba"
        ? fit_scalar_decoder(weights, roundtrip.texels)
        : fit_affine_decoder(weights, roundtrip.texels);
    const std::vector<float> reconstructed = reconstruct(roundtrip.texels, decoder);
    const double mse = elementwise_mse(weights, reconstructed);
    const double activation_mse = activation_relative_mse(
        weights, reconstructed, rows, columns, inputs);
    std::printf("latent format=%s mode=%s bytes=%zu bpw=%.5f dual-plane=%u/%u alpha-plane=%u "
                "sL=%.8g sA=%.8g b=%.8g MSE=%.8g activation-relative-MSE=%.8g\n",
                format.name, name, roundtrip.compressed_bytes,
                roundtrip.compressed_bytes * 8.0 / weights.size(),
                roundtrip.dual_plane_blocks, roundtrip.block_count,
                roundtrip.alpha_dual_plane_blocks, decoder.scale_l, decoder.scale_a,
                decoder.offset, mse, activation_mse);
    double score = activation_mse;
    if (selection_inputs != nullptr) {
        score = activation_relative_mse(weights, reconstructed, rows, columns,
                                        *selection_inputs);
        std::printf("latent-selection mode=%s calibration-relative-MSE=%.8g\n",
                    name, score);
    }
    return std::isfinite(mse) && std::isfinite(activation_mse) ? score : NAN;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string tensor_name;
    std::string trace_path;
    std::string calibration_trace_path;
    bool search_levels = false;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--search-levels") {
            search_levels = true;
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
                         "usage: %s [--search-levels] [--model path --tensor name] "
                         "[--trace path] [--calibration-trace path]\n",
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

    const auto [minimum_it, maximum_it] = std::minmax_element(weights.begin(), weights.end());
    const float minimum = *minimum_it;
    const float range = std::max(*maximum_it - minimum, 1e-6f);
    const std::vector<float> scalar_latents = make_scalar_latents(weights, minimum, range);
    const std::vector<float> row_column_latents = make_row_column_latents(weights, rows, columns);
    activations inputs = make_default_activations(columns);
    activations calibration_inputs = inputs;
    if (!trace_path.empty()) {
        ggml_vk_astc_activation_trace loaded;
        if (!ggml_vk_astc_load_activation_trace(trace_path, loaded, error) ||
            loaded.columns != columns) {
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
            loaded.columns != columns) {
            std::fprintf(stderr, "invalid calibration activation trace: %s\n", error.c_str());
            return 1;
        }
        calibration_inputs.samples = loaded.samples;
        calibration_inputs.values = std::move(loaded.values);
    }
    for (const auto & format : { ggml_vk_astc_4x4_unorm_rgba,
                                 ggml_vk_astc_5x5_unorm_rgba,
                                 ggml_vk_astc_6x6_unorm_rgba }) {
        const std::vector<float> additive_latents = make_additive_latents(
            weights, minimum, range, rows, columns, format.block_width, false,
            kDefaultCoarseLevels);
        const std::vector<float> block_latents = make_additive_latents(
            weights, minimum, range, rows, columns, format.block_width, true,
            kDefaultCoarseLevels);
        if (!std::isfinite(run_case("scalar-rgba", weights, scalar_latents,
                                    rows, columns, format, inputs)) ||
            !std::isfinite(run_case("row-column-additive", weights, row_column_latents,
                                    rows, columns, format, inputs)) ||
            !std::isfinite(run_case("luminance-alpha-additive", weights, additive_latents,
                                    rows, columns, format, inputs)) ||
            !std::isfinite(run_case("luminance-alpha-block-residual", weights, block_latents,
                                    rows, columns, format, inputs))) {
            std::fprintf(stderr, "ASTC latent smoke failed\n");
            return 1;
        }
        if (search_levels) {
            double best_score = INFINITY;
            uint32_t best_levels = 0;
            for (const uint32_t coarse_levels : { 3u, 5u, 8u, 16u, 32u }) {
                const std::vector<float> candidate = make_additive_latents(
                    weights, minimum, range, rows, columns, format.block_width, false,
                    coarse_levels);
                char name[64];
                std::snprintf(name, sizeof(name), "projection-levels-%u", coarse_levels);
                const double score = run_case(name, weights, candidate, rows, columns,
                                              format, inputs, &calibration_inputs);
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
    return 0;
}
