#include <astcenc.h>

#include "astc-vulkan-contract.h"
#include "astc-vulkan-gauge.h"
#include "astc-vulkan-hash.h"
#include "astc-vulkan-block-ldlq.h"
#include "astc-vulkan-pv.h"
#include "astc-vulkan-d1-prescreen.h"
#if defined(ASTC_VULKAN_D1_PRESCREEN_GPU)
#include "astc-vulkan-d1-prescreen-dispatch.h"
#endif
#include "astc-vulkan-input.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <cerrno>
#include <cstring>
#include <vector>

namespace {

std::pair<uint32_t, uint32_t> d1_prescreen_dimensions(astc_vulkan_footprint footprint) {
    switch (footprint) {
    case astc_vulkan_footprint::k4x4: return {4, 4};
    case astc_vulkan_footprint::k5x5: return {5, 5};
    case astc_vulkan_footprint::k6x6: return {6, 6};
    case astc_vulkan_footprint::k8x6: return {8, 6};
    case astc_vulkan_footprint::k10x6: return {10, 6};
    case astc_vulkan_footprint::k8x8: return {8, 8};
    case astc_vulkan_footprint::k10x8: return {10, 8};
    default: return {1, 1};
    }
}

constexpr uint32_t kRows = 32;
constexpr uint32_t kColumns = 256;
constexpr uint32_t kDefaultCoarseLevels = 16;
float g_astc_preset = ASTCENC_PRE_THOROUGH;
double g_block_ldlq_damping = 1e-4;
enum class residual_basis { free, block_constant, block_row, block_column, block_plane };
const char * astc_preset_name() {
    if (g_astc_preset == ASTCENC_PRE_FAST) return "fast";
    if (g_astc_preset == ASTCENC_PRE_MEDIUM) return "medium";
    return "thorough";
}

template<typename T>
std::string vector_hash(const std::vector<T> & values) {
    return astc_vulkan_fnv1a64_tagged(values.data(), values.size() * sizeof(T));
}
// Keep the reference encoder path separate from the experimental recall path.
// The latter changes only which legal astcenc candidates reach the existing
// exact-decode selector; it does not change the ASTC payload format or runtime
// reconstruction contract.
enum class encoder_search_mode { standard, neural };
astc_vulkan_ldlq_order g_block_ldlq_order = astc_vulkan_ldlq_order::forward;
bool g_directional_shortlists = false;
uint32_t g_stability_shards = 2;

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

std::vector<double> matvec_outputs(const std::vector<float> & weights, uint32_t rows,
                                   uint32_t columns, const activations & inputs);

struct captured_astc_candidate {
    std::array<uint8_t, 16> block{};
    float error = 0.0f;
};

struct astc_candidate_capture {
    uint32_t blocks_x = 0;
    uint32_t blocks_y = 0;
    uint32_t blocks_z = 1;
    uint32_t block_width = 1;
    uint32_t block_height = 1;
    uint32_t block_depth = 1;
    uint32_t max_per_block = 4;
    uint32_t callback_count = 0;
    std::vector<std::vector<captured_astc_candidate>> blocks;
};

#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
void capture_astc_candidate(void * user_data, unsigned int pos_x, unsigned int pos_y,
                            unsigned int pos_z, const uint8_t block_data[16], float errorval,
                            unsigned int, int) {
    auto * capture = static_cast<astc_candidate_capture *>(user_data);
    if (capture == nullptr || capture->blocks_x == 0 || capture->blocks_y == 0) return;
    const uint32_t block_x = pos_x / capture->block_width;
    const uint32_t block_y = pos_y / capture->block_height;
    const uint32_t block_z = pos_z / capture->block_depth;
    const size_t block_index = (static_cast<size_t>(block_z) * capture->blocks_y + block_y) *
                               capture->blocks_x + block_x;
    if (block_index >= capture->blocks.size()) return;
    ++capture->callback_count;
    auto & candidates = capture->blocks[block_index];
    const auto duplicate = std::find_if(candidates.begin(), candidates.end(),
        [&](const captured_astc_candidate & candidate) {
            return std::equal(candidate.block.begin(), candidate.block.end(), block_data);
        });
    if (duplicate != candidates.end()) return;
    captured_astc_candidate candidate;
    std::copy_n(block_data, 16, candidate.block.begin());
    candidate.error = errorval;
    candidates.push_back(candidate);
    std::sort(candidates.begin(), candidates.end(),
              [](const captured_astc_candidate & left, const captured_astc_candidate & right) {
                  return left.error < right.error;
              });
    if (candidates.size() > capture->max_per_block) candidates.pop_back();
}
#endif

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

activations slice_activation_samples(const activations & inputs,
                                     uint32_t first_sample, uint32_t sample_count) {
    activations result;
    result.samples = std::min(sample_count, inputs.samples - std::min(first_sample, inputs.samples));
    result.columns = inputs.columns;
    const size_t offset = static_cast<size_t>(std::min(first_sample, inputs.samples)) * inputs.columns;
    result.values.assign(inputs.values.begin() + offset,
                         inputs.values.begin() + offset + static_cast<size_t>(result.samples) * result.columns);
    return result;
}

void print_calibration_shard_summary(const char * selector,
                                     const std::vector<float> & reference,
                                     const std::vector<float> & candidate,
                                     uint32_t rows, uint32_t columns,
                                     const activations & calibration,
                                     uint32_t shard_count = 4) {
    if (calibration.samples < shard_count || shard_count == 0) return;
    const uint32_t shard_size = (calibration.samples + shard_count - 1) / shard_count;
    std::vector<double> losses;
    for (uint32_t first = 0; first < calibration.samples; first += shard_size) {
        const uint32_t count = std::min(shard_size, calibration.samples - first);
        losses.push_back(activation_relative_mse(reference, candidate, rows, columns,
                                                  slice_activation_samples(calibration, first, count)));
    }
    double mean = 0.0;
    for (double loss : losses) mean += loss;
    mean /= std::max<size_t>(losses.size(), 1);
    double variance = 0.0;
    for (double loss : losses) variance += (loss - mean) * (loss - mean);
    variance /= std::max<size_t>(losses.size(), 1);
    std::printf("latent-calibration-shards selector=%s count=%zu mean=%.8g std=%.8g min=%.8g max=%.8g\n",
                selector, losses.size(), mean, std::sqrt(variance),
                *std::min_element(losses.begin(), losses.end()),
                *std::max_element(losses.begin(), losses.end()));
}

struct hessian_stats {
    uint32_t rank = 0;
    double maximum_eigenvalue = 0.0;
    double minimum_positive_eigenvalue = 0.0;
    double damped_condition = 0.0;
};

// Estimate the input Hessian H_I = X^T X on small probes. The Jacobi sweep is
// intentionally diagnostic, not a production factorization; large layers use
// the sample-count rank bound and skip the cubic eigensolve.
hessian_stats estimate_hessian_stats(const activations & inputs) {
    hessian_stats result;
    const uint32_t columns = inputs.columns;
    if (columns == 0 || inputs.samples == 0) return result;
    if (columns > 128) {
        result.rank = std::min(inputs.samples, columns);
        return result;
    }
    std::vector<double> gram(static_cast<size_t>(columns) * columns, 0.0);
    for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
        const float * row = inputs.values.data() + static_cast<size_t>(sample) * columns;
        for (uint32_t i = 0; i < columns; ++i) {
            for (uint32_t j = 0; j <= i; ++j) {
                gram[static_cast<size_t>(i) * columns + j] += row[i] * row[j];
            }
        }
    }
    for (uint32_t i = 0; i < columns; ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            gram[static_cast<size_t>(j) * columns + i] = gram[static_cast<size_t>(i) * columns + j];
        }
    }
    for (uint32_t iteration = 0; iteration < 32 * columns; ++iteration) {
        uint32_t p = 0;
        uint32_t q = 0;
        double largest = 0.0;
        for (uint32_t i = 0; i < columns; ++i) {
            for (uint32_t j = i + 1; j < columns; ++j) {
                const double value = std::abs(gram[static_cast<size_t>(i) * columns + j]);
                if (value > largest) {
                    largest = value;
                    p = i;
                    q = j;
                }
            }
        }
        if (largest < 1e-12) break;
        const double app = gram[static_cast<size_t>(p) * columns + p];
        const double aqq = gram[static_cast<size_t>(q) * columns + q];
        const double apq = gram[static_cast<size_t>(p) * columns + q];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (uint32_t k = 0; k < columns; ++k) {
            const double gpk = gram[static_cast<size_t>(p) * columns + k];
            const double gqk = gram[static_cast<size_t>(q) * columns + k];
            gram[static_cast<size_t>(p) * columns + k] = cosine * gpk - sine * gqk;
            gram[static_cast<size_t>(q) * columns + k] = sine * gpk + cosine * gqk;
        }
        for (uint32_t k = 0; k < columns; ++k) {
            const double gkp = gram[static_cast<size_t>(k) * columns + p];
            const double gkq = gram[static_cast<size_t>(k) * columns + q];
            gram[static_cast<size_t>(k) * columns + p] = cosine * gkp - sine * gkq;
            gram[static_cast<size_t>(k) * columns + q] = sine * gkp + cosine * gkq;
        }
    }
    std::vector<double> eigenvalues(columns);
    for (uint32_t i = 0; i < columns; ++i) {
        eigenvalues[i] = std::max(gram[static_cast<size_t>(i) * columns + i], 0.0);
        result.maximum_eigenvalue = std::max(result.maximum_eigenvalue, eigenvalues[i]);
    }
    const double threshold = result.maximum_eigenvalue * 1e-8;
    double minimum = INFINITY;
    for (double eigenvalue : eigenvalues) {
        if (eigenvalue > threshold) {
            ++result.rank;
            minimum = std::min(minimum, eigenvalue);
        }
    }
    result.minimum_positive_eigenvalue = std::isfinite(minimum) ? minimum : 0.0;
    const double damping = std::max(result.maximum_eigenvalue * 1e-4, 1e-12);
    result.damped_condition = (result.maximum_eigenvalue + damping) / damping;
    return result;
}

template<typename T>
bool write_binary(const std::string & path, const std::vector<T> & values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::fprintf(stderr, "cannot open binary output '%s': %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!output) {
        std::fprintf(stderr, "cannot write binary output '%s' (%zu bytes): %s\n",
                     path.c_str(), values.size() * sizeof(T), std::strerror(errno));
        return false;
    }
    return true;
}

bool write_export_metadata(const std::string & path, const ggml_vk_astc_format_contract & format,
                           const char * mode, uint32_t rows, uint32_t columns,
                           const affine_decoder & decoder,
                           size_t compressed_bytes, size_t decoded_texels,
                           const char * encoder_profile = "standard",
                           uint32_t validation_prefix = 0,
                           const char * candidate_family = "none",
                           const char * source_hash = "unspecified",
                           const char * calibration_hash = "unspecified",
                           const char * validation_hash = "unspecified",
                           const char * holdout_hash = "unspecified",
                           const char * objective = "activation-relative-mse",
                           const char * optimizer = "none") {
    std::ofstream file(path);
    if (!file) return false;
    file << "version=1\n"
         << "format=" << format.name << "\n"
         << "mode=" << mode << "\n"
         << "rows=" << rows << "\n"
         << "columns=" << columns << "\n"
         << "block_width=" << format.block_width << "\n"
         << "block_height=" << format.block_height << "\n"
         << "encoder_profile=" << encoder_profile << "\n"
         << "candidate_family=" << candidate_family << "\n"
         << "objective=" << objective << "\n"
         << "optimizer=" << optimizer << "\n"
         << "validation_prefix=" << validation_prefix << "\n"
         << "astc_preset=" << astc_preset_name() << "\n"
         << "source_hash=" << source_hash << "\n"
         << "calibration_hash=" << calibration_hash << "\n"
         << "validation_hash=" << validation_hash << "\n"
         << "holdout_hash=" << holdout_hash << "\n"
         << "scale_l=" << decoder.scale_l << "\n"
         << "scale_a=" << decoder.scale_a << "\n"
         << "offset=" << decoder.offset << "\n"
         << "compressed_bytes=" << compressed_bytes << "\n"
         << "decoded_texels=" << decoded_texels << "\n";
    return static_cast<bool>(file);
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
    std::vector<uint32_t> selected_indices;
    uint32_t forward_changes = 0;
    uint32_t reverse_changes = 0;
};

struct astc_candidate {
    const char * name = nullptr;
    std::vector<float> reconstructed;
    std::vector<uint8_t> compressed;
    int32_t source_block = -1;
};

bool candidate_allowed_for_block(const astc_candidate & candidate, uint32_t block) {
    return candidate.source_block < 0 || candidate.source_block == static_cast<int32_t>(block);
}

struct feedback_proposal {
    uint32_t block = 0;
    uint32_t candidate = 0;
    double predicted_gain = 0.0;
};

void apply_block_candidate(const astc_candidate & candidate, uint32_t block,
                           uint32_t blocks_x, uint32_t rows, uint32_t columns,
                           const ggml_vk_astc_format_contract & format,
                           std::vector<float> & reconstructed,
                           std::vector<uint8_t> & compressed) {
    const uint32_t row0 = (block / blocks_x) * format.block_height;
    const uint32_t column0 = (block % blocks_x) * format.block_width;
    const uint32_t row_end = std::min(row0 + format.block_height, rows);
    const uint32_t column_end = std::min(column0 + format.block_width, columns);
    for (uint32_t row = row0; row < row_end; ++row) {
        for (uint32_t column = column0; column < column_end; ++column) {
            const size_t index = static_cast<size_t>(row) * columns + column;
            reconstructed[index] = candidate.reconstructed[index];
        }
    }
    std::copy_n(candidate.compressed.data() + static_cast<size_t>(block) * 16, 16,
                compressed.data() + static_cast<size_t>(block) * 16);
}

double block_activation_error(const std::vector<float> & reference,
                              const astc_candidate & candidate,
                              uint32_t block, uint32_t blocks_x,
                              uint32_t rows, uint32_t columns,
                              const ggml_vk_astc_format_contract & format,
                              const activations & inputs) {
    const uint32_t row0 = (block / blocks_x) * format.block_height;
    const uint32_t column0 = (block % blocks_x) * format.block_width;
    const uint32_t row_end = std::min(row0 + format.block_height, rows);
    const uint32_t column_end = std::min(column0 + format.block_width, columns);
    double error = 0.0;
    for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
        const float * input = inputs.values.data() + static_cast<size_t>(sample) * columns;
        for (uint32_t row = row0; row < row_end; ++row) {
            double delta = 0.0;
            for (uint32_t column = column0; column < column_end; ++column) {
                const size_t index = static_cast<size_t>(row) * columns + column;
                delta += (candidate.reconstructed[index] - reference[index]) * input[column];
            }
            error += delta * delta;
        }
    }
    return error;
}

double block_output_gain(const std::vector<double> & residual,
                         const astc_candidate & from, const astc_candidate & to,
                         uint32_t block, uint32_t blocks_x,
                         uint32_t rows, uint32_t columns,
                         const ggml_vk_astc_format_contract & format,
                         const activations & inputs) {
    const uint32_t row0 = (block / blocks_x) * format.block_height;
    const uint32_t row_end = std::min(row0 + format.block_height, rows);
    const uint32_t column0 = (block % blocks_x) * format.block_width;
    const uint32_t column_end = std::min(column0 + format.block_width, columns);
    double residual_dot_delta = 0.0;
    double delta_norm = 0.0;
    for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
        const float * input = inputs.values.data() + static_cast<size_t>(sample) * columns;
        for (uint32_t row = row0; row < row_end; ++row) {
            double delta = 0.0;
            for (uint32_t column = column0; column < column_end; ++column) {
                const size_t index = static_cast<size_t>(row) * columns + column;
                delta += (to.reconstructed[index] - from.reconstructed[index]) * input[column];
            }
            const size_t output_index = static_cast<size_t>(sample) * rows + row;
            residual_dot_delta += residual[output_index] * delta;
            delta_norm += delta * delta;
        }
    }
    return 2.0 * residual_dot_delta - delta_norm;
}

void subtract_block_delta(std::vector<double> & residual,
                          const astc_candidate & from, const astc_candidate & to,
                          uint32_t block, uint32_t blocks_x,
                          uint32_t rows, uint32_t columns,
                          const ggml_vk_astc_format_contract & format,
                          const activations & inputs) {
    const uint32_t row0 = (block / blocks_x) * format.block_height;
    const uint32_t row_end = std::min(row0 + format.block_height, rows);
    const uint32_t column0 = (block % blocks_x) * format.block_width;
    const uint32_t column_end = std::min(column0 + format.block_width, columns);
    for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
        const float * input = inputs.values.data() + static_cast<size_t>(sample) * columns;
        for (uint32_t row = row0; row < row_end; ++row) {
            double delta = 0.0;
            for (uint32_t column = column0; column < column_end; ++column) {
                const size_t index = static_cast<size_t>(row) * columns + column;
                delta += (to.reconstructed[index] - from.reconstructed[index]) * input[column];
            }
            residual[static_cast<size_t>(sample) * rows + row] -= delta;
        }
    }
}

bool local_select_astc_blocks(const std::vector<float> & reference,
                              const std::vector<astc_candidate> & candidates,
                              uint32_t rows, uint32_t columns,
                              const ggml_vk_astc_format_contract & format,
                              const activations & calibration,
                              coordinate_result & result,
                              const std::vector<std::vector<uint32_t>> * shortlists) {
    if (candidates.empty()) return false;
    const uint32_t blocks_x = (columns + format.block_width - 1) / format.block_width;
    const uint32_t blocks_y = (rows + format.block_height - 1) / format.block_height;
    const uint32_t block_count = blocks_x * blocks_y;
    if (shortlists != nullptr && shortlists->size() != block_count) return false;
    result.reconstructed = candidates.front().reconstructed;
    result.compressed = candidates.front().compressed;
    result.selected_indices.assign(block_count, 0);
    for (uint32_t block = 0; block < block_count; ++block) {
        const std::vector<uint32_t> fallback = [&]() {
            std::vector<uint32_t> all(candidates.size());
            for (uint32_t index = 0; index < candidates.size(); ++index) all[index] = index;
            return all;
        }();
        const std::vector<uint32_t> & candidate_indices = shortlists == nullptr ?
            fallback : (*shortlists)[block];
        double best_error = INFINITY;
        uint32_t best_index = 0;
        for (uint32_t candidate_index : candidate_indices) {
            if (candidate_index >= candidates.size()) return false;
            if (!candidate_allowed_for_block(candidates[candidate_index], block)) continue;
            const double error = block_activation_error(reference, candidates[candidate_index],
                                                        block, blocks_x, rows, columns,
                                                        format, calibration);
            if (error < best_error) {
                best_error = error;
                best_index = candidate_index;
            }
        }
        result.selected_indices[block] = best_index;
        apply_block_candidate(candidates[best_index], block, blocks_x, rows, columns,
                              format, result.reconstructed, result.compressed);
    }
    return true;
}

// Select from one fixed legal pool using simultaneous residual-feedback rounds.
// Each round reads the same residual for every block, so the decisions are
// independent and can be dispatched in parallel. Cross-block interactions are
// intentionally omitted within a round; this is the fixed-pool Hessian/Jacobi
// approximation that will be compared against sequential coordinate descent.
bool hessian_feedback_select_astc_blocks(
        const std::vector<float> & reference,
        const std::vector<astc_candidate> & candidates,
        uint32_t rows, uint32_t columns,
        const ggml_vk_astc_format_contract & format,
        const activations & calibration,
        coordinate_result & result,
        const std::vector<std::vector<uint32_t>> * shortlists,
        uint32_t rounds = 3) {
    if (!local_select_astc_blocks(reference, candidates, rows, columns, format,
                                  calibration, result, shortlists)) return false;
    const uint32_t blocks_x = (columns + format.block_width - 1) / format.block_width;
    const uint32_t blocks_y = (rows + format.block_height - 1) / format.block_height;
    const uint32_t block_count = blocks_x * blocks_y;
    const std::vector<double> expected = matvec_outputs(reference, rows, columns, calibration);
    for (uint32_t round = 0; round < rounds; ++round) {
        const std::vector<double> actual = matvec_outputs(result.reconstructed, rows, columns, calibration);
        std::vector<double> residual(expected.size());
        for (size_t index = 0; index < residual.size(); ++index) residual[index] = expected[index] - actual[index];
        std::vector<uint32_t> next_indices = result.selected_indices;
        for (uint32_t block = 0; block < block_count; ++block) {
            const uint32_t row0 = (block / blocks_x) * format.block_height;
            const uint32_t row_end = std::min(row0 + format.block_height, rows);
            const uint32_t row_count = row_end - row0;
            const std::vector<uint32_t> fallback = [&]() {
                std::vector<uint32_t> all(candidates.size());
                for (uint32_t index = 0; index < candidates.size(); ++index) all[index] = index;
                return all;
            }();
            const std::vector<uint32_t> & candidate_indices = shortlists == nullptr ?
                fallback : (*shortlists)[block];
            double best_error = INFINITY;
            uint32_t best_index = result.selected_indices[block];
            for (uint32_t candidate_index : candidate_indices) {
                if (candidate_index >= candidates.size()) return false;
                if (!candidate_allowed_for_block(candidates[candidate_index], block)) continue;
                const astc_candidate & candidate = candidates[candidate_index];
                const uint32_t column0 = (block % blocks_x) * format.block_width;
                const uint32_t column_end = std::min(column0 + format.block_width, columns);
                double residual_dot_delta = 0.0;
                double delta_norm = 0.0;
                for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                    const float * input = calibration.values.data() + static_cast<size_t>(sample) * columns;
                    for (uint32_t row = row0; row < row_end; ++row) {
                        double delta = 0.0;
                        for (uint32_t column = column0; column < column_end; ++column) {
                            const size_t index = static_cast<size_t>(row) * columns + column;
                            delta += (candidate.reconstructed[index] -
                                      candidates[result.selected_indices[block]].reconstructed[index]) * input[column];
                        }
                        const size_t output_index = static_cast<size_t>(sample) * rows + row;
                        residual_dot_delta += residual[output_index] * delta;
                        delta_norm += delta * delta;
                    }
                }
                const double error = -2.0 * residual_dot_delta + delta_norm;
                if (error < best_error) {
                    best_error = error;
                    best_index = candidate_index;
                }
            }
            next_indices[block] = best_index;
        }
        uint32_t changes = 0;
        result.reconstructed = candidates.front().reconstructed;
        result.compressed = candidates.front().compressed;
        for (uint32_t block = 0; block < block_count; ++block) {
            changes += next_indices[block] != result.selected_indices[block] ? 1u : 0u;
            result.selected_indices[block] = next_indices[block];
            apply_block_candidate(candidates[next_indices[block]], block, blocks_x, rows, columns,
                                  format, result.reconstructed, result.compressed);
        }
        result.forward_changes += changes;
        if (changes == 0) break;
    }
    return true;
}

// Keep proposal generation parallel-friendly, but serialize only the cheap
// acceptance pass. Candidates are never regenerated and every accepted choice
// is checked against the current residual, preventing simultaneous overshoot.
bool conflict_aware_select_astc_blocks(
        const std::vector<float> & reference,
        const std::vector<astc_candidate> & candidates,
        uint32_t rows, uint32_t columns,
        const ggml_vk_astc_format_contract & format,
        const activations & calibration,
        coordinate_result & result,
        const std::vector<std::vector<uint32_t>> * shortlists) {
    if (!local_select_astc_blocks(reference, candidates, rows, columns, format,
                                  calibration, result, shortlists)) return false;
    const uint32_t blocks_x = (columns + format.block_width - 1) / format.block_width;
    const uint32_t blocks_y = (rows + format.block_height - 1) / format.block_height;
    const uint32_t block_count = blocks_x * blocks_y;
    const std::vector<double> expected = matvec_outputs(reference, rows, columns, calibration);
    std::vector<double> actual = matvec_outputs(result.reconstructed, rows, columns, calibration);
    std::vector<double> residual(expected.size());
    for (size_t index = 0; index < residual.size(); ++index) {
        residual[index] = expected[index] - actual[index];
    }
    std::vector<feedback_proposal> proposals;
    proposals.reserve(block_count);
    for (uint32_t block = 0; block < block_count; ++block) {
        const uint32_t row0 = (block / blocks_x) * format.block_height;
        const uint32_t row_end = std::min(row0 + format.block_height, rows);
        const uint32_t column0 = (block % blocks_x) * format.block_width;
        const uint32_t column_end = std::min(column0 + format.block_width, columns);
        const std::vector<uint32_t> fallback = [&]() {
            std::vector<uint32_t> all(candidates.size());
            for (uint32_t index = 0; index < candidates.size(); ++index) all[index] = index;
            return all;
        }();
        const std::vector<uint32_t> & candidate_indices = shortlists == nullptr ?
            fallback : (*shortlists)[block];
        feedback_proposal best;
        best.block = block;
        best.candidate = result.selected_indices[block];
        best.predicted_gain = 0.0;
        for (uint32_t candidate_index : candidate_indices) {
            if (candidate_index >= candidates.size()) return false;
            if (!candidate_allowed_for_block(candidates[candidate_index], block)) continue;
            double residual_dot_delta = 0.0;
            double delta_norm = 0.0;
            for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                const float * input = calibration.values.data() + static_cast<size_t>(sample) * columns;
                for (uint32_t row = row0; row < row_end; ++row) {
                    double delta = 0.0;
                    for (uint32_t column = column0; column < column_end; ++column) {
                        const size_t index = static_cast<size_t>(row) * columns + column;
                        delta += (candidates[candidate_index].reconstructed[index] -
                                  candidates[result.selected_indices[block]].reconstructed[index]) * input[column];
                    }
                    const size_t output_index = static_cast<size_t>(sample) * rows + row;
                    residual_dot_delta += residual[output_index] * delta;
                    delta_norm += delta * delta;
                }
            }
            const double gain = 2.0 * residual_dot_delta - delta_norm;
            if (gain > best.predicted_gain) {
                best.candidate = candidate_index;
                best.predicted_gain = gain;
            }
        }
        if (best.predicted_gain > 0.0) proposals.push_back(best);
    }
    std::sort(proposals.begin(), proposals.end(), [](const feedback_proposal & left,
                                                     const feedback_proposal & right) {
        return left.predicted_gain > right.predicted_gain;
    });
    for (const feedback_proposal & proposal : proposals) {
        const uint32_t block = proposal.block;
        const uint32_t row0 = (block / blocks_x) * format.block_height;
        const uint32_t row_end = std::min(row0 + format.block_height, rows);
        const uint32_t column0 = (block % blocks_x) * format.block_width;
        const uint32_t column_end = std::min(column0 + format.block_width, columns);
        double residual_dot_delta = 0.0;
        double delta_norm = 0.0;
        for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
            const float * input = calibration.values.data() + static_cast<size_t>(sample) * columns;
            for (uint32_t row = row0; row < row_end; ++row) {
                double delta = 0.0;
                for (uint32_t column = column0; column < column_end; ++column) {
                    const size_t index = static_cast<size_t>(row) * columns + column;
                    delta += (candidates[proposal.candidate].reconstructed[index] -
                              result.reconstructed[index]) * input[column];
                }
                const size_t output_index = static_cast<size_t>(sample) * rows + row;
                residual_dot_delta += residual[output_index] * delta;
                delta_norm += delta * delta;
            }
        }
        const double gain = 2.0 * residual_dot_delta - delta_norm;
        if (gain <= 1e-18) continue;
        for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
            const float * input = calibration.values.data() + static_cast<size_t>(sample) * columns;
            for (uint32_t row = row0; row < row_end; ++row) {
                double delta = 0.0;
                for (uint32_t column = column0; column < column_end; ++column) {
                    const size_t index = static_cast<size_t>(row) * columns + column;
                    delta += (candidates[proposal.candidate].reconstructed[index] -
                              result.reconstructed[index]) * input[column];
                }
                residual[static_cast<size_t>(sample) * rows + row] -= delta;
            }
        }
        result.selected_indices[block] = proposal.candidate;
        apply_block_candidate(candidates[proposal.candidate], block, blocks_x, rows, columns,
                              format, result.reconstructed, result.compressed);
        ++result.forward_changes;
    }
    return true;
}

// Gate the same fixed-pool proposals by two calibration shards. A proposal
// must improve the normalized residual energy in every shard, which targets
// calibration-specific directions without changing ASTC candidate generation.
bool stability_select_astc_blocks(
        const std::vector<float> & reference,
        const std::vector<astc_candidate> & candidates,
        uint32_t rows, uint32_t columns,
        const ggml_vk_astc_format_contract & format,
        const activations & calibration,
        coordinate_result & result,
        const std::vector<std::vector<uint32_t>> * shortlists) {
    if (calibration.samples < g_stability_shards || g_stability_shards < 2) return false;
    if (!local_select_astc_blocks(reference, candidates, rows, columns, format,
                                  calibration, result, shortlists)) return false;
    const uint32_t blocks_x = (columns + format.block_width - 1) / format.block_width;
    const uint32_t blocks_y = (rows + format.block_height - 1) / format.block_height;
    const uint32_t block_count = blocks_x * blocks_y;
    const uint32_t shard_count = std::min(g_stability_shards, calibration.samples);
    const uint32_t shard_size = (calibration.samples + shard_count - 1) / shard_count;
    std::vector<activations> shards;
    std::vector<std::vector<double>> residuals(shard_count);
    std::vector<double> residual_energy(shard_count, 0.0);
    for (uint32_t shard = 0; shard < shard_count; ++shard) {
        const uint32_t first = shard * shard_size;
        const uint32_t count = std::min(shard_size, calibration.samples - first);
        shards.push_back(slice_activation_samples(calibration, first, count));
        const std::vector<double> shard_expected = matvec_outputs(reference, rows, columns, shards[shard]);
        const std::vector<double> shard_actual = matvec_outputs(result.reconstructed, rows, columns, shards[shard]);
        residuals[shard].resize(shard_expected.size());
        for (size_t index = 0; index < residuals[shard].size(); ++index) {
            residuals[shard][index] = shard_expected[index] - shard_actual[index];
            residual_energy[shard] += residuals[shard][index] * residuals[shard][index];
        }
    }
    std::vector<feedback_proposal> proposals;
    proposals.reserve(block_count);
    for (uint32_t block = 0; block < block_count; ++block) {
        const std::vector<uint32_t> fallback = [&]() {
            std::vector<uint32_t> all(candidates.size());
            for (uint32_t index = 0; index < candidates.size(); ++index) all[index] = index;
            return all;
        }();
        const std::vector<uint32_t> & candidate_indices = shortlists == nullptr ? fallback : (*shortlists)[block];
        feedback_proposal best;
        best.block = block;
        best.candidate = result.selected_indices[block];
        best.predicted_gain = 0.0;
        for (uint32_t candidate_index : candidate_indices) {
            if (candidate_index >= candidates.size()) return false;
            if (!candidate_allowed_for_block(candidates[candidate_index], block)) continue;
            double score = INFINITY;
            for (uint32_t shard = 0; shard < shard_count; ++shard) {
                const double gain = block_output_gain(
                    residuals[shard], candidates[result.selected_indices[block]],
                    candidates[candidate_index], block, blocks_x, rows, columns,
                    format, shards[shard]);
                score = std::min(score, gain / std::max(residual_energy[shard], 1e-18));
            }
            if (score > best.predicted_gain) {
                best.candidate = candidate_index;
                best.predicted_gain = score;
            }
        }
        if (best.predicted_gain > 0.0) proposals.push_back(best);
    }
    std::sort(proposals.begin(), proposals.end(), [](const feedback_proposal & left,
                                                     const feedback_proposal & right) {
        return left.predicted_gain > right.predicted_gain;
    });
    for (const feedback_proposal & proposal : proposals) {
        const uint32_t block = proposal.block;
        const uint32_t current_index = result.selected_indices[block];
        double robust_gain = INFINITY;
        for (uint32_t shard = 0; shard < shard_count; ++shard) {
            const double gain = block_output_gain(
                residuals[shard], candidates[current_index], candidates[proposal.candidate],
                block, blocks_x, rows, columns, format, shards[shard]);
            robust_gain = std::min(robust_gain,
                                   gain / std::max(residual_energy[shard], 1e-18));
        }
        if (robust_gain <= 1e-12) continue;
        for (uint32_t shard = 0; shard < 2; ++shard) {
            subtract_block_delta(residuals[shard], candidates[current_index],
                                 candidates[proposal.candidate], block, blocks_x,
                                 rows, columns, format, shards[shard]);
        }
        result.selected_indices[block] = proposal.candidate;
        apply_block_candidate(candidates[proposal.candidate], block, blocks_x, rows, columns,
                              format, result.reconstructed, result.compressed);
        ++result.forward_changes;
    }
    return true;
}

// Build a bounded per-block pool that preserves both the local optimum and
// candidates with different activation-space error directions. This is an
// offline selector aid; every candidate remains a complete legal ASTC stream.
std::vector<std::vector<uint32_t>> make_diverse_block_shortlists(
        const std::vector<float> & reference,
        const std::vector<astc_candidate> & candidates,
        uint32_t rows, uint32_t columns,
        const ggml_vk_astc_format_contract & format,
        const activations & calibration,
        uint32_t maximum_candidates,
        bool angular_distance) {
    const uint32_t blocks_x = (columns + format.block_width - 1) / format.block_width;
    const uint32_t blocks_y = (rows + format.block_height - 1) / format.block_height;
    const uint32_t block_count = blocks_x * blocks_y;
    std::vector<std::vector<uint32_t>> shortlists(block_count);
    if (candidates.empty() || maximum_candidates == 0) return shortlists;

    for (uint32_t block = 0; block < block_count; ++block) {
        const uint32_t row0 = (block / blocks_x) * format.block_height;
        const uint32_t column0 = (block % blocks_x) * format.block_width;
        const uint32_t row_end = std::min(row0 + format.block_height, rows);
        const uint32_t column_end = std::min(column0 + format.block_width, columns);
        const uint32_t row_count = row_end - row0;
        const size_t vector_size = static_cast<size_t>(calibration.samples) * row_count;
        std::vector<std::vector<double>> errors(candidates.size(),
                                                std::vector<double>(vector_size, 0.0));
        std::vector<double> local_scores(candidates.size(), 0.0);
        std::vector<double> error_norms(candidates.size(), 0.0);
        for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
            for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                const float * input = calibration.values.data() +
                    static_cast<size_t>(sample) * columns;
                for (uint32_t row = row0; row < row_end; ++row) {
                    double delta = 0.0;
                    for (uint32_t column = column0; column < column_end; ++column) {
                        const size_t weight_index = static_cast<size_t>(row) * columns + column;
                        delta += (candidates[candidate_index].reconstructed[weight_index] -
                                  reference[weight_index]) * input[column];
                    }
                    const size_t vector_index = static_cast<size_t>(sample) * row_count + row - row0;
                    errors[candidate_index][vector_index] = delta;
                    local_scores[candidate_index] += delta * delta;
                }
            }
            error_norms[candidate_index] = std::sqrt(local_scores[candidate_index]);
        }

        const size_t local_best = static_cast<size_t>(std::min_element(
            local_scores.begin(), local_scores.end()) - local_scores.begin());
        // Candidate zero is the stable ordinary-stream baseline. Keeping it in
        // every shortlist makes the comparison and the initial residual exact.
        shortlists[block].push_back(0);
        if (maximum_candidates > 1 && local_best != 0) {
            shortlists[block].push_back(static_cast<uint32_t>(local_best));
        }
        const double loss_limit = std::max(local_scores[local_best] * 8.0, 1e-24);
        std::vector<bool> selected(candidates.size(), false);
        selected[0] = true;
        selected[local_best] = true;
        while (shortlists[block].size() < maximum_candidates) {
            size_t best_index = candidates.size();
            double best_distance = -1.0;
            for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
                if (selected[candidate_index] || local_scores[candidate_index] > loss_limit) continue;
                double distance = std::numeric_limits<double>::infinity();
                for (uint32_t chosen : shortlists[block]) {
                    double difference = 0.0;
                    if (angular_distance) {
                        const double denominator = error_norms[candidate_index] * error_norms[chosen];
                        if (denominator > 1e-18) {
                            double dot = 0.0;
                            for (size_t index = 0; index < vector_size; ++index) {
                                dot += errors[candidate_index][index] * errors[chosen][index];
                            }
                            difference = 1.0 - std::clamp(dot / denominator, -1.0, 1.0);
                        }
                    } else {
                        for (size_t index = 0; index < vector_size; ++index) {
                            const double value = errors[candidate_index][index] - errors[chosen][index];
                            difference += value * value;
                        }
                    }
                    distance = std::min(distance, difference);
                }
                if (distance > best_distance) {
                    best_distance = distance;
                    best_index = candidate_index;
                }
            }
            if (best_index == candidates.size()) break;
            selected[best_index] = true;
            shortlists[block].push_back(static_cast<uint32_t>(best_index));
        }
    }
    return shortlists;
}

// Choose whole, independently decodable ASTC blocks from a legal stream pool.
// Calibration activations are the only selection signal; evaluation is outside.
bool coordinate_select_astc_blocks(const std::vector<float> & reference,
                                   const std::vector<astc_candidate> & candidates,
                                   uint32_t rows, uint32_t columns,
                                   const ggml_vk_astc_format_contract & format,
                                   const activations & calibration,
                                   const affine_decoder & decoder,
                                   coordinate_result & result,
                                   const std::vector<std::vector<uint32_t>> * shortlists = nullptr,
                                   double replacement_penalty = 0.0) {
    if (candidates.empty()) return false;
    const uint32_t blocks_x = (columns + format.block_width - 1) / format.block_width;
    const uint32_t blocks_y = (rows + format.block_height - 1) / format.block_height;
    const uint32_t block_count = blocks_x * blocks_y;
    if (shortlists != nullptr && shortlists->size() != block_count) return false;
    if (shortlists != nullptr) {
        for (const auto & shortlist : *shortlists) {
            for (uint32_t candidate_index : shortlist) {
                if (candidate_index >= candidates.size()) return false;
            }
        }
    }
    std::vector<uint32_t> all_candidate_indices(candidates.size());
    for (uint32_t index = 0; index < candidates.size(); ++index) all_candidate_indices[index] = index;
    const std::vector<double> expected = matvec_outputs(reference, rows, columns, calibration);
    std::vector<double> actual = matvec_outputs(candidates.front().reconstructed, rows, columns, calibration);
    std::vector<double> residual(expected.size());
    double residual_error = 0.0;
    for (size_t index = 0; index < residual.size(); ++index) {
        residual[index] = expected[index] - actual[index];
        residual_error += residual[index] * residual[index];
    }
    const double initial_residual_error = residual_error;
    result.reconstructed = candidates.front().reconstructed;
    result.compressed = candidates.front().compressed;
    std::vector<uint32_t> selected_indices(block_count, 0);
    result.selected_indices = selected_indices;
    const uint32_t regularization_baseline = candidates.size() > 1 ? 1 : 0;
    const double penalty_unit = replacement_penalty * initial_residual_error /
                                std::max<uint32_t>(block_count, 1);

    auto sweep = [&](bool reverse) {
        uint32_t changes = 0;
        for (uint32_t ordinal = 0; ordinal < blocks_x * blocks_y; ++ordinal) {
            const uint32_t block = reverse ? blocks_x * blocks_y - ordinal - 1 : ordinal;
            const uint32_t row0 = (block / blocks_x) * format.block_height;
            const uint32_t column0 = (block % blocks_x) * format.block_width;
            const uint32_t row_count = std::min(row0 + format.block_height, rows) - row0;
            const std::vector<uint32_t> & candidate_indices =
                shortlists == nullptr ? all_candidate_indices : (*shortlists)[block];
            const double current_objective = residual_error +
                penalty_unit * (selected_indices[block] == regularization_baseline ? 0.0 : 1.0);
            for (uint32_t candidate_index : candidate_indices) {
                if (!candidate_allowed_for_block(candidates[candidate_index], block)) continue;
                const astc_candidate & candidate = candidates[candidate_index];
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
                const double candidate_objective = candidate_error +
                    penalty_unit * (candidate_index == regularization_baseline ? 0.0 : 1.0);
                if (candidate_objective + 1e-18 >= current_objective) continue;
                for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                    for (uint32_t row = row0; row < std::min(row0 + format.block_height, rows); ++row) {
                        const size_t index = static_cast<size_t>(sample) * rows + row;
                        residual[index] -= delta[static_cast<size_t>(sample) * row_count + row - row0];
                    }
                }
                residual_error = candidate_error;
                selected_indices[block] = candidate_index;
                apply_block_candidate(candidate, block, blocks_x, rows, columns, format,
                                      result.reconstructed, result.compressed);
                result.selected_indices[block] = candidate_index;
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
    result.selected_indices = std::move(selected_indices);
    return maximum_error <= 1e-6;
}

// Greedy block-LDLQ-style target regeneration. This is deliberately a small
// PoC: each committed block shifts the continuous target of later columns
// using H = X^T X, while the actual choice remains a legal ASTC candidate.
bool block_ldlq_select_astc_blocks(const std::vector<float> & reference,
                                   const std::vector<astc_candidate> & candidates,
                                   uint32_t rows, uint32_t columns,
                                   const ggml_vk_astc_format_contract & format,
                                   const activations & calibration,
                                   const affine_decoder & decoder,
                                   coordinate_result & result,
                                   const std::vector<std::vector<uint32_t>> * shortlists = nullptr) {
    if (candidates.empty() || calibration.samples == 0) return false;
    const uint32_t blocks_x = (columns + format.block_width - 1) / format.block_width;
    const uint32_t blocks_y = (rows + format.block_height - 1) / format.block_height;
    const uint32_t block_count = blocks_x * blocks_y;
    if (shortlists != nullptr && shortlists->size() != block_count) return false;
    std::vector<double> gram(static_cast<size_t>(columns) * columns, 0.0);
    for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
        const float * input = calibration.values.data() + static_cast<size_t>(sample) * columns;
        for (uint32_t left = 0; left < columns; ++left) {
            for (uint32_t right = 0; right < columns; ++right) {
                gram[static_cast<size_t>(left) * columns + right] +=
                    static_cast<double>(input[left]) * input[right];
            }
        }
    }
    double maximum_diagonal = 0.0;
    for (uint32_t column = 0; column < columns; ++column) {
        maximum_diagonal = std::max(maximum_diagonal,
                                    gram[static_cast<size_t>(column) * columns + column]);
    }
    const double damping = std::max(maximum_diagonal * g_block_ldlq_damping, 1e-12);
    std::vector<float> target = reference;
    result.reconstructed = candidates.front().reconstructed;
    result.compressed = candidates.front().compressed;
    result.selected_indices.assign(block_count, 0);
    result.forward_changes = 0;
    result.reverse_changes = 0;

    std::vector<uint32_t> block_order;
    block_order.reserve(block_count);
    for (uint32_t row_block = 0; row_block < blocks_y; ++row_block) {
        std::vector<uint32_t> row_order;
        for (uint32_t column_block = 0; column_block < blocks_x; ++column_block) {
            row_order.push_back(row_block * blocks_x + column_block);
        }
        if (g_block_ldlq_order == astc_vulkan_ldlq_order::reverse) {
            std::reverse(row_order.begin(), row_order.end());
        } else if (g_block_ldlq_order == astc_vulkan_ldlq_order::pivot) {
            std::sort(row_order.begin(), row_order.end(), [&](uint32_t left, uint32_t right) {
                const uint32_t left_column0 = (left % blocks_x) * format.block_width;
                const uint32_t right_column0 = (right % blocks_x) * format.block_width;
                double left_score = 0.0;
                double right_score = 0.0;
                for (uint32_t column = left_column0;
                     column < std::min(left_column0 + format.block_width, columns); ++column) {
                    for (uint32_t other = 0; other < columns; ++other) {
                        left_score += std::abs(gram[static_cast<size_t>(column) * columns + other]);
                    }
                }
                for (uint32_t column = right_column0;
                     column < std::min(right_column0 + format.block_width, columns); ++column) {
                    for (uint32_t other = 0; other < columns; ++other) {
                        right_score += std::abs(gram[static_cast<size_t>(column) * columns + other]);
                    }
                }
                return left_score > right_score;
            });
        }
        block_order.insert(block_order.end(), row_order.begin(), row_order.end());
    }

    for (uint32_t ordinal = 0; ordinal < block_count; ++ordinal) {
        const uint32_t block = block_order[ordinal];
        const uint32_t row0 = (block / blocks_x) * format.block_height;
        const uint32_t column0 = (block % blocks_x) * format.block_width;
        const uint32_t row_end = std::min(row0 + format.block_height, rows);
        const uint32_t column_end = std::min(column0 + format.block_width, columns);
        const std::vector<uint32_t> all_indices = [&]() {
            std::vector<uint32_t> indices(candidates.size());
            for (uint32_t index = 0; index < candidates.size(); ++index) indices[index] = index;
            return indices;
        }();
        const std::vector<uint32_t> & candidate_indices =
            shortlists == nullptr ? all_indices : (*shortlists)[block];
        double best_score = INFINITY;
        uint32_t best_candidate = candidates.size();
        for (uint32_t candidate_index : candidate_indices) {
            if (candidate_index >= candidates.size()) return false;
            if (!candidate_allowed_for_block(candidates[candidate_index], block)) continue;
            const auto & candidate = candidates[candidate_index];
            double score = 0.0;
            for (uint32_t row = row0; row < row_end; ++row) {
                for (uint32_t left = column0; left < column_end; ++left) {
                    const double left_error = candidate.reconstructed[static_cast<size_t>(row) * columns + left] -
                                              target[static_cast<size_t>(row) * columns + left];
                    for (uint32_t right = column0; right < column_end; ++right) {
                        const double right_error = candidate.reconstructed[static_cast<size_t>(row) * columns + right] -
                                                   target[static_cast<size_t>(row) * columns + right];
                        score += left_error * gram[static_cast<size_t>(left) * columns + right] * right_error;
                    }
                }
            }
            if (score < best_score) {
                best_score = score;
                best_candidate = candidate_index;
            }
        }
        if (best_candidate == candidates.size()) return false;
        const auto & chosen = candidates[best_candidate];
        for (uint32_t row = row0; row < row_end; ++row) {
            std::vector<double> committed_error(column_end - column0, 0.0);
            for (uint32_t column = column0; column < column_end; ++column) {
                committed_error[column - column0] =
                    chosen.reconstructed[static_cast<size_t>(row) * columns + column] -
                    reference[static_cast<size_t>(row) * columns + column];
            }
            const auto update_future_block = [&](uint32_t future_block) {
                const uint32_t future_column0 = (future_block % blocks_x) * format.block_width;
                const uint32_t future_count = std::min(format.block_width, columns - future_column0);
                std::vector<double> rhs(future_count, 0.0);
                for (uint32_t future = 0; future < future_count; ++future) {
                    for (uint32_t current = 0; current < committed_error.size(); ++current) {
                        rhs[future] += gram[static_cast<size_t>(future_column0 + future) * columns +
                                           column0 + current] * committed_error[current];
                    }
                }
                std::vector<double> update;
                if (!astc_vulkan_ldlq_solve_damped_block(gram, columns, future_column0,
                                                         future_count, damping, rhs, update)) return false;
                for (uint32_t future = 0; future < future_count; ++future) {
                    target[static_cast<size_t>(row) * columns + future_column0 + future] -= update[future];
                }
                return true;
            };
            for (uint32_t future_ordinal = ordinal + 1; future_ordinal < block_count; ++future_ordinal) {
                const uint32_t future_block = block_order[future_ordinal];
                if ((future_block / blocks_x) != (block / blocks_x)) continue;
                if (!update_future_block(future_block)) return false;
            }
        }
        if (best_candidate != 0) ++result.forward_changes;
        result.selected_indices[block] = best_candidate;
        apply_block_candidate(chosen, block, blocks_x, rows, columns, format,
                              result.reconstructed, result.compressed);
    }
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
                    astc_roundtrip_result & result,
                    unsigned int candidate_limit = 0,
                    astc_candidate_capture * candidate_capture = nullptr) {
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
    if (candidate_limit != 0) {
        config.tune_candidate_limit = candidate_limit;
    }
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (candidate_capture != nullptr) {
        candidate_capture->blocks_x = (columns + format.block_width - 1) / format.block_width;
        candidate_capture->blocks_y = (rows + format.block_height - 1) / format.block_height;
        candidate_capture->blocks_z = 1;
        candidate_capture->block_width = format.block_width;
        candidate_capture->block_height = format.block_height;
        candidate_capture->block_depth = 1;
        candidate_capture->blocks.assign(
            static_cast<size_t>(candidate_capture->blocks_x) * candidate_capture->blocks_y,
            {});
        config.candidate_callback = capture_astc_candidate;
        config.candidate_callback_user_data = candidate_capture;
    }
#else
    (void) candidate_capture;
#endif
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

// The gauge-only selector repeatedly encodes independent 6x6 images with one
// immutable ASTC configuration. Keep one context per outer worker and share
// the parent's read-only tables; a context is never used concurrently.
class astc_persistent_context_pool {
public:
    astc_persistent_context_pool() = default;
    astc_persistent_context_pool(const astc_persistent_context_pool &) = delete;
    astc_persistent_context_pool & operator=(const astc_persistent_context_pool &) = delete;

    ~astc_persistent_context_pool() {
        for (astcenc_context * worker : workers_) astcenc_context_free(worker);
    }

    bool initialize(const ggml_vk_astc_format_contract & format, uint32_t worker_count) {
        astcenc_config config{};
        if (astcenc_config_init(ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
                                g_astc_preset, 0, &config) != ASTCENC_SUCCESS) return false;
        workers_.resize(worker_count, nullptr);
        source_scratch_.resize(worker_count);
        for (astcenc_context * & worker : workers_) {
            if (astcenc_context_alloc(&config, 1, &worker, nullptr) != ASTCENC_SUCCESS) return false;
        }
        return true;
    }

    std::vector<float> & source_scratch(uint32_t worker_index, size_t texel_count) {
        std::vector<float> & source = source_scratch_[worker_index];
        source.resize(texel_count);
        return source;
    }

    bool roundtrip(uint32_t worker_index, const std::vector<float> & source,
                   uint32_t rows, uint32_t columns,
                   const ggml_vk_astc_format_contract & format,
                   astc_roundtrip_result & result) const {
        if (worker_index >= workers_.size()) return false;
        astcenc_context * context = workers_[worker_index];
        result = {};
        result.compressed_bytes = ggml_vk_astc_image_storage_bytes(format, columns, rows);
        result.compressed.resize(result.compressed_bytes);
        void * source_slice = const_cast<float *>(source.data());
        astcenc_image source_image{ columns, rows, 1, ASTCENC_TYPE_F32, &source_slice };
        const astcenc_swizzle swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A };
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
        astcenc_compress_reset(context);
        return status == ASTCENC_SUCCESS;
    }

private:
    std::vector<astcenc_context *> workers_;
    std::vector<std::vector<float>> source_scratch_;
};

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

bool inspect_astc_blocks(const std::vector<std::array<uint8_t, 16>> & payloads,
                         const ggml_vk_astc_format_contract & format,
                         std::vector<astcenc_block_info> & infos) {
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
                            g_astc_preset, 0, &config) != ASTCENC_SUCCESS) return false;
    astcenc_context * context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (astcenc_context_alloc(&config, 1, &context, nullptr) != ASTCENC_SUCCESS) return false;
#else
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) return false;
#endif
    infos.resize(payloads.size());
    astcenc_error status = ASTCENC_SUCCESS;
    for (size_t index = 0; index < payloads.size(); ++index) {
        status = astcenc_get_block_info(context, payloads[index].data(), &infos[index]);
        if (status != ASTCENC_SUCCESS) break;
    }
    astcenc_context_free(context);
    return status == ASTCENC_SUCCESS;
}

void print_astc_mode_histogram(const char * role, const std::vector<astcenc_block_info> & infos) {
    // The tuple contains only standard ASTC block-header decisions. It is
    // intentionally independent of source/gauge metadata so standard and
    // neural candidate banks can be compared without changing the bitstream.
    using mode_key = std::array<uint32_t, 7>;
    std::map<mode_key, uint32_t> histogram;
    for (const astcenc_block_info & info : infos) {
        const mode_key key{
            static_cast<uint32_t>(info.color_endpoint_modes[0]),
            info.partition_count,
            info.weight_x,
            info.weight_y,
            info.weight_level_count,
            info.color_level_count,
            info.is_dual_plane_block ? info.dual_plane_component + 1u : 0u,
        };
        ++histogram[key];
    }
    for (const auto & [key, count] : histogram) {
        std::printf("latent-astc-mode role=%s endpoint-mode=%u partitions=%u weight-grid=%ux%u "
                    "weight-levels=%u endpoint-levels=%u dual-plane-component=%u count=%u\n",
                    role, key[0], key[1], key[2], key[3], key[4], key[5], key[6], count);
    }
}

double decoded_block_activation_error(const std::vector<float> & reference,
                                      const std::vector<float> & decoded_block,
                                      uint32_t row0, uint32_t column0,
                                      uint32_t rows, uint32_t columns,
                                      const ggml_vk_astc_format_contract & format,
                                      const activations & inputs) {
    double error = 0.0;
    for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
        const float * input = inputs.values.data() + static_cast<size_t>(sample) * columns;
        for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
            if (row0 + local_row >= rows) continue;
            double expected = 0.0, actual = 0.0;
            for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                if (column0 + local_column >= columns) continue;
                const size_t local = static_cast<size_t>(local_row) * format.block_width + local_column;
                const size_t global = static_cast<size_t>(row0 + local_row) * columns + column0 + local_column;
                expected += reference[global] * input[column0 + local_column];
                actual += decoded_block[local] * input[column0 + local_column];
            }
            const double delta = expected - actual;
            error += delta * delta;
        }
    }
    return error;
}

bool decode_loop_alpha_search(const std::vector<float> & weights,
                              const latent_representation & block_latents,
                              uint32_t rows, uint32_t columns,
                              const ggml_vk_astc_format_contract & format,
                              const activations & calibration,
                              const activations & validation,
                              const activations & holdout,
                              bool scalar_anchored_gauge = false,
                              const std::string & commit_log_path = {},
                              const std::string & selected_payload_path = {},
                              const std::string & decoded_reference_path = {},
                              const std::string & validation_payload_path = {},
                              const std::string & validation_reference_path = {},
                              const std::string & validation_metadata_path = {},
                              const std::string & neutral_payload_path = {},
                              const std::string & neutral_reference_path = {},
                              const std::string & neutral_metadata_path = {},
                              const std::string & row_strip_log_path = {},
                              uint32_t candidate_threads = 1,
                              bool row_strip_select = false,
                              bool row_strip_chunked = false,
                              bool row_strip_diagnostics = true,
                              bool persistent_worker_contexts = false,
                              bool scalar_anchored_c_delta = false,
                              encoder_search_mode encoder_search = encoder_search_mode::standard,
                              uint32_t neural_candidate_limit = 16,
                              bool weight_grid_gauge = false,
                              uint32_t source_levels = 0,
                              bool pv_lite_grid = false,
                              bool pv_lite_coarse_grid = false,
                              bool pv_alternate = false) {
    // Partial blocks use deterministic clamp padding. Padding is never perturbed
    // and is excluded from the neural objective.
    struct alpha_option {
        uint32_t row0;
        uint32_t column0;
        std::vector<float> decoded;
        std::array<uint8_t, 16> payload;
        uint32_t factor_index = 0;
    };
    struct alpha_block_result {
        bool valid = false;
        uint32_t row0 = 0;
        uint32_t column0 = 0;
        std::vector<float> neutral_decoded;
        std::vector<float> best_decoded;
        std::array<uint8_t, 16> neutral_payload{};
        std::vector<alpha_option> alternatives;
        double neutral_loss = INFINITY;
        double best_loss = INFINITY;
        uint32_t best_factor = 0;
        uint32_t unique_payloads = 0;
    };
    struct neural_candidate {
        alpha_option option;
        double loss = INFINITY;
        bool stock = false;
    };
    std::vector<alpha_option> options;
    std::vector<float> neutral(weights.size());
    std::vector<float> selected(weights.size());
    const uint32_t blocks_x = (columns + format.block_width - 1) / format.block_width;
    std::vector<std::array<uint8_t, 16>> neutral_payloads(
        static_cast<size_t>((rows + format.block_height - 1) / format.block_height) * blocks_x);
    auto write_decoded_reference = [&](const std::vector<std::array<uint8_t, 16>> & payloads) {
        if (decoded_reference_path.empty()) return true;
        std::vector<uint8_t> compressed(payloads.size() * 16);
        for (size_t index = 0; index < payloads.size(); ++index) {
            std::copy(payloads[index].begin(), payloads[index].end(), compressed.begin() + index * 16);
        }
        std::vector<float> decoded;
        return astc_decode(compressed, rows, columns, format, decoded) &&
               write_binary(decoded_reference_path, decoded);
    };
    const bool pv_grid = pv_lite_grid || pv_lite_coarse_grid;
    const char * candidate_family = pv_alternate ? "scalar-anchored-pv-alternating-v1" :
        astc_vulkan_gauge_candidate_family(weight_grid_gauge, pv_lite_grid, pv_lite_coarse_grid);
    // PV-lite v1 is deliberately a fixed coefficient grid, not a claim of a
    // complete alternating optimizer. It supplies a small P-step candidate
    // family while the existing exact ASTC encode/decode and V-step selector
    // remain unchanged. The neutral scalar-anchored candidate is mandatory.
    const std::vector<astc_vulkan_gauge_factor> factors =
        astc_vulkan_make_gauge_factors(scalar_anchored_c_delta,
                                       scalar_anchored_gauge,
                                       weight_grid_gauge,
                                       pv_lite_grid,
                                       pv_lite_coarse_grid);
    const std::string source_hash = vector_hash(weights);
    const std::string calibration_hash = vector_hash(calibration.values);
    const std::string validation_hash = vector_hash(validation.values);
    const std::string holdout_hash = vector_hash(holdout.values);
    const uint32_t blocks_y = (rows + format.block_height - 1) / format.block_height;
    const uint32_t block_count = blocks_x * blocks_y;
    std::vector<alpha_block_result> block_results;
    if (!row_strip_chunked) block_results.resize(block_count);
    candidate_threads = std::max(1u, candidate_threads);
    candidate_threads = std::min(candidate_threads, block_count == 0 ? 1u : block_count);
    std::atomic<uint32_t> next_block{ 0 };
    std::atomic<bool> failed{ false };
    auto generate_block = [&](uint32_t block_index, uint32_t worker_index,
                              astc_persistent_context_pool * persistent_contexts) {
        alpha_block_result result;
        result.row0 = (block_index / blocks_x) * format.block_height;
        result.column0 = (block_index % blocks_x) * format.block_width;
        const uint32_t row0 = result.row0;
        const uint32_t column0 = result.column0;
        float gauge_headroom = 1.0f;
        if (scalar_anchored_gauge) {
            for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                    // Padding is a deterministic encoder aid only. It must
                    // neither add semantic loss terms nor reduce the gauge
                    // amplitude available to the real tensor positions.
                    if (row0 + local_row >= rows || column0 + local_column >= columns) continue;
                    const uint32_t source_row = std::min(row0 + local_row, rows - 1);
                    const uint32_t source_column = std::min(column0 + local_column, columns - 1);
                    const float q = block_latents.texels[
                        (static_cast<size_t>(source_row) * columns + source_column) * 4];
                    gauge_headroom = std::min(gauge_headroom, std::min(q, 1.0f - q));
                }
            }
        }
        std::vector<std::array<uint8_t, 16>> seen;
        std::vector<neural_candidate> neural_candidates;
        std::vector<std::array<uint8_t, 16>> neural_seen_payloads;
        std::vector<float> local_source;
        for (uint32_t factor_index = 0; factor_index < factors.size(); ++factor_index) {
            std::vector<float> & source = persistent_contexts != nullptr ?
                persistent_contexts->source_scratch(worker_index,
                    static_cast<size_t>(format.block_width) * format.block_height * 4) : local_source;
            source.resize(static_cast<size_t>(format.block_width) * format.block_height * 4);
            for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                    const uint32_t source_row = std::min(row0 + local_row, rows - 1);
                    const uint32_t source_column = std::min(column0 + local_column, columns - 1);
                    const bool valid = row0 + local_row < rows && column0 + local_column < columns;
                    const size_t global = static_cast<size_t>(source_row) * columns + source_column;
                    float * dst = source.data() +
                        (static_cast<size_t>(local_row) * format.block_width + local_column) * 4;
                    const float * src = block_latents.texels.data() + global * 4;
                    std::copy_n(src, 4, dst);
                    if (scalar_anchored_gauge && valid) {
                        const float correction = factors[factor_index].correction * gauge_headroom;
                        const float x = format.block_width > 1 ?
                            2.0f * local_column / static_cast<float>(format.block_width - 1) - 1.0f : 0.0f;
                        const float y = format.block_height > 1 ?
                            2.0f * local_row / static_cast<float>(format.block_height - 1) - 1.0f : 0.0f;
                        const float basis_value = astc_vulkan_gauge_basis_value(
                            factors[factor_index].basis, x, y);
                        const float delta = factors[factor_index].gauge * gauge_headroom * basis_value;
                        dst[0] = dst[1] = dst[2] = src[0] + correction + delta;
                        dst[3] = src[0] + correction - delta;
                    } else if (valid) {
                        dst[3] = std::clamp(0.5f + factors[factor_index].gauge * (src[3] - 0.5f), 0.0f, 1.0f);
                    }
                }
            }
            astc_roundtrip_result roundtrip;
            // First make the ordinary encode. It defines the immutable scalar
            // anchor and preserves the exact standard path for every source
            // member of the gauge family.
            const bool encoded = persistent_contexts != nullptr ?
                persistent_contexts->roundtrip(worker_index, source, format.block_height, format.block_width,
                                               format, roundtrip) :
                astc_roundtrip(source, format.block_height, format.block_width, format, nullptr, roundtrip);
            if (!encoded ||
                roundtrip.compressed.size() != 16) {
                return result;
            }
            // Neural v1 deliberately performs a second, wider search. Its
            // output is *only* an alternative candidate; it can never replace
            // the mandatory stock payload merely by changing astcenc tuning.
            astc_roundtrip_result expanded_roundtrip;
            astc_candidate_capture capture;
            if (encoder_search == encoder_search_mode::neural) {
                capture.max_per_block = std::max(1u, neural_candidate_limit);
                if (!astc_roundtrip(source, format.block_height, format.block_width, format, nullptr,
                                    expanded_roundtrip, neural_candidate_limit, &capture) ||
                    expanded_roundtrip.compressed.size() != 16) {
                    return result;
                }
            }
            bool duplicate = false;
            for (const auto & seen_payload : seen) {
                if (std::equal(seen_payload.begin(), seen_payload.end(), roundtrip.compressed.begin())) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            std::array<uint8_t, 16> seen_payload{};
            std::copy_n(roundtrip.compressed.begin(), seen_payload.size(), seen_payload.begin());
            seen.push_back(seen_payload);
            const std::vector<float> decoded = reconstruct(roundtrip.texels, block_latents.decoder);
            std::array<uint8_t, 16> payload{};
            std::copy_n(roundtrip.compressed.begin(), payload.size(), payload.begin());
            const double loss = decoded_block_activation_error(
                weights, decoded, row0, column0, rows, columns, format, calibration);
            if (factor_index == 0) {
                result.neutral_decoded = decoded;
                result.neutral_payload = payload;
                result.neutral_loss = loss;
            } else {
                result.alternatives.push_back({ row0, column0, decoded, payload, factor_index });
            }
            if (loss < result.best_loss) {
                result.best_loss = loss;
                result.best_decoded = decoded;
                result.best_factor = factor_index;
            }
            if (encoder_search != encoder_search_mode::neural) continue;

            // The ordinary astcenc result is mandatory. The callback exposes
            // additional legal candidates before the harness would otherwise
            // discard them based only on image error.
            auto add_neural_candidate = [&](const std::array<uint8_t, 16> & candidate_payload,
                                            const std::vector<float> & candidate_decoded,
                                            bool stock) {
                const auto duplicate_payload = std::find(neural_seen_payloads.begin(), neural_seen_payloads.end(),
                                                         candidate_payload);
                if (duplicate_payload != neural_seen_payloads.end()) {
                    const size_t duplicate_index = static_cast<size_t>(
                        std::distance(neural_seen_payloads.begin(), duplicate_payload));
                    neural_candidates[duplicate_index].stock |= stock;
                    return;
                }
                neural_seen_payloads.push_back(candidate_payload);
                neural_candidates.push_back({ { row0, column0, candidate_decoded, candidate_payload, factor_index },
                                              decoded_block_activation_error(weights, candidate_decoded,
                                                                             row0, column0, rows, columns,
                                                                             format, calibration), stock });
            };
            add_neural_candidate(payload, decoded, true);
            if (expanded_roundtrip.compressed.size() == 16) {
                std::array<uint8_t, 16> expanded_payload{};
                std::copy_n(expanded_roundtrip.compressed.begin(), expanded_payload.size(), expanded_payload.begin());
                add_neural_candidate(expanded_payload,
                                     reconstruct(expanded_roundtrip.texels, block_latents.decoder), false);
            }
            if (capture.blocks.empty()) continue;
            for (const captured_astc_candidate & captured : capture.blocks.front()) {
                if (captured.block == payload) continue;
                std::vector<uint8_t> compressed(captured.block.begin(), captured.block.end());
                std::vector<float> candidate_texels;
                if (!astc_decode(compressed, format.block_height, format.block_width,
                                 format, candidate_texels)) {
                    return alpha_block_result{};
                }
                add_neural_candidate(captured.block,
                                     reconstruct(candidate_texels, block_latents.decoder), false);
            }
        }
        if (pv_alternate && scalar_anchored_gauge) {
            // Full PV v1: the P-step moves only two bounded coefficients
            // (gauge and block correction), while every V-step calls the
            // exact ASTC encode/decode oracle. The resulting payload is just
            // another legal candidate for the unchanged global selector.
            const std::vector<float> initial{ 0.0f, 0.0f };
            const std::vector<float> steps{ 0.25f * gauge_headroom, 0.25f * gauge_headroom };
            auto make_pv_source = [&](const std::vector<float> & continuous, std::vector<float> & destination) {
                if (continuous.size() != 2) return false;
                destination.resize(static_cast<size_t>(format.block_width) * format.block_height * 4);
                for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                    for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                        const uint32_t source_row = std::min(row0 + local_row, rows - 1);
                        const uint32_t source_column = std::min(column0 + local_column, columns - 1);
                        const bool valid = row0 + local_row < rows && column0 + local_column < columns;
                        const size_t global = static_cast<size_t>(source_row) * columns + source_column;
                        float * dst = destination.data() +
                            (static_cast<size_t>(local_row) * format.block_width + local_column) * 4;
                        const float q = block_latents.texels[global * 4];
                        std::copy_n(block_latents.texels.data() + global * 4, 4, dst);
                        if (valid) {
                            const float correction = continuous[1] * gauge_headroom;
                            const float delta = continuous[0] * gauge_headroom;
                            dst[0] = dst[1] = dst[2] = q + correction + delta;
                            dst[3] = q + correction - delta;
                        }
                    }
                }
                return true;
            };
            astc_vulkan_pv_result pv_result;
            const bool pv_ok = astc_vulkan_pv_alternate(
                initial, steps, 2,
                [&](const std::vector<float> & continuous, std::vector<float> & deployed) {
                    std::vector<float> source;
                    if (!make_pv_source(continuous, source)) return false;
                    astc_roundtrip_result roundtrip;
                    if (!(persistent_contexts != nullptr ?
                            persistent_contexts->roundtrip(worker_index, source, format.block_height,
                                                           format.block_width, format, roundtrip) :
                            astc_roundtrip(source, format.block_height, format.block_width, format, nullptr,
                                           roundtrip)) || roundtrip.compressed.size() != 16) return false;
                    deployed = reconstruct(roundtrip.texels, block_latents.decoder);
                    return true;
                },
                [&](const std::vector<float> & deployed) {
                    return decoded_block_activation_error(weights, deployed, row0, column0,
                                                          rows, columns, format, calibration);
                }, pv_result);
            if (pv_ok && !pv_result.deployed.empty()) {
                std::vector<float> source;
                astc_roundtrip_result roundtrip;
                if (make_pv_source(pv_result.continuous, source) &&
                    (persistent_contexts != nullptr ?
                        persistent_contexts->roundtrip(worker_index, source, format.block_height,
                                                       format.block_width, format, roundtrip) :
                        astc_roundtrip(source, format.block_height, format.block_width, format, nullptr,
                                       roundtrip)) && roundtrip.compressed.size() == 16) {
                    std::array<uint8_t, 16> payload{};
                    std::copy_n(roundtrip.compressed.begin(), payload.size(), payload.begin());
                    if (std::find(seen.begin(), seen.end(), payload) == seen.end()) {
                        seen.push_back(payload);
                        result.alternatives.push_back({ row0, column0, pv_result.deployed, payload, 0 });
                        if (pv_result.objective < result.best_loss) {
                            result.best_loss = pv_result.objective;
                            result.best_decoded = pv_result.deployed;
                            result.best_factor = 1;
                        }
                    }
                }
            }
        }
        if (encoder_search == encoder_search_mode::neural) {
            // Retain the stock payload from every gauge source as a regression
            // anchor, then fill a capped bank with exact local neural winners
            // that cover different calibration-space directions. This is
            // deliberately upstream of the unchanged conflict-aware selector.
            std::vector<size_t> retained;
            const auto retain = [&](size_t index) {
                if (std::find(retained.begin(), retained.end(), index) == retained.end()) retained.push_back(index);
            };
            for (size_t index = 0; index < neural_candidates.size(); ++index) {
                if (neural_candidates[index].stock) retain(index);
            }
            std::vector<size_t> by_loss(neural_candidates.size());
            for (size_t index = 0; index < by_loss.size(); ++index) by_loss[index] = index;
            std::sort(by_loss.begin(), by_loss.end(), [&](size_t left, size_t right) {
                return neural_candidates[left].loss < neural_candidates[right].loss;
            });
            if (!by_loss.empty()) retain(by_loss.front());
            const size_t cap = std::max<size_t>(neural_candidate_limit, retained.size());
            auto calibration_delta = [&](size_t candidate_index) {
                const alpha_option & option = neural_candidates[candidate_index].option;
                std::vector<double> delta(static_cast<size_t>(calibration.samples) * format.block_height, 0.0);
                for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                    const float * input = calibration.values.data() + static_cast<size_t>(sample) * columns;
                    for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                        if (row0 + local_row >= rows) continue;
                        for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                            if (column0 + local_column >= columns) continue;
                            const size_t local = static_cast<size_t>(local_row) * format.block_width + local_column;
                            delta[static_cast<size_t>(sample) * format.block_height + local_row] +=
                                (option.decoded[local] - result.neutral_decoded[local]) *
                                input[column0 + local_column];
                        }
                    }
                }
                return delta;
            };
            std::vector<std::vector<double>> retained_deltas;
            retained_deltas.reserve(cap);
            for (size_t index : retained) retained_deltas.push_back(calibration_delta(index));
            while (retained.size() < cap && retained.size() < neural_candidates.size()) {
                size_t best = neural_candidates.size();
                double best_diversity = -1.0;
                for (size_t candidate = 0; candidate < neural_candidates.size(); ++candidate) {
                    if (std::find(retained.begin(), retained.end(), candidate) != retained.end()) continue;
                    const std::vector<double> delta = calibration_delta(candidate);
                    double maximum_abs_cosine = 0.0;
                    for (const std::vector<double> & other : retained_deltas) {
                        double dot = 0.0, left_norm = 0.0, right_norm = 0.0;
                        for (size_t value = 0; value < delta.size(); ++value) {
                            dot += delta[value] * other[value];
                            left_norm += delta[value] * delta[value];
                            right_norm += other[value] * other[value];
                        }
                        if (left_norm > 1e-18 && right_norm > 1e-18) {
                            maximum_abs_cosine = std::max(maximum_abs_cosine,
                                std::abs(dot / std::sqrt(left_norm * right_norm)));
                        }
                    }
                    const double diversity = 1.0 - maximum_abs_cosine;
                    if (diversity > best_diversity ||
                        (diversity == best_diversity && best != neural_candidates.size() &&
                         neural_candidates[candidate].loss < neural_candidates[best].loss)) {
                        best = candidate;
                        best_diversity = diversity;
                    }
                }
                if (best == neural_candidates.size()) break;
                retained.push_back(best);
                retained_deltas.push_back(calibration_delta(best));
            }
            result.alternatives.clear();
            result.best_loss = INFINITY;
            result.best_decoded = result.neutral_decoded;
            result.best_factor = 0;
            for (size_t index : retained) {
                const neural_candidate & candidate = neural_candidates[index];
                if (candidate.option.payload != result.neutral_payload) {
                    result.alternatives.push_back(candidate.option);
                }
                if (candidate.loss < result.best_loss) {
                    result.best_loss = candidate.loss;
                    result.best_decoded = candidate.option.decoded;
                    result.best_factor = candidate.option.payload == result.neutral_payload ? 0 : 1;
                }
            }
            result.unique_payloads = static_cast<uint32_t>(neural_candidates.size());
        }
        if (encoder_search != encoder_search_mode::neural) {
            result.unique_payloads = static_cast<uint32_t>(seen.size());
        }
        result.valid = !result.neutral_decoded.empty() && !result.best_decoded.empty();
        return result;
    };
    if (row_strip_chunked) {
        // Keep only one six-row candidate dictionary resident. Each strip has
        // independent calibration residual support; its compact commit sequence
        // is merged globally below using the same gain/tie-break rule.
        struct strip_step {
            uint32_t row0 = 0;
            uint32_t column0 = 0;
            std::vector<float> decoded;
            std::array<uint8_t, 16> payload{};
            uint32_t factor_index = 0;
            double gain = 0.0;
        };
        struct strip_metrics {
            uint32_t strip = 0;
            uint32_t candidates = 0;
            uint32_t accepted = 0;
            double gain = 0.0;
            double generate_seconds = 0.0;
            double select_seconds = 0.0;
        };
        std::vector<alpha_option> strip_options;
        std::vector<std::vector<strip_step>> strip_steps(blocks_y);
        std::vector<strip_metrics> strip_metrics_log;
        strip_metrics_log.reserve(blocks_y);
        std::vector<float> neutral(weights.size());
        std::vector<float> selected(weights.size());
        std::vector<std::array<uint8_t, 16>> neutral_payloads(block_count);
        uint32_t neutral_wins = 0, non_neutral_wins = 0, unique_blocks = 0;
        double neutral_local_loss = 0.0, selected_local_loss = 0.0;
        double delta_energy = 0.0, gram_energy = 0.0, positive_cosine_sum = 0.0;
        uint32_t positive_cosine_pairs = 0;
        size_t peak_candidate_count = 0, peak_candidate_workset_bytes = 0;
        astc_persistent_context_pool persistent_context_pool;
        astc_persistent_context_pool * persistent_contexts = nullptr;
        // Neural recall carries callback-local candidate state. Keep its first
        // implementation deliberately reference-oriented; the optimized
        // persistent context path remains an exact standard-mode regression
        // baseline until callback state is made worker-local in astcenc.
        if (persistent_worker_contexts && scalar_anchored_gauge &&
            encoder_search == encoder_search_mode::standard) {
            if (!persistent_context_pool.initialize(format, candidate_threads)) return false;
            persistent_contexts = &persistent_context_pool;
        }
        for (uint32_t strip = 0; strip < blocks_y; ++strip) {
            const auto generation_begin = std::chrono::steady_clock::now();
            std::vector<alpha_block_result> strip_results(blocks_x);
            std::atomic<uint32_t> next_column{ 0 };
            std::atomic<bool> strip_failed{ false };
            auto strip_worker = [&](uint32_t worker_index) {
                for (;;) {
                    const uint32_t column_block = next_column.fetch_add(1, std::memory_order_relaxed);
                    if (column_block >= blocks_x || strip_failed.load(std::memory_order_relaxed)) return;
                    alpha_block_result result = generate_block(
                        strip * blocks_x + column_block, worker_index, persistent_contexts);
                    if (!result.valid) {
                        strip_failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                    strip_results[column_block] = std::move(result);
                }
            };
            std::vector<std::thread> strip_workers;
            strip_workers.reserve(candidate_threads);
            for (uint32_t index = 0; index < candidate_threads; ++index) {
                strip_workers.emplace_back(strip_worker, index);
            }
            for (std::thread & thread : strip_workers) thread.join();
            if (strip_failed.load(std::memory_order_relaxed)) return false;
            const double generation_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - generation_begin).count();

            strip_options.clear();
            strip_options.reserve(blocks_x * (factors.size() - 1));
            for (alpha_block_result & result : strip_results) {
                neutral_local_loss += result.neutral_loss;
                selected_local_loss += result.best_loss;
                unique_blocks += result.unique_payloads;
                if (result.best_factor == 0) ++neutral_wins; else ++non_neutral_wins;
                neutral_payloads[strip * blocks_x + result.column0 / format.block_width] = result.neutral_payload;
                // Move, rather than copy, the decoded alternatives into the
                // strip-local dictionary. This is the resident candidate pool
                // that is released at the end of the loop iteration.
                for (alpha_option & alternative : result.alternatives) {
                    strip_options.push_back(std::move(alternative));
                }
                result.alternatives.clear();
                for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                    for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                        if (result.row0 + local_row >= rows || result.column0 + local_column >= columns) continue;
                        const size_t local = static_cast<size_t>(local_row) * format.block_width + local_column;
                        const size_t global = static_cast<size_t>(result.row0 + local_row) * columns +
                                              result.column0 + local_column;
                        neutral[global] = result.neutral_decoded[local];
                        selected[global] = result.best_decoded[local];
                    }
                }
            }
            std::vector<double> strip_residual(static_cast<size_t>(calibration.samples) * format.block_height, 0.0);
            for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                const float * input = calibration.values.data() + static_cast<size_t>(sample) * columns;
                for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                    const uint32_t row = strip * format.block_height + local_row;
                    if (row >= rows) continue;
                    double expected = 0.0, actual = 0.0;
                    for (uint32_t column = 0; column < columns; ++column) {
                        expected += weights[static_cast<size_t>(row) * columns + column] * input[column];
                        actual += neutral[static_cast<size_t>(row) * columns + column] * input[column];
                    }
                    strip_residual[static_cast<size_t>(sample) * format.block_height + local_row] = expected - actual;
                }
            }
            auto make_strip_delta = [&](const alpha_option & option) {
                std::vector<double> delta(static_cast<size_t>(calibration.samples) * format.block_height, 0.0);
                for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                    const float * input = calibration.values.data() + static_cast<size_t>(sample) * columns;
                    for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                        if (option.row0 + local_row >= rows) continue;
                        double value = 0.0;
                        for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                            if (option.column0 + local_column >= columns) continue;
                            const size_t local = static_cast<size_t>(local_row) * format.block_width + local_column;
                            const size_t global = static_cast<size_t>(option.row0 + local_row) * columns +
                                                  option.column0 + local_column;
                            value += (option.decoded[local] - neutral[global]) * input[option.column0 + local_column];
                        }
                        delta[static_cast<size_t>(sample) * format.block_height + local_row] = value;
                    }
                }
                return delta;
            };
            std::vector<std::vector<double>> strip_deltas;
            strip_deltas.reserve(strip_options.size());
            for (const alpha_option & option : strip_options) strip_deltas.push_back(make_strip_delta(option));
            size_t candidate_workset_bytes = strip_options.size() * sizeof(alpha_option) +
                                             strip_deltas.size() * sizeof(std::vector<double>);
            for (size_t index = 0; index < strip_options.size(); ++index) {
                candidate_workset_bytes += strip_options[index].decoded.capacity() * sizeof(float);
                candidate_workset_bytes += strip_deltas[index].capacity() * sizeof(double);
            }
            peak_candidate_count = std::max(peak_candidate_count, strip_options.size());
            peak_candidate_workset_bytes = std::max(peak_candidate_workset_bytes, candidate_workset_bytes);
            if (row_strip_diagnostics) {
                for (size_t left = 0; left < strip_deltas.size(); ++left) {
                    for (size_t right = 0; right < strip_deltas.size(); ++right) {
                        double dot = 0.0, left_norm = 0.0, right_norm = 0.0;
                        for (size_t index = 0; index < strip_deltas[left].size(); ++index) {
                            dot += strip_deltas[left][index] * strip_deltas[right][index];
                            left_norm += strip_deltas[left][index] * strip_deltas[left][index];
                            right_norm += strip_deltas[right][index] * strip_deltas[right][index];
                        }
                        if (left == right) delta_energy += left_norm;
                        gram_energy += dot * dot;
                        if (left < right && dot > 0.0 && left_norm > 1e-18 && right_norm > 1e-18) {
                            positive_cosine_sum += dot / std::sqrt(left_norm * right_norm);
                            ++positive_cosine_pairs;
                        }
                    }
                }
            }
            const auto selection_begin = std::chrono::steady_clock::now();
            std::vector<bool> strip_committed(blocks_x, false);
            double strip_gain = 0.0;
            for (;;) {
                double best_gain = 0.0;
                size_t best_option = strip_options.size();
                for (size_t option_index = 0; option_index < strip_options.size(); ++option_index) {
                    const alpha_option & option = strip_options[option_index];
                    const uint32_t block = option.column0 / format.block_width;
                    if (strip_committed[block]) continue;
                    double dot = 0.0, norm = 0.0;
                    for (size_t index = 0; index < strip_deltas[option_index].size(); ++index) {
                        dot += strip_residual[index] * strip_deltas[option_index][index];
                        norm += strip_deltas[option_index][index] * strip_deltas[option_index][index];
                    }
                    const double gain = 2.0 * dot - norm;
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_option = option_index;
                    }
                }
                if (best_option == strip_options.size()) break;
                const alpha_option & option = strip_options[best_option];
                strip_committed[option.column0 / format.block_width] = true;
                for (size_t index = 0; index < strip_deltas[best_option].size(); ++index) {
                    strip_residual[index] -= strip_deltas[best_option][index];
                }
                strip_step step;
                step.row0 = option.row0;
                step.column0 = option.column0;
                step.decoded = option.decoded;
                step.payload = option.payload;
                step.factor_index = option.factor_index;
                step.gain = best_gain;
                strip_steps[strip].push_back(step);
                strip_gain += best_gain;
            }
            strip_metrics_log.push_back({ strip, static_cast<uint32_t>(strip_options.size()),
                                          static_cast<uint32_t>(strip_steps[strip].size()), strip_gain,
                                          generation_seconds,
                                          std::chrono::duration<double>(
                                              std::chrono::steady_clock::now() - selection_begin).count() });
        }
        if (!row_strip_log_path.empty()) {
            std::ofstream row_strip_log(row_strip_log_path);
            if (!row_strip_log) return false;
            row_strip_log << "strip,candidates,accepted,local_residual_gain,generation_seconds,selection_seconds\n";
            for (const strip_metrics & metrics : strip_metrics_log) {
                row_strip_log << metrics.strip << ',' << metrics.candidates << ','
                              << metrics.accepted << ',' << metrics.gain << ','
                              << metrics.generate_seconds << ',' << metrics.select_seconds << '\n';
            }
        }
        const double neutral_calibration = activation_relative_mse(weights, neutral, rows, columns, calibration);
        const double neutral_validation = activation_relative_mse(weights, neutral, rows, columns, validation);
        const double neutral_holdout = activation_relative_mse(weights, neutral, rows, columns, holdout);
        const double calibration_loss = activation_relative_mse(weights, selected, rows, columns, calibration);
        const double holdout_loss = activation_relative_mse(weights, selected, rows, columns, holdout);
        const std::vector<double> expected = matvec_outputs(weights, rows, columns, calibration);
        const std::vector<double> neutral_output = matvec_outputs(neutral, rows, columns, calibration);
        double residual_energy = 0.0, expected_energy = 0.0;
        for (size_t index = 0; index < expected.size(); ++index) {
            const double error = expected[index] - neutral_output[index];
            residual_energy += error * error;
            expected_energy += expected[index] * expected[index];
        }
        const std::vector<double> validation_expected = matvec_outputs(weights, rows, columns, validation);
        const std::vector<double> validation_neutral = matvec_outputs(neutral, rows, columns, validation);
        std::vector<double> validation_residual(validation_expected.size());
        double validation_energy = 0.0, validation_expected_energy = 0.0;
        for (size_t index = 0; index < validation_residual.size(); ++index) {
            validation_residual[index] = validation_expected[index] - validation_neutral[index];
            validation_energy += validation_residual[index] * validation_residual[index];
            validation_expected_energy += validation_expected[index] * validation_expected[index];
        }
        std::ofstream commit_log;
        if (!commit_log_path.empty()) {
            commit_log.open(commit_log_path);
            if (!commit_log) return false;
            commit_log << "commit,calibration_relative_mse,validation_relative_mse,"
                       << "marginal_residual_gain,cumulative_residual_gain";
            if (pv_grid) commit_log << ",factor_index,basis,gauge,correction";
            commit_log << "\n";
        }
        std::vector<size_t> strip_cursors(blocks_y, 0);
        std::vector<const strip_step *> committed_steps;
        uint32_t commits = 0, best_validation_commit = 0;
        double best_validation = neutral_validation;
        for (;;) {
            double best_gain = 0.0;
            uint32_t best_strip = blocks_y;
            for (uint32_t strip = 0; strip < blocks_y; ++strip) {
                if (strip_cursors[strip] == strip_steps[strip].size()) continue;
                const strip_step & step = strip_steps[strip][strip_cursors[strip]];
                if (step.gain > best_gain ||
                    (step.gain == best_gain && best_strip != blocks_y && strip < best_strip)) {
                    best_gain = step.gain;
                    best_strip = strip;
                }
            }
            if (best_strip == blocks_y) break;
            const strip_step & step = strip_steps[best_strip][strip_cursors[best_strip]++];
            committed_steps.push_back(&step);
            residual_energy -= best_gain;
            for (uint32_t sample = 0; sample < validation.samples; ++sample) {
                const float * input = validation.values.data() + static_cast<size_t>(sample) * columns;
                for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                    if (step.row0 + local_row >= rows) continue;
                    double delta = 0.0;
                    for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                        if (step.column0 + local_column >= columns) continue;
                        const size_t local = static_cast<size_t>(local_row) * format.block_width + local_column;
                        const size_t global = static_cast<size_t>(step.row0 + local_row) * columns +
                                              step.column0 + local_column;
                        delta += (step.decoded[local] - neutral[global]) * input[step.column0 + local_column];
                    }
                    const size_t output = static_cast<size_t>(sample) * rows + step.row0 + local_row;
                    const double previous = validation_residual[output];
                    validation_residual[output] -= delta;
                    validation_energy += validation_residual[output] * validation_residual[output] - previous * previous;
                }
            }
            ++commits;
            const double validation_loss = validation_energy / validation_expected_energy;
            if (validation_loss < best_validation) {
                best_validation = validation_loss;
                best_validation_commit = commits;
            }
            if (commit_log) {
                commit_log << commits << ',' << residual_energy / expected_energy << ',' << validation_loss << ','
                           << best_gain << ',' << neutral_calibration - residual_energy / expected_energy;
                if (pv_grid) {
                    const astc_vulkan_gauge_factor & factor = factors[step.factor_index];
                    commit_log << ',' << step.factor_index << ','
                               << astc_vulkan_gauge_basis_name(factor.basis) << ','
                               << factor.gauge << ',' << factor.correction;
                }
                commit_log << '\n';
            }
        }
        std::vector<float> conflict_aware = neutral;
        std::vector<float> validation_stopped = neutral;
        for (size_t index = 0; index < committed_steps.size(); ++index) {
            const strip_step & step = *committed_steps[index];
            for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                    if (step.row0 + local_row >= rows || step.column0 + local_column >= columns) continue;
                    const size_t local = static_cast<size_t>(local_row) * format.block_width + local_column;
                    const size_t global = static_cast<size_t>(step.row0 + local_row) * columns +
                                          step.column0 + local_column;
                    conflict_aware[global] = step.decoded[local];
                    if (index < best_validation_commit) validation_stopped[global] = step.decoded[local];
                }
            }
        }
        const double conflict_calibration = activation_relative_mse(weights, conflict_aware, rows, columns, calibration);
        const double conflict_holdout = activation_relative_mse(weights, conflict_aware, rows, columns, holdout);
        const double stopped_holdout = activation_relative_mse(weights, validation_stopped, rows, columns, holdout);
        const char * encoder_profile = pv_alternate ? "pv-alternating-v1" :
                                      pv_lite_coarse_grid ? "pv-lite-coarse-grid-v1" :
                                      pv_lite_grid ? "pv-lite-grid-v1" :
                                      weight_grid_gauge ? "weight-grid-gauge-v1" : "standard";
        std::vector<std::array<uint8_t, 16>> final_payloads = neutral_payloads;
        for (const strip_step * step : committed_steps) {
            final_payloads[(step->row0 / format.block_height) * blocks_x +
                           step->column0 / format.block_width] = step->payload;
        }
        if (!selected_payload_path.empty() && !write_binary(selected_payload_path, final_payloads)) return false;
        if (!write_decoded_reference(final_payloads)) return false;
        if (!neutral_payload_path.empty() && !write_binary(neutral_payload_path, neutral_payloads)) return false;
        if (!neutral_metadata_path.empty() && !write_export_metadata(
                neutral_metadata_path, format, "gauge-la-neutral", rows, columns,
                block_latents.decoder, neutral_payloads.size() * 16,
                static_cast<size_t>(rows) * columns * 4, encoder_profile, 0, candidate_family,
                source_hash.c_str(), calibration_hash.c_str(), validation_hash.c_str(), holdout_hash.c_str())) return false;
        if (!neutral_reference_path.empty()) {
            std::vector<uint8_t> compressed(neutral_payloads.size() * 16);
            for (size_t index = 0; index < neutral_payloads.size(); ++index) {
                std::copy(neutral_payloads[index].begin(), neutral_payloads[index].end(),
                          compressed.begin() + index * 16);
            }
            std::vector<float> decoded;
            if (!astc_decode(compressed, rows, columns, format, decoded) ||
                !write_binary(neutral_reference_path, decoded)) return false;
        }
        std::vector<std::array<uint8_t, 16>> validation_payloads = neutral_payloads;
        for (size_t index = 0; index < committed_steps.size() && index < best_validation_commit; ++index) {
            const strip_step * step = committed_steps[index];
            validation_payloads[(step->row0 / format.block_height) * blocks_x +
                                step->column0 / format.block_width] = step->payload;
        }
        if (!validation_payload_path.empty() && !write_binary(validation_payload_path, validation_payloads)) return false;
        if (!validation_metadata_path.empty() && !write_export_metadata(
                validation_metadata_path, format, "gauge-la-validation", rows, columns,
                block_latents.decoder, validation_payloads.size() * 16,
                static_cast<size_t>(rows) * columns * 4, encoder_profile,
                best_validation_commit, candidate_family,
                source_hash.c_str(), calibration_hash.c_str(), validation_hash.c_str(), holdout_hash.c_str())) return false;
        if (!validation_reference_path.empty()) {
            std::vector<uint8_t> compressed(validation_payloads.size() * 16);
            for (size_t index = 0; index < validation_payloads.size(); ++index) {
                std::copy(validation_payloads[index].begin(), validation_payloads[index].end(),
                          compressed.begin() + index * 16);
            }
            std::vector<float> decoded;
            if (!astc_decode(compressed, rows, columns, format, decoded) ||
                !write_binary(validation_reference_path, decoded)) return false;
        }
        if (!validation_payload_path.empty() || !validation_reference_path.empty() ||
            !validation_metadata_path.empty() || !neutral_payload_path.empty() ||
            !neutral_reference_path.empty() || !neutral_metadata_path.empty()) {
            std::printf("latent-validation-export format=%s commit=%u payload=%s reference=%s metadata=%s neutral-payload=%s neutral-reference=%s neutral-metadata=%s\n",
                        format.name, best_validation_commit,
                        validation_payload_path.empty() ? "" : validation_payload_path.c_str(),
                        validation_reference_path.empty() ? "" : validation_reference_path.c_str(),
                        validation_metadata_path.empty() ? "" : validation_metadata_path.c_str(),
                        neutral_payload_path.empty() ? "" : neutral_payload_path.c_str(),
                        neutral_reference_path.empty() ? "" : neutral_reference_path.c_str(),
                        neutral_metadata_path.empty() ? "" : neutral_metadata_path.c_str());
        }
        std::vector<std::array<uint8_t, 16>> selected_payloads;
        std::vector<std::array<uint8_t, 16>> baseline_payloads;
        selected_payloads.reserve(committed_steps.size());
        baseline_payloads.reserve(committed_steps.size());
        for (const strip_step * step : committed_steps) {
            selected_payloads.push_back(step->payload);
            baseline_payloads.push_back(neutral_payloads[(step->row0 / format.block_height) * blocks_x +
                                                         step->column0 / format.block_width]);
        }
        std::vector<astcenc_block_info> selected_infos, baseline_infos;
        if (!inspect_astc_blocks(selected_payloads, format, selected_infos) ||
            !inspect_astc_blocks(baseline_payloads, format, baseline_infos)) return false;
        print_astc_mode_histogram("selected", selected_infos);
        print_astc_mode_histogram("scalar-anchor", baseline_infos);
        uint32_t changed_payloads = 0, changed_dual_plane = 0, changed_partition_count = 0;
        uint32_t changed_endpoint_mode = 0, changed_weight_grid = 0, changed_weight_levels = 0;
        for (size_t index = 0; index < selected_infos.size(); ++index) {
            const astcenc_block_info & selected_info = selected_infos[index];
            const astcenc_block_info & baseline_info = baseline_infos[index];
            if (selected_payloads[index] != baseline_payloads[index]) ++changed_payloads;
            if (selected_info.is_dual_plane_block != baseline_info.is_dual_plane_block ||
                selected_info.dual_plane_component != baseline_info.dual_plane_component) ++changed_dual_plane;
            if (selected_info.partition_count != baseline_info.partition_count) ++changed_partition_count;
            if (selected_info.color_endpoint_modes[0] != baseline_info.color_endpoint_modes[0]) ++changed_endpoint_mode;
            if (selected_info.weight_x != baseline_info.weight_x || selected_info.weight_y != baseline_info.weight_y ||
                selected_info.weight_z != baseline_info.weight_z) ++changed_weight_grid;
            if (selected_info.weight_level_count != baseline_info.weight_level_count ||
                selected_info.color_level_count != baseline_info.color_level_count) ++changed_weight_levels;
        }
        const double effective_rank = gram_energy > 1e-18 ? delta_energy * delta_energy / gram_energy : 0.0;
        std::printf("latent-decode-loop-alpha-chunked strips=%u peak-candidates=%zu "
                    "peak-candidate-workset-bytes=%zu compact-steps=%zu diagnostics=%s\n",
                    blocks_y, peak_candidate_count, peak_candidate_workset_bytes, committed_steps.size(),
                    row_strip_diagnostics ? "full" : "light");
        std::printf("latent-decode-loop-alpha-gauge-modes accepted=%zu payload-changed=%u dual-plane-changed=%u "
                    "partition-changed=%u endpoint-mode-changed=%u weight-grid-changed=%u weight-levels-changed=%u\n",
                    committed_steps.size(), changed_payloads, changed_dual_plane, changed_partition_count,
                    changed_endpoint_mode, changed_weight_grid, changed_weight_levels);
        if (pv_grid) {
            std::vector<uint32_t> factor_counts(factors.size(), 0);
            for (const strip_step * step : committed_steps) ++factor_counts[step->factor_index];
            for (uint32_t index = 0; index < factor_counts.size(); ++index) {
                if (factor_counts[index] == 0) continue;
                const astc_vulkan_gauge_factor & factor = factors[index];
                std::printf("latent-pv-lite-factor index=%u basis=%s gauge=%g correction=%g commits=%u\n",
                            index, astc_vulkan_gauge_basis_name(factor.basis), factor.gauge, factor.correction,
                            factor_counts[index]);
            }
        }
        std::printf("latent-decode-loop-alpha format=%s mode=%s encoder-search=%s source-levels=%u blocks=%u neutral-wins=%u alpha-wins=%u "
                    "unique-candidates=%u neutral-calibration=%.8g selected-calibration=%.8g "
                    "neutral-holdout=%.8g selected-holdout=%.8g local-gain=%.8g "
                    "candidate-effective-rank=%.4g mean-positive-cosine=%.4g conflict-commits=%u "
                    "conflict-calibration=%.8g conflict-holdout=%.8g validation-best-commit=%u "
                    "validation-best=%.8g validation-stopped-holdout=%.8g\n",
                    format.name, pv_alternate ? "scalar-anchored-pv-alternating" :
                    (scalar_anchored_c_delta ? "scalar-anchored-c-delta" :
                    (weight_grid_gauge ?
                        (pv_lite_coarse_grid ? "scalar-anchored-pv-lite-coarse-grid" :
                         (pv_lite_grid ? "scalar-anchored-pv-lite-grid" : "scalar-anchored-weight-grid-gauge")) :
                        "scalar-anchored-gauge")),
                    encoder_search == encoder_search_mode::neural ? "neural" : "standard",
                    source_levels,
                    block_count, neutral_wins, non_neutral_wins, unique_blocks,
                    neutral_calibration, calibration_loss, neutral_holdout, holdout_loss,
                    neutral_local_loss - selected_local_loss, effective_rank,
                    positive_cosine_pairs == 0 ? 0.0 : positive_cosine_sum / positive_cosine_pairs,
                    commits, conflict_calibration, conflict_holdout, best_validation_commit,
                    best_validation, stopped_holdout);
        return std::isfinite(calibration_loss) && std::isfinite(holdout_loss);
    }
    auto worker = [&]() {
        for (;;) {
            const uint32_t block_index = next_block.fetch_add(1, std::memory_order_relaxed);
            if (block_index >= block_count || failed.load(std::memory_order_relaxed)) return;
            alpha_block_result result = generate_block(block_index, 0, nullptr);
            if (!result.valid) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            block_results[block_index] = std::move(result);
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(candidate_threads);
    for (uint32_t index = 0; index < candidate_threads; ++index) workers.emplace_back(worker);
    for (std::thread & thread : workers) thread.join();
    if (failed.load(std::memory_order_relaxed)) return false;
    uint32_t neutral_wins = 0, non_neutral_wins = 0, unique_blocks = 0;
    double neutral_local_loss = 0.0, selected_local_loss = 0.0;
    options.reserve(block_count * (factors.size() - 1));
    for (const alpha_block_result & result : block_results) {
        neutral_local_loss += result.neutral_loss;
        selected_local_loss += result.best_loss;
        unique_blocks += result.unique_payloads;
        if (result.best_factor == 0) ++neutral_wins; else ++non_neutral_wins;
        neutral_payloads[(result.row0 / format.block_height) * blocks_x +
                        result.column0 / format.block_width] = result.neutral_payload;
        options.insert(options.end(), result.alternatives.begin(), result.alternatives.end());
        for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
            for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                if (result.row0 + local_row >= rows || result.column0 + local_column >= columns) continue;
                const size_t local = static_cast<size_t>(local_row) * format.block_width + local_column;
                const size_t global = static_cast<size_t>(result.row0 + local_row) * columns +
                                      result.column0 + local_column;
                neutral[global] = result.neutral_decoded[local];
                selected[global] = result.best_decoded[local];
            }
        }
    }
    const double neutral_calibration = activation_relative_mse(weights, neutral, rows, columns, calibration);
    const double neutral_validation = activation_relative_mse(weights, neutral, rows, columns, validation);
    const double neutral_holdout = activation_relative_mse(weights, neutral, rows, columns, holdout);
    const double calibration_loss = activation_relative_mse(weights, selected, rows, columns, calibration);
    const double holdout_loss = activation_relative_mse(weights, selected, rows, columns, holdout);
    std::vector<float> conflict_aware = neutral;
    const std::vector<double> expected = matvec_outputs(weights, rows, columns, calibration);
    const std::vector<double> neutral_output = matvec_outputs(neutral, rows, columns, calibration);
    std::vector<double> residual(expected.size());
    for (size_t index = 0; index < residual.size(); ++index) residual[index] = expected[index] - neutral_output[index];
    double residual_energy = 0.0, expected_energy = 0.0;
    for (size_t index = 0; index < residual.size(); ++index) {
        residual_energy += residual[index] * residual[index];
        expected_energy += expected[index] * expected[index];
    }
    const std::vector<double> validation_expected = matvec_outputs(weights, rows, columns, validation);
    const std::vector<double> validation_neutral = matvec_outputs(neutral, rows, columns, validation);
    std::vector<double> validation_residual(validation_expected.size());
    double validation_energy = 0.0, validation_expected_energy = 0.0;
    for (size_t index = 0; index < validation_residual.size(); ++index) {
        validation_residual[index] = validation_expected[index] - validation_neutral[index];
        validation_energy += validation_residual[index] * validation_residual[index];
        validation_expected_energy += validation_expected[index] * validation_expected[index];
    }
    std::ofstream commit_log;
    if (!commit_log_path.empty()) {
        commit_log.open(commit_log_path);
        if (!commit_log) return false;
        commit_log << "commit,calibration_relative_mse,validation_relative_mse,"
                   << "marginal_residual_gain,cumulative_residual_gain\n";
    }
    auto make_delta = [&](const alpha_option & option, const activations & inputs) {
        // A candidate changes only the output rows covered by its ASTC block.
        // Keep this sparse footprint so large crop sweeps do not allocate one
        // dense activation vector per legal candidate.
        std::vector<double> delta(static_cast<size_t>(inputs.samples) * format.block_height, 0.0);
        for (uint32_t sample = 0; sample < inputs.samples; ++sample) {
            const float * input = inputs.values.data() + static_cast<size_t>(sample) * columns;
            for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                if (option.row0 + local_row >= rows) continue;
                double value = 0.0;
                for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                    if (option.column0 + local_column >= columns) continue;
                    const size_t local = static_cast<size_t>(local_row) * format.block_width + local_column;
                    const size_t global = static_cast<size_t>(option.row0 + local_row) * columns +
                                          option.column0 + local_column;
                    value += (option.decoded[local] - neutral[global]) * input[option.column0 + local_column];
                }
                delta[static_cast<size_t>(sample) * format.block_height + local_row] = value;
            }
        }
        return delta;
    };
    std::vector<std::vector<double>> option_deltas;
    std::vector<std::vector<double>> validation_option_deltas;
    option_deltas.reserve(options.size());
    validation_option_deltas.reserve(options.size());
    double delta_energy = 0.0, gram_energy = 0.0, positive_cosine_sum = 0.0;
    uint32_t positive_cosine_pairs = 0;
    for (const alpha_option & option : options) {
        option_deltas.push_back(make_delta(option, calibration));
        validation_option_deltas.push_back(make_delta(option, validation));
    }
    for (size_t left = 0; left < option_deltas.size(); ++left) {
        for (size_t right = 0; right < option_deltas.size(); ++right) {
            if (options[left].row0 != options[right].row0) continue;
            double dot = 0.0, left_norm = 0.0, right_norm = 0.0;
            for (size_t index = 0; index < option_deltas[left].size(); ++index) {
                dot += option_deltas[left][index] * option_deltas[right][index];
                left_norm += option_deltas[left][index] * option_deltas[left][index];
                right_norm += option_deltas[right][index] * option_deltas[right][index];
            }
            if (left == right) delta_energy += left_norm;
            gram_energy += dot * dot;
            if (left < right && dot > 0.0 && left_norm > 1e-18 && right_norm > 1e-18) {
                positive_cosine_sum += dot / std::sqrt(left_norm * right_norm);
                ++positive_cosine_pairs;
            }
        }
    }
    const double effective_rank = gram_energy > 1e-18 ? delta_energy * delta_energy / gram_energy : 0.0;
    std::vector<bool> committed(static_cast<size_t>((rows + format.block_height - 1) / format.block_height) * blocks_x, false);
    std::vector<const alpha_option *> committed_options;
    uint32_t commits = 0;
    uint32_t best_validation_commit = 0;
    double best_validation = neutral_validation;
    auto commit_option = [&](size_t best_option, double best_gain) {
        const alpha_option & option = options[best_option];
        const uint32_t block_index = (option.row0 / format.block_height) * blocks_x +
                                     option.column0 / format.block_width;
        committed[block_index] = true;
        committed_options.push_back(&option);
        const std::vector<double> & best_delta = option_deltas[best_option];
        for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
            for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                if (option.row0 + local_row >= rows) continue;
                const size_t local = static_cast<size_t>(sample) * format.block_height + local_row;
                const size_t output = static_cast<size_t>(sample) * rows + option.row0 + local_row;
                residual[output] -= best_delta[local];
            }
        }
        residual_energy -= best_gain;
        const std::vector<double> & validation_delta = validation_option_deltas[best_option];
        for (uint32_t sample = 0; sample < validation.samples; ++sample) {
            for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                if (option.row0 + local_row >= rows) continue;
                const size_t local = static_cast<size_t>(sample) * format.block_height + local_row;
                const size_t output = static_cast<size_t>(sample) * rows + option.row0 + local_row;
                const double previous = validation_residual[output];
                validation_residual[output] -= validation_delta[local];
                validation_energy += validation_residual[output] * validation_residual[output] -
                                     previous * previous;
            }
        }
        ++commits;
        const double validation_loss = validation_energy / validation_expected_energy;
        if (validation_loss < best_validation) {
            best_validation = validation_loss;
            best_validation_commit = commits;
        }
        if (commit_log) {
            commit_log << commits << ',' << residual_energy / expected_energy << ',' << validation_loss << ','
                       << best_gain << ',' << neutral_calibration - residual_energy / expected_energy << '\n';
        }
    };
    if (!row_strip_select) for (;;) {
        double best_gain = 0.0;
        size_t best_option = options.size();
        for (size_t option_index = 0; option_index < options.size(); ++option_index) {
            const alpha_option & option = options[option_index];
            const uint32_t block_index = (option.row0 / format.block_height) * blocks_x +
                                         option.column0 / format.block_width;
            if (committed[block_index]) continue;
            const std::vector<double> & delta = option_deltas[option_index];
            double dot = 0.0, norm = 0.0;
            for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                    if (option.row0 + local_row >= rows) continue;
                    const size_t local = static_cast<size_t>(sample) * format.block_height + local_row;
                    const size_t output = static_cast<size_t>(sample) * rows + option.row0 + local_row;
                    dot += residual[output] * delta[local];
                    norm += delta[local] * delta[local];
                }
            }
            const double gain = 2.0 * dot - norm;
            if (gain > best_gain) {
                best_gain = gain;
                best_option = option_index;
            }
        }
        if (best_option == options.size()) break;
        commit_option(best_option, best_gain);
    }
    if (row_strip_select) {
        struct strip_step { size_t option_index; double gain; };
        std::vector<std::vector<strip_step>> strip_steps(blocks_y);
        for (uint32_t strip = 0; strip < blocks_y; ++strip) {
            std::vector<bool> strip_committed(blocks_x, false);
            std::vector<double> strip_residual(static_cast<size_t>(calibration.samples) * format.block_height, 0.0);
            for (uint32_t sample = 0; sample < calibration.samples; ++sample) {
                for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
                    const uint32_t row = strip * format.block_height + local_row;
                    if (row >= rows) continue;
                    strip_residual[static_cast<size_t>(sample) * format.block_height + local_row] =
                        residual[static_cast<size_t>(sample) * rows + row];
                }
            }
            for (;;) {
                double best_gain = 0.0;
                size_t best_option = options.size();
                for (size_t option_index = 0; option_index < options.size(); ++option_index) {
                    const alpha_option & option = options[option_index];
                    if (option.row0 / format.block_height != strip) continue;
                    const uint32_t local_block = option.column0 / format.block_width;
                    if (strip_committed[local_block]) continue;
                    const std::vector<double> & delta = option_deltas[option_index];
                    double dot = 0.0, norm = 0.0;
                    for (size_t index = 0; index < delta.size(); ++index) {
                        dot += strip_residual[index] * delta[index];
                        norm += delta[index] * delta[index];
                    }
                    const double gain = 2.0 * dot - norm;
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_option = option_index;
                    }
                }
                if (best_option == options.size()) break;
                const alpha_option & option = options[best_option];
                strip_committed[option.column0 / format.block_width] = true;
                const std::vector<double> & delta = option_deltas[best_option];
                for (size_t index = 0; index < delta.size(); ++index) strip_residual[index] -= delta[index];
                strip_steps[strip].push_back({ best_option, best_gain });
            }
        }
        std::vector<size_t> strip_cursors(blocks_y, 0);
        for (;;) {
            double best_gain = 0.0;
            size_t best_strip = blocks_y;
            size_t best_option = options.size();
            for (uint32_t strip = 0; strip < blocks_y; ++strip) {
                if (strip_cursors[strip] == strip_steps[strip].size()) continue;
                const strip_step & step = strip_steps[strip][strip_cursors[strip]];
                if (step.gain > best_gain ||
                    (step.gain == best_gain && step.option_index < best_option)) {
                    best_gain = step.gain;
                    best_option = step.option_index;
                    best_strip = strip;
                }
            }
            if (best_strip == blocks_y) break;
            ++strip_cursors[best_strip];
            commit_option(best_option, best_gain);
        }
    }
    std::vector<float> validation_stopped = neutral;
    for (size_t index = 0; index < committed_options.size(); ++index) {
        const alpha_option & option = *committed_options[index];
        for (uint32_t local_row = 0; local_row < format.block_height; ++local_row) {
            for (uint32_t local_column = 0; local_column < format.block_width; ++local_column) {
                if (option.row0 + local_row >= rows || option.column0 + local_column >= columns) continue;
                const size_t local = static_cast<size_t>(local_row) * format.block_width + local_column;
                conflict_aware[static_cast<size_t>(option.row0 + local_row) * columns +
                               option.column0 + local_column] = option.decoded[local];
                if (index < best_validation_commit) {
                    validation_stopped[static_cast<size_t>(option.row0 + local_row) * columns +
                                       option.column0 + local_column] = option.decoded[local];
                }
            }
        }
    }
    const double conflict_calibration = activation_relative_mse(weights, conflict_aware, rows, columns, calibration);
    const double conflict_holdout = activation_relative_mse(weights, conflict_aware, rows, columns, holdout);
    const double stopped_holdout = activation_relative_mse(weights, validation_stopped, rows, columns, holdout);
    std::vector<std::array<uint8_t, 16>> final_payloads = neutral_payloads;
    for (const alpha_option * option : committed_options) {
        const uint32_t block_index = (option->row0 / format.block_height) * blocks_x +
                                     option->column0 / format.block_width;
        final_payloads[block_index] = option->payload;
    }
    if (!selected_payload_path.empty() && !write_binary(selected_payload_path, final_payloads)) return false;
    if (!write_decoded_reference(final_payloads)) return false;
    if (scalar_anchored_gauge && !committed_options.empty()) {
        std::vector<std::array<uint8_t, 16>> selected_payloads;
        selected_payloads.reserve(committed_options.size());
        std::vector<std::array<uint8_t, 16>> baseline_payloads;
        baseline_payloads.reserve(committed_options.size());
        for (const alpha_option * option : committed_options) {
            selected_payloads.push_back(option->payload);
            const uint32_t block_index = (option->row0 / format.block_height) * blocks_x +
                                         option->column0 / format.block_width;
            baseline_payloads.push_back(neutral_payloads[block_index]);
        }
        std::vector<astcenc_block_info> selected_infos;
        std::vector<astcenc_block_info> baseline_infos;
        if (!inspect_astc_blocks(selected_payloads, format, selected_infos) ||
            !inspect_astc_blocks(baseline_payloads, format, baseline_infos)) return false;
        print_astc_mode_histogram("selected", selected_infos);
        print_astc_mode_histogram("scalar-anchor", baseline_infos);
        uint32_t changed_payloads = 0, changed_dual_plane = 0, changed_partition_count = 0;
        uint32_t changed_endpoint_mode = 0, changed_weight_grid = 0, changed_weight_levels = 0;
        for (size_t index = 0; index < selected_infos.size(); ++index) {
            const astcenc_block_info & selected_info = selected_infos[index];
            const astcenc_block_info & baseline_info = baseline_infos[index];
            if (selected_payloads[index] != baseline_payloads[index]) ++changed_payloads;
            if (selected_info.is_dual_plane_block != baseline_info.is_dual_plane_block ||
                selected_info.dual_plane_component != baseline_info.dual_plane_component) ++changed_dual_plane;
            if (selected_info.partition_count != baseline_info.partition_count) ++changed_partition_count;
            if (selected_info.color_endpoint_modes[0] != baseline_info.color_endpoint_modes[0]) ++changed_endpoint_mode;
            if (selected_info.weight_x != baseline_info.weight_x ||
                selected_info.weight_y != baseline_info.weight_y ||
                selected_info.weight_z != baseline_info.weight_z) ++changed_weight_grid;
            if (selected_info.weight_level_count != baseline_info.weight_level_count ||
                selected_info.color_level_count != baseline_info.color_level_count) ++changed_weight_levels;
        }
        std::printf("latent-decode-loop-alpha-gauge-modes accepted=%zu payload-changed=%u dual-plane-changed=%u "
                    "partition-changed=%u endpoint-mode-changed=%u weight-grid-changed=%u weight-levels-changed=%u\n",
                    committed_options.size(), changed_payloads, changed_dual_plane, changed_partition_count,
                    changed_endpoint_mode, changed_weight_grid, changed_weight_levels);
    }
    std::printf("latent-decode-loop-alpha format=%s mode=%s encoder-search=%s source-levels=%u blocks=%u neutral-wins=%u alpha-wins=%u "
                "unique-candidates=%u neutral-calibration=%.8g selected-calibration=%.8g "
                "neutral-holdout=%.8g selected-holdout=%.8g local-gain=%.8g "
                "candidate-effective-rank=%.4g mean-positive-cosine=%.4g conflict-commits=%u "
                "conflict-calibration=%.8g conflict-holdout=%.8g validation-best-commit=%u "
                "validation-best=%.8g validation-stopped-holdout=%.8g\n",
                format.name, pv_alternate ? "scalar-anchored-pv-alternating" :
                    (scalar_anchored_gauge ?
                    (weight_grid_gauge ?
                        (pv_lite_coarse_grid ? "scalar-anchored-pv-lite-coarse-grid" :
                         (pv_lite_grid ? "scalar-anchored-pv-lite-grid" : "scalar-anchored-weight-grid-gauge")) :
                        "scalar-anchored-gauge") :
                    "block-alpha"),
                encoder_search == encoder_search_mode::neural ? "neural" : "standard",
                source_levels,
                neutral_wins + non_neutral_wins, neutral_wins, non_neutral_wins,
                unique_blocks, neutral_calibration, calibration_loss, neutral_holdout, holdout_loss,
                neutral_local_loss - selected_local_loss, effective_rank,
                positive_cosine_pairs == 0 ? 0.0 : positive_cosine_sum / positive_cosine_pairs,
                commits, conflict_calibration, conflict_holdout, best_validation_commit,
                best_validation, stopped_holdout);
    return std::isfinite(calibration_loss) && std::isfinite(holdout_loss);
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
                                          float minimum, float range,
                                          uint32_t source_levels = 0) {
    latent_representation result;
    result.texels.resize(weights.size() * 4);
    constexpr float kFewLevelLatentMargin = 1.0f / 16.0f;
    const bool few_level = source_levels >= 2;
    const float latent_scale = few_level ? 1.0f - 2.0f * kFewLevelLatentMargin : 1.0f;
    result.decoder = { range / latent_scale, 0.0,
                       minimum - range * kFewLevelLatentMargin / latent_scale };
    for (size_t index = 0; index < weights.size(); ++index) {
        float normalized = (weights[index] - minimum) / range;
        if (few_level) {
            normalized = std::round(normalized * (source_levels - 1)) / (source_levels - 1);
            normalized = kFewLevelLatentMargin + normalized * latent_scale;
        }
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

latent_representation make_additive_latents_with_basis(const std::vector<float> & weights,
                                                       float minimum, float range,
                                                       uint32_t rows, uint32_t columns,
                                                       uint32_t block, residual_basis basis,
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
    if (basis != residual_basis::free) {
        // Remove high-frequency residual detail with an ASTC-friendly local
        // basis. The constant/row/column controls are intentionally simple;
        // plane and confidence gating remain separate experiments.
        for (uint32_t row0 = 0; row0 < rows; row0 += block) {
            for (uint32_t column0 = 0; column0 < columns; column0 += block) {
                const uint32_t row_end = std::min(row0 + block, rows);
                const uint32_t column_end = std::min(column0 + block, columns);
                if (basis == residual_basis::block_plane) {
                    std::vector<float> row_means(row_end - row0, 0.0f);
                    std::vector<float> column_means(column_end - column0, 0.0f);
                    double total = 0.0;
                    for (uint32_t row = row0; row < row_end; ++row) {
                        for (uint32_t column = column0; column < column_end; ++column) {
                            const float value = residuals[static_cast<size_t>(row) * columns + column];
                            row_means[row - row0] += value;
                            column_means[column - column0] += value;
                            total += value;
                        }
                    }
                    const float overall = static_cast<float>(total / std::max((row_end - row0) * (column_end - column0), 1u));
                    for (float & value : row_means) value /= std::max(column_end - column0, 1u);
                    for (float & value : column_means) value /= std::max(row_end - row0, 1u);
                    for (uint32_t row = row0; row < row_end; ++row) {
                        for (uint32_t column = column0; column < column_end; ++column) {
                            residuals[static_cast<size_t>(row) * columns + column] =
                                row_means[row - row0] + column_means[column - column0] - overall;
                        }
                    }
                } else if (basis == residual_basis::block_row) {
                    for (uint32_t row = row0; row < row_end; ++row) {
                        double sum = 0.0;
                        for (uint32_t column = column0; column < column_end; ++column) {
                            sum += residuals[static_cast<size_t>(row) * columns + column];
                        }
                        const float mean = static_cast<float>(sum / std::max(column_end - column0, 1u));
                        for (uint32_t column = column0; column < column_end; ++column) {
                            residuals[static_cast<size_t>(row) * columns + column] = mean;
                        }
                    }
                } else if (basis == residual_basis::block_column) {
                    for (uint32_t column = column0; column < column_end; ++column) {
                        double sum = 0.0;
                        for (uint32_t row = row0; row < row_end; ++row) {
                            sum += residuals[static_cast<size_t>(row) * columns + column];
                        }
                        const float mean = static_cast<float>(sum / std::max(row_end - row0, 1u));
                        for (uint32_t row = row0; row < row_end; ++row) {
                            residuals[static_cast<size_t>(row) * columns + column] = mean;
                        }
                    }
                } else {
                    double sum = 0.0;
                    uint32_t count = 0;
                    for (uint32_t row = row0; row < row_end; ++row) {
                        for (uint32_t column = column0; column < column_end; ++column) {
                            sum += residuals[static_cast<size_t>(row) * columns + column];
                            ++count;
                        }
                    }
                    const float mean = static_cast<float>(sum / std::max(count, 1u));
                    for (uint32_t row = row0; row < row_end; ++row) {
                        for (uint32_t column = column0; column < column_end; ++column) {
                            residuals[static_cast<size_t>(row) * columns + column] = mean;
                        }
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

latent_representation make_additive_latents(const std::vector<float> & weights,
                                            float minimum, float range,
                                            uint32_t rows, uint32_t columns,
                                            uint32_t block, bool block_residual,
                                            uint32_t coarse_levels) {
    return make_additive_latents_with_basis(
        weights, minimum, range, rows, columns, block,
        block_residual ? residual_basis::block_constant : residual_basis::free,
        coarse_levels);
}

latent_representation make_activation_constant_latents(
        const std::vector<float> & weights, float minimum, float range,
        uint32_t rows, uint32_t columns, uint32_t block,
        const activations & calibration, bool shard_gate, bool complexity_gate,
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
    const uint32_t shard_count = shard_gate ? std::min<uint32_t>(4, calibration.samples) : 1;
    for (uint32_t row0 = 0; row0 < rows; row0 += block) {
        for (uint32_t column0 = 0; column0 < columns; column0 += block) {
            const uint32_t row_end = std::min(row0 + block, rows);
            const uint32_t column_end = std::min(column0 + block, columns);
            double numerator = 0.0, denominator = 0.0, residual_energy = 0.0;
            std::vector<double> shard_corrections;
            for (uint32_t shard = 0; shard < shard_count; ++shard) {
                const uint32_t first = calibration.samples * shard / shard_count;
                const uint32_t last = calibration.samples * (shard + 1) / shard_count;
                double shard_numerator = 0.0, shard_denominator = 0.0;
                for (uint32_t sample = first; sample < last; ++sample) {
                    const float * x = calibration.values.data() + static_cast<size_t>(sample) * columns;
                    double u = 0.0;
                    for (uint32_t column = column0; column < column_end; ++column) u += x[column];
                    for (uint32_t row = row0; row < row_end; ++row) {
                        double output_residual = 0.0;
                        for (uint32_t column = column0; column < column_end; ++column) {
                            output_residual += residuals[static_cast<size_t>(row) * columns + column] * x[column];
                        }
                        shard_numerator += output_residual * u;
                        shard_denominator += u * u;
                        residual_energy += output_residual * output_residual;
                    }
                }
                numerator += shard_numerator;
                denominator += shard_denominator;
                shard_corrections.push_back(shard_denominator > 1e-18 ? shard_numerator / shard_denominator : 0.0);
            }
            double correction = denominator > 1e-18 ? numerator / denominator : 0.0;
            double gate = 1.0;
            if (shard_gate && shard_corrections.size() > 1) {
                double mean = 0.0;
                for (double value : shard_corrections) mean += value;
                mean /= shard_corrections.size();
                double variance = 0.0;
                for (double value : shard_corrections) variance += (value - mean) * (value - mean);
                variance /= shard_corrections.size();
                const double consistency = std::abs(mean) / (std::sqrt(variance) + 1e-9);
                gate *= std::clamp((consistency - 1.0) / 3.0, 0.0, 1.0);
            }
            if (complexity_gate && denominator > 1e-18 && residual_energy > 1e-18) {
                const double projected_energy = correction * correction * denominator;
                gate *= std::sqrt(std::clamp(projected_energy / residual_energy, 0.0, 1.0));
            }
            const float projected = static_cast<float>(std::clamp(
                correction * gate, -static_cast<double>(residual_radius),
                static_cast<double>(residual_radius)));
            for (uint32_t row = row0; row < row_end; ++row) {
                for (uint32_t column = column0; column < column_end; ++column) {
                    residuals[static_cast<size_t>(row) * columns + column] = projected;
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
        texel[0] = texel[1] = texel[2] = l;
        texel[3] = a;
    }
    return result;
}

bool parse_residual_basis(const std::string & name, residual_basis & basis) {
    if (name == "constant") basis = residual_basis::block_constant;
    else if (name == "row") basis = residual_basis::block_row;
    else if (name == "column") basis = residual_basis::block_column;
    else if (name == "plane") basis = residual_basis::block_plane;
    else return false;
    return true;
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
                         bool include_fast_candidate, bool include_diverse_candidates,
                         bool regularized_selection, bool selector_compare = false,
                         bool include_candidate_capacity = false) {
    astc_roundtrip_result standard;
    astc_roundtrip_result neural;
    astc_candidate_capture captured_candidates;
    if (!astc_roundtrip(latents.texels, rows, columns, format, nullptr, standard) ||
        !astc_roundtrip(latents.texels, rows, columns, format, &latents.decoder, neural, 0,
                        include_candidate_capacity ? &captured_candidates : nullptr)) return false;
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
    if (include_fast_candidate || include_diverse_candidates) {
        const float saved_preset = g_astc_preset;
        g_astc_preset = ASTCENC_PRE_FAST;
        astc_roundtrip_result fast;
        const bool encoded = astc_roundtrip(latents.texels, rows, columns, format, &latents.decoder, fast);
        g_astc_preset = saved_preset;
        if (!encoded) return false;
        candidates.push_back({ "neural-rank-fast", reconstruct(fast.texels, latents.decoder),
                               std::move(fast.compressed) });
    }
    if (include_diverse_candidates) {
        const float saved_preset = g_astc_preset;
        g_astc_preset = ASTCENC_PRE_MEDIUM;
        astc_roundtrip_result medium;
        const bool encoded = astc_roundtrip(latents.texels, rows, columns, format,
                                             &latents.decoder, medium);
        g_astc_preset = saved_preset;
        if (!encoded) return false;
        candidates.push_back({ "neural-rank-medium", reconstruct(medium.texels, latents.decoder),
                               std::move(medium.compressed) });
    }
    if (include_candidate_capacity) {
        size_t captured_block_count = 0;
        size_t captured_candidate_count = 0;
        for (const auto & block_candidates : captured_candidates.blocks) {
            if (!block_candidates.empty()) ++captured_block_count;
            captured_candidate_count += block_candidates.size();
        }
        for (size_t block = 0; block < captured_candidates.blocks.size(); ++block) {
            for (size_t candidate_index = 0;
                 candidate_index < captured_candidates.blocks[block].size(); ++candidate_index) {
                std::vector<uint8_t> compressed = neural.compressed;
                std::copy(captured_candidates.blocks[block][candidate_index].block.begin(),
                          captured_candidates.blocks[block][candidate_index].block.end(),
                          compressed.data() + block * 16);
                std::vector<float> texels;
                if (!astc_decode(compressed, rows, columns, format, texels)) return false;
                char name[64];
                std::snprintf(name, sizeof(name), "astc-topk-block-%zu-candidate-%zu",
                              block, candidate_index);
                candidates.push_back({ name, reconstruct(texels, latents.decoder),
                                       std::move(compressed), static_cast<int32_t>(block) });
            }
        }
        std::printf("latent-candidate-pool format=%s callbacks=%u blocks=%zu candidates=%zu "
                    "per-block-cap=%u pool=%zu\n", format.name,
                    captured_candidates.callback_count, captured_block_count,
                    captured_candidate_count, captured_candidates.max_per_block,
                    candidates.size());
    }
    const uint32_t selection_samples = regularized_selection ? calibration.samples / 2 : calibration.samples;
    if (regularized_selection && calibration.samples < 4) return false;
    const activations selection_inputs = slice_activation_samples(calibration, 0, selection_samples);
    const activations validation_inputs = regularized_selection ?
        slice_activation_samples(calibration, selection_samples, calibration.samples - selection_samples) :
        activations{};
    std::vector<std::vector<uint32_t>> shortlists;
    if (include_diverse_candidates) {
        shortlists = make_diverse_block_shortlists(weights, candidates, rows, columns,
                                                    format, selection_inputs, 4,
                                                    g_directional_shortlists);
    }
    if (selector_compare) {
        const uint32_t blocks_x = (columns + format.block_width - 1) / format.block_width;
        const uint32_t blocks_y = (rows + format.block_height - 1) / format.block_height;
        const uint32_t block_count = blocks_x * blocks_y;
        const hessian_stats stats = estimate_hessian_stats(calibration);
        std::printf("latent-hessian-stats samples=%u columns=%u rank=%u "
                    "max-eigen=%.8g min-positive-eigen=%.8g damped-condition=%.8g\n",
                    calibration.samples, calibration.columns, stats.rank,
                    stats.maximum_eigenvalue, stats.minimum_positive_eigenvalue,
                    stats.damped_condition);
        coordinate_result local;
        coordinate_result feedback;
        coordinate_result conflict_aware;
        coordinate_result stability;
        coordinate_result coordinate;
        coordinate_result ldlq;
        if (!local_select_astc_blocks(weights, candidates, rows, columns, format,
                                      calibration, local, &shortlists) ||
            !hessian_feedback_select_astc_blocks(weights, candidates, rows, columns, format,
                                                 calibration, feedback, &shortlists) ||
            !conflict_aware_select_astc_blocks(weights, candidates, rows, columns, format,
                                                calibration, conflict_aware, &shortlists) ||
            !stability_select_astc_blocks(weights, candidates, rows, columns, format,
                                          calibration, stability, &shortlists) ||
            !coordinate_select_astc_blocks(weights, candidates, rows, columns, format,
                                           calibration, latents.decoder, coordinate, &shortlists)) {
            return false;
        }
        if (!block_ldlq_select_astc_blocks(weights, candidates, rows, columns, format,
                                           calibration, latents.decoder, ldlq, &shortlists)) {
            return false;
        }
        const double local_loss = activation_relative_mse(weights, local.reconstructed,
                                                           rows, columns, holdout);
        const double feedback_loss = activation_relative_mse(weights, feedback.reconstructed,
                                                              rows, columns, holdout);
        const double conflict_loss = activation_relative_mse(weights, conflict_aware.reconstructed,
                                                              rows, columns, holdout);
        const double stability_loss = activation_relative_mse(weights, stability.reconstructed,
                                                               rows, columns, holdout);
        const double coordinate_loss = activation_relative_mse(weights, coordinate.reconstructed,
                                                               rows, columns, holdout);
        const double ldlq_loss = activation_relative_mse(weights, ldlq.reconstructed,
                                                         rows, columns, holdout);
        const double local_calibration = activation_relative_mse(weights, local.reconstructed,
                                                                  rows, columns, calibration);
        const double feedback_calibration = activation_relative_mse(weights, feedback.reconstructed,
                                                                     rows, columns, calibration);
        const double conflict_calibration = activation_relative_mse(weights, conflict_aware.reconstructed,
                                                                      rows, columns, calibration);
        const double stability_calibration = activation_relative_mse(weights, stability.reconstructed,
                                                                       rows, columns, calibration);
        const double coordinate_calibration = activation_relative_mse(weights, coordinate.reconstructed,
                                                                       rows, columns, calibration);
        const double ldlq_calibration = activation_relative_mse(weights, ldlq.reconstructed,
                                                                 rows, columns, calibration);
        print_calibration_shard_summary("local", weights, local.reconstructed,
                                        rows, columns, calibration);
        print_calibration_shard_summary("block-ldlq", weights, ldlq.reconstructed,
                                        rows, columns, calibration);
        const double denominator = local_loss - coordinate_loss;
        const double recovered_gain = denominator > 0.0 ?
            (local_loss - feedback_loss) / denominator : NAN;
        const double conflict_denominator = local_loss - coordinate_loss;
        const double conflict_gain = conflict_denominator > 0.0 ?
            (local_loss - conflict_loss) / conflict_denominator : NAN;
        const double stability_gain = conflict_denominator > 0.0 ?
            (local_loss - stability_loss) / conflict_denominator : NAN;
        const auto unique_choices = [](const coordinate_result & result) {
            std::vector<uint32_t> choices = result.selected_indices;
            std::sort(choices.begin(), choices.end());
            choices.erase(std::unique(choices.begin(), choices.end()), choices.end());
            return choices.size();
        };
        size_t minimum_coverage = std::numeric_limits<size_t>::max();
        size_t maximum_coverage = 0;
        size_t total_coverage = 0;
        for (uint32_t block = 0; block < blocks_x * blocks_y; ++block) {
            size_t coverage = 0;
            for (const auto & candidate : candidates) {
                if (candidate_allowed_for_block(candidate, block)) ++coverage;
            }
            minimum_coverage = std::min(minimum_coverage, coverage);
            maximum_coverage = std::max(maximum_coverage, coverage);
            total_coverage += coverage;
        }
        const double average_coverage = block_count == 0 ? 0.0 :
            static_cast<double>(total_coverage) / block_count;
        std::printf("latent-selector-compare format=%s pool=%zu "
                    "local-calibration=%.8g coordinate-calibration=%.8g "
                    "block-ldlq-calibration=%.8g "
                    "hessian-feedback-calibration=%.8g conflict-aware-calibration=%.8g "
                    "stability-calibration=%.8g "
                    "local-holdout=%.8g coordinate-holdout=%.8g block-ldlq-holdout=%.8g "
                    "block-ldlq-changes=%u block-ldlq-order=%s "
                    "hessian-feedback-holdout=%.8g conflict-aware-holdout=%.8g stability-holdout=%.8g "
                    "hessian-recovered-coordinate-gain=%.8g conflict-aware-recovered-coordinate-gain=%.8g "
                    "stability-recovered-coordinate-gain=%.8g feedback-round-changes=%u "
                    "conflict-aware-accepted=%u stability-accepted=%u stability-shards=%u "
                    "candidate-coverage-min=%zu candidate-coverage-max=%zu candidate-coverage-average=%.2f "
                    "candidate-shortlist-distance=%s "
                    "local-unique-choices=%zu conflict-unique-choices=%zu stability-unique-choices=%zu\n",
                    format.name, candidates.size(), local_calibration, coordinate_calibration,
                    ldlq_calibration, feedback_calibration, conflict_calibration, stability_calibration, local_loss,
                    coordinate_loss, ldlq_loss, ldlq.forward_changes,
                    astc_vulkan_ldlq_order_name(g_block_ldlq_order),
                    feedback_loss, conflict_loss, stability_loss, recovered_gain,
                    conflict_gain, stability_gain, feedback.forward_changes,
                    conflict_aware.forward_changes, stability.forward_changes,
                    g_stability_shards,
                    minimum_coverage == std::numeric_limits<size_t>::max() ? 0 : minimum_coverage,
                    maximum_coverage, average_coverage,
                    g_directional_shortlists ? "angular" : "euclidean",
                    unique_choices(local), unique_choices(conflict_aware), unique_choices(stability));
        return std::isfinite(local_loss) && std::isfinite(feedback_loss) &&
               std::isfinite(conflict_loss) && std::isfinite(stability_loss) &&
               std::isfinite(coordinate_loss) && std::isfinite(ldlq_loss);
    }
    auto select = [&](const activations & inputs, double penalty, coordinate_result & result) {
        return coordinate_select_astc_blocks(weights, candidates, rows, columns, format, inputs,
                                             latents.decoder, result,
                                             include_diverse_candidates ? &shortlists : nullptr,
                                             penalty);
    };
    double selected_penalty = 0.0;
    double validation_mse = NAN;
    if (regularized_selection) {
        const std::array<double, 7> penalties = { 0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03 };
        double best_validation = INFINITY;
        for (double penalty : penalties) {
            coordinate_result candidate_result;
            if (!select(selection_inputs, penalty, candidate_result)) return false;
            const double score = activation_relative_mse(weights, candidate_result.reconstructed,
                                                          rows, columns, validation_inputs);
            if (score < best_validation) {
                best_validation = score;
                selected_penalty = penalty;
            }
        }
        validation_mse = best_validation;
    }
    coordinate_result selected;
    if (!select(regularized_selection ? calibration : selection_inputs,
                selected_penalty, selected)) return false;
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
    if (regularized_selection) {
        std::printf("latent-coordinate-regularized format=%s penalty=%.8g validation-relative-MSE=%.8g "
                    "selection-samples=%u validation-samples=%u\n",
                    format.name, selected_penalty, validation_mse,
                    selection_inputs.samples, validation_inputs.samples);
    }
    if (include_diverse_candidates) {
        size_t minimum = std::numeric_limits<size_t>::max();
        size_t maximum = 0;
        size_t total = 0;
        for (const auto & shortlist : shortlists) {
            minimum = std::min(minimum, shortlist.size());
            maximum = std::max(maximum, shortlist.size());
            total += shortlist.size();
        }
        const double average = shortlists.empty() ? 0.0 :
            static_cast<double>(total) / shortlists.size();
        std::printf("latent-coordinate-diverse format=%s shortlist-min=%zu shortlist-max=%zu "
                    "shortlist-average=%.3f blocks=%zu\n",
                    format.name, minimum == std::numeric_limits<size_t>::max() ? 0 : minimum,
                    maximum, average, shortlists.size());
    }
    return std::isfinite(calibration_mse) && std::isfinite(holdout_mse);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string tensor_name;
    std::string trace_path;
    std::string calibration_trace_path;
    std::string validation_trace_path;
    std::string decode_loop_log_path;
    std::string decode_loop_payloads_path;
    std::string decode_loop_reference_path;
    std::string validation_payload_path;
    std::string validation_reference_path;
    std::string validation_metadata_path;
    std::string neutral_payload_path;
    std::string neutral_reference_path;
    std::string neutral_metadata_path;
    std::string row_strip_log_path;
    uint32_t candidate_threads = 1;
    bool row_strip_select = false;
    bool row_strip_chunked = false;
    bool row_strip_diagnostics = true;
    bool persistent_worker_contexts = false;
    bool search_levels = false;
    bool neural_rank = false;
    bool coordinate_select = false;
    bool coordinate_only = false;
    bool coordinate_fast_candidate = false;
    bool coordinate_diverse = false;
    bool coordinate_regularized = false;
    bool selector_compare = false;
    bool candidate_sweep = false;
    std::string footprint;
    std::string preset = "thorough";
    uint32_t maximum_samples = 0;
    uint32_t maximum_calibration_samples = 0;
    uint32_t maximum_rows = 0;
    uint32_t maximum_columns = 0;
    std::string export_astc_path;
    std::string export_reference_path;
    std::string export_weights_path;
    std::string export_metadata_path;
    std::string export_mode = "additive";
    bool export_only = false;
    std::string residual_basis_name;
    bool residual_basis_only = false;
    bool activation_alpha_sweep = false;
    bool decode_loop_alpha_sweep = false;
    bool scalar_anchored_gauge_sweep = false;
    bool scalar_anchored_c_delta_sweep = false;
    bool weight_grid_gauge_sweep = false;
    bool few_level_weight_grid_gauge_sweep = false;
    bool pv_lite_grid_sweep = false;
    bool pv_lite_coarse_grid_sweep = false;
    bool pv_alternate = false;
    bool d1_prescreen = false;
    std::string d1_prescreen_gpu_shader;
    encoder_search_mode encoder_search = encoder_search_mode::standard;
    uint32_t neural_candidate_limit = 16;
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
        } else if (option == "--coordinate-diverse") {
            coordinate_diverse = true;
        } else if (option == "--coordinate-regularized") {
            coordinate_regularized = true;
        } else if (option == "--selector-compare") {
            selector_compare = true;
        } else if (option == "--candidate-sweep") {
            candidate_sweep = true;
        } else if (option == "--candidate-angular") {
            g_directional_shortlists = true;
        } else if (option == "--stability-shards" && index + 1 < argc) {
            g_stability_shards = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (option == "--footprint" && index + 1 < argc) {
            footprint = argv[++index];
        } else if (option == "--preset" && index + 1 < argc) {
            preset = argv[++index];
        } else if (option == "--max-samples" && index + 1 < argc) {
            maximum_samples = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (option == "--max-calibration-samples" && index + 1 < argc) {
            maximum_calibration_samples = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (option == "--ldlq-damping" && index + 1 < argc) {
            g_block_ldlq_damping = std::stod(argv[++index]);
        } else if (option == "--ldlq-order" && index + 1 < argc) {
            const std::string order = argv[++index];
            if (order == "forward") g_block_ldlq_order = astc_vulkan_ldlq_order::forward;
            else if (order == "reverse") g_block_ldlq_order = astc_vulkan_ldlq_order::reverse;
            else if (order == "pivot") g_block_ldlq_order = astc_vulkan_ldlq_order::pivot;
            else {
                std::fprintf(stderr, "unsupported --ldlq-order value: %s\n", order.c_str());
                return 2;
            }
        } else if (option == "--max-rows" && index + 1 < argc) {
            maximum_rows = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (option == "--max-columns" && index + 1 < argc) {
            maximum_columns = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if ((option == "--export-astc" || option == "--export-reference" ||
                    option == "--export-weights" || option == "--export-metadata") && index + 1 < argc) {
            const std::string value = argv[++index];
            if (option == "--export-astc") export_astc_path = value;
            else if (option == "--export-reference") export_reference_path = value;
            else if (option == "--export-weights") export_weights_path = value;
            else export_metadata_path = value;
        } else if (option == "--export-mode" && index + 1 < argc) {
            export_mode = argv[++index];
        } else if (option == "--export-only") {
            export_only = true;
        } else if (option == "--residual-basis" && index + 1 < argc) {
            residual_basis_name = argv[++index];
        } else if (option == "--residual-basis-only") {
            residual_basis_only = true;
        } else if (option == "--activation-alpha-sweep") {
            activation_alpha_sweep = true;
        } else if (option == "--decode-loop-alpha-sweep") {
            decode_loop_alpha_sweep = true;
        } else if (option == "--scalar-anchored-gauge-sweep") {
            scalar_anchored_gauge_sweep = true;
        } else if (option == "--scalar-anchored-c-delta-sweep") {
            scalar_anchored_c_delta_sweep = true;
        } else if (option == "--weight-grid-gauge-sweep") {
            weight_grid_gauge_sweep = true;
        } else if (option == "--few-level-weight-grid-gauge-sweep") {
            few_level_weight_grid_gauge_sweep = true;
        } else if (option == "--pv-lite-grid-sweep") {
            pv_lite_grid_sweep = true;
            weight_grid_gauge_sweep = true;
        } else if (option == "--pv-lite-coarse-grid-sweep") {
            pv_lite_coarse_grid_sweep = true;
            weight_grid_gauge_sweep = true;
        } else if (option == "--pv-alternate") {
            pv_alternate = true;
        } else if (option == "--d1-prescreen") {
            d1_prescreen = true;
        } else if (option == "--d1-prescreen-gpu" && index + 1 < argc) {
            d1_prescreen = true;
            d1_prescreen_gpu_shader = argv[++index];
            scalar_anchored_gauge_sweep = true;
        } else if (option == "--decode-loop-log" && index + 1 < argc) {
            decode_loop_log_path = argv[++index];
        } else if (option == "--decode-loop-payloads" && index + 1 < argc) {
            decode_loop_payloads_path = argv[++index];
        } else if (option == "--decode-loop-reference" && index + 1 < argc) {
            decode_loop_reference_path = argv[++index];
        } else if (option == "--validation-payload" && index + 1 < argc) {
            validation_payload_path = argv[++index];
        } else if (option == "--validation-reference" && index + 1 < argc) {
            validation_reference_path = argv[++index];
        } else if (option == "--validation-metadata" && index + 1 < argc) {
            validation_metadata_path = argv[++index];
        } else if (option == "--neutral-payload" && index + 1 < argc) {
            neutral_payload_path = argv[++index];
        } else if (option == "--neutral-reference" && index + 1 < argc) {
            neutral_reference_path = argv[++index];
        } else if (option == "--neutral-metadata" && index + 1 < argc) {
            neutral_metadata_path = argv[++index];
        } else if (option == "--row-strip-log" && index + 1 < argc) {
            row_strip_log_path = argv[++index];
        } else if (option == "--candidate-threads" && index + 1 < argc) {
            candidate_threads = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (option == "--row-strip-select") {
            row_strip_select = true;
        } else if (option == "--row-strip-chunked") {
            row_strip_select = true;
            row_strip_chunked = true;
        } else if (option == "--row-strip-light-diagnostics") {
            row_strip_diagnostics = false;
        } else if (option == "--persistent-worker-contexts") {
            persistent_worker_contexts = true;
        } else if (option == "--encoder-search" && index + 1 < argc) {
            const std::string search = argv[++index];
            if (search == "standard") encoder_search = encoder_search_mode::standard;
            else if (search == "neural") encoder_search = encoder_search_mode::neural;
            else {
                std::fprintf(stderr, "unsupported --encoder-search value: %s\n", search.c_str());
                return 2;
            }
        } else if (option == "--neural-candidate-limit" && index + 1 < argc) {
            neural_candidate_limit = std::max(1u, static_cast<uint32_t>(std::stoul(argv[++index])));
        } else if ((option == "--model" || option == "--tensor" || option == "--trace" ||
                    option == "--calibration-trace" || option == "--validation-trace") &&
                   index + 1 < argc) {
            const std::string value = argv[++index];
            if (option == "--model") model_path = value;
            else if (option == "--tensor") tensor_name = value;
            else if (option == "--trace") trace_path = value;
            else if (option == "--calibration-trace") calibration_trace_path = value;
            else validation_trace_path = value;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--search-levels] [--neural-rank] [--coordinate-select] [--coordinate-only] [--coordinate-fast-candidate] [--coordinate-diverse] [--coordinate-regularized] [--selector-compare] [--candidate-sweep] [--candidate-angular] [--stability-shards N] "
                         "[--footprint 4x4|5x5|6x6|8x5|8x6|10x6|8x8|10x8] [--d1-prescreen|--d1-prescreen-gpu shader.spv] [--preset thorough|medium|fast] [--model path --tensor name] "
                         "[--trace path] [--calibration-trace path] [--validation-trace path] [--decode-loop-log path] [--decode-loop-payloads path] [--decode-loop-reference path] [--validation-payload path --validation-reference path --validation-metadata path] [--neutral-payload path --neutral-reference path --neutral-metadata path] [--row-strip-log path] [--candidate-threads N] [--row-strip-select] [--row-strip-chunked] [--row-strip-light-diagnostics] [--persistent-worker-contexts] [--encoder-search standard|neural] [--neural-candidate-limit N] [--max-samples N] [--max-calibration-samples N] [--ldlq-damping R] [--ldlq-order forward|reverse|pivot] [--max-rows N] [--max-columns N] "
                         "[--export-astc path --export-reference path --export-weights path --export-metadata path --export-mode scalar|additive] [--export-only] [--residual-basis constant|row|column|plane] [--activation-alpha-sweep] [--decode-loop-alpha-sweep] [--scalar-anchored-gauge-sweep] [--weight-grid-gauge-sweep] [--few-level-weight-grid-gauge-sweep] [--pv-lite-grid-sweep] [--pv-lite-coarse-grid-sweep] [--pv-alternate] [--scalar-anchored-c-delta-sweep]\n",
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
    if (coordinate_diverse && !coordinate_select) {
        std::fprintf(stderr, "--coordinate-diverse requires --coordinate-select\n");
        return 2;
    }
    if (coordinate_regularized && !coordinate_select) {
        std::fprintf(stderr, "--coordinate-regularized requires --coordinate-select\n");
        return 2;
    }
    if (selector_compare) {
        neural_rank = true;
        coordinate_select = true;
        coordinate_only = true;
        coordinate_diverse = true;
    }
    if (candidate_sweep) {
        neural_rank = true;
        coordinate_select = true;
        coordinate_only = true;
        selector_compare = true;
        coordinate_diverse = true;
    }
    if (export_astc_path.empty() != export_reference_path.empty() ||
        (!export_astc_path.empty() && footprint.empty())) {
        std::fprintf(stderr, "exports require --export-astc, --export-reference, and --footprint\n");
        return 2;
    }
    if (export_only && export_astc_path.empty()) {
        std::fprintf(stderr, "--export-only requires --export-astc and --export-reference\n");
        return 2;
    }
    if (!validation_metadata_path.empty() && validation_payload_path.empty()) {
        std::fprintf(stderr, "--validation-metadata requires --validation-payload\n");
        return 2;
    }
    if ((!neutral_reference_path.empty() || !neutral_metadata_path.empty()) && neutral_payload_path.empty()) {
        std::fprintf(stderr, "neutral reference and metadata require --neutral-payload\n");
        return 2;
    }
    if (!export_metadata_path.empty() && export_astc_path.empty()) {
        std::fprintf(stderr, "--export-metadata requires --export-astc and --export-reference\n");
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
    if (export_mode != "scalar" && export_mode != "additive") {
        std::fprintf(stderr, "unsupported --export-mode value: %s\n", export_mode.c_str());
        return 2;
    }
    residual_basis selected_basis = residual_basis::free;
    if (!residual_basis_name.empty() && !parse_residual_basis(residual_basis_name, selected_basis)) {
        std::fprintf(stderr, "unsupported --residual-basis value: %s\n", residual_basis_name.c_str());
        return 2;
    }
    if (residual_basis_only && residual_basis_name.empty()) {
        std::fprintf(stderr, "--residual-basis-only requires --residual-basis\n");
        return 2;
    }
    if ((activation_alpha_sweep || decode_loop_alpha_sweep || scalar_anchored_gauge_sweep ||
         scalar_anchored_c_delta_sweep) && export_only) {
        std::fprintf(stderr, "Alpha sweeps cannot be combined with --export-only\n");
        return 2;
    }
    if (!std::isfinite(g_block_ldlq_damping) || g_block_ldlq_damping < 0.0) {
        std::fprintf(stderr, "--ldlq-damping must be a finite non-negative value\n");
        return 2;
    }
    if (g_stability_shards < 2) {
        std::fprintf(stderr, "--stability-shards must be at least 2\n");
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

    if (!export_weights_path.empty()) {
        if (!write_binary(export_weights_path, weights)) {
            std::fprintf(stderr, "FP32 weight export failed\n");
            return 1;
        }
        std::printf("latent-weight-export rows=%u columns=%u weights=%s\n",
                    rows, columns, export_weights_path.c_str());
    }

    const auto [minimum_it, maximum_it] = std::minmax_element(weights.begin(), weights.end());
    const float minimum = *minimum_it;
    const float range = std::max(*maximum_it - minimum, 1e-6f);
    latent_representation scalar_latents;
    latent_representation row_column_latents;
    if (!export_only || export_mode == "scalar") {
        scalar_latents = make_scalar_latents(weights, minimum, range);
    }
    if (!export_only) {
        row_column_latents = make_row_column_latents(weights, rows, columns);
    }
    activations inputs = make_default_activations(columns);
    activations calibration_inputs = inputs;
    activations validation_inputs = inputs;
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
    if (validation_trace_path.empty()) validation_inputs = inputs;
    if (!validation_trace_path.empty()) {
        ggml_vk_astc_activation_trace loaded;
        if (!ggml_vk_astc_load_activation_trace(validation_trace_path, loaded, error) ||
            loaded.columns != trace_columns) {
            std::fprintf(stderr, "invalid validation activation trace: %s\n", error.c_str());
            return 1;
        }
        validation_inputs.samples = loaded.samples;
        validation_inputs.values = std::move(loaded.values);
    }
    limit_activation_samples(inputs, maximum_samples);
    limit_activation_samples(calibration_inputs,
                             maximum_calibration_samples == 0 ? maximum_samples :
                                                                  maximum_calibration_samples);
    crop_activation_columns(inputs, columns);
    crop_activation_columns(calibration_inputs, columns);
    crop_activation_columns(validation_inputs, columns);
    std::vector<astc_vulkan_footprint> prescreen_footprints;
    if (d1_prescreen && footprint.empty()) {
        const char * prescreen_backend = "cpu";
        std::vector<float> column_energy(columns, 0.0f);
        for (uint32_t sample = 0; sample < calibration_inputs.samples; ++sample) {
            const float * input = calibration_inputs.values.data() + static_cast<size_t>(sample) * columns;
            for (uint32_t column = 0; column < columns; ++column) column_energy[column] += input[column] * input[column];
        }
        const std::vector<astc_vulkan_d1_prescreen_candidate> screen_candidates{
            {astc_vulkan_footprint::k4x4, 16}, {astc_vulkan_footprint::k5x5, 16},
            {astc_vulkan_footprint::k6x6, 16}, {astc_vulkan_footprint::k8x6, 16},
            {astc_vulkan_footprint::k10x6, 8}, {astc_vulkan_footprint::k8x8, 8},
            {astc_vulkan_footprint::k10x8, 8}};
        std::vector<astc_vulkan_d1_prescreen_score> scores;
        std::vector<astc_vulkan_d1_prescreen_score> shortlist;
        bool screen_ok = false;
#if defined(ASTC_VULKAN_D1_PRESCREEN_GPU)
        if (!d1_prescreen_gpu_shader.empty()) {
            std::string gpu_error;
            screen_ok = astc_vulkan_score_d1_prescreen_gpu_default(
                d1_prescreen_gpu_shader, weights, rows, columns, column_energy,
                screen_candidates, scores, gpu_error);
            if (screen_ok) prescreen_backend = "gpu";
            if (!screen_ok) {
                std::fprintf(stderr, "D1 GPU pre-screen unavailable (%s); falling back to CPU\n", gpu_error.c_str());
            }
        }
#endif
        if (!screen_ok) screen_ok = astc_vulkan_score_d1_prescreen_cpu(
            weights, rows, columns, column_energy, screen_candidates, scores);
        if (!screen_ok ||
            !astc_vulkan_select_d1_prescreen_shortlist(scores, 4, 0.0, shortlist)) {
            std::fprintf(stderr, "D1 pre-screen failed\n");
            return 1;
        }
        std::printf("d1-prescreen backend=%s shortlist:", prescreen_backend);
        for (const auto & score : shortlist) {
            prescreen_footprints.push_back(score.candidate.footprint);
            const auto dimensions = d1_prescreen_dimensions(score.candidate.footprint);
            std::printf(" %ux%u(levels=%u,relative=%.6g,bpw=%.4g)", dimensions.first,
                        dimensions.second, score.candidate.levels, score.normalized_error,
                        score.bits_per_weight);
        }
        std::printf("\n");
    }
    for (const auto & format : { ggml_vk_astc_4x4_unorm_rgba,
                                 ggml_vk_astc_5x5_unorm_rgba,
                                 ggml_vk_astc_6x6_unorm_rgba,
                                 ggml_vk_astc_8x5_unorm_rgba,
                                 ggml_vk_astc_8x6_unorm_rgba,
                                 ggml_vk_astc_10x6_unorm_rgba,
                                 ggml_vk_astc_8x8_unorm_rgba,
                                 ggml_vk_astc_10x8_unorm_rgba }) {
        const std::string format_footprint = std::to_string(format.block_width) + "x" +
                                             std::to_string(format.block_height);
        if (!footprint.empty() && footprint != format_footprint) continue;
        if (footprint.empty() && d1_prescreen) {
            const auto selected = std::find_if(prescreen_footprints.begin(), prescreen_footprints.end(),
                [&](astc_vulkan_footprint candidate) {
                    const auto dimensions = d1_prescreen_dimensions(candidate);
                    return format.block_width == dimensions.first && format.block_height == dimensions.second;
                });
            if (selected == prescreen_footprints.end()) continue;
        }
        latent_representation additive_latents;
        if (!export_only || export_mode == "additive") {
            additive_latents = make_additive_latents(
                weights, minimum, range, rows, columns, format.block_width, false,
                kDefaultCoarseLevels);
        }
        latent_representation block_latents;
        if (!export_only) {
            block_latents = make_additive_latents(
                weights, minimum, range, rows, columns, format.block_width, true,
                kDefaultCoarseLevels);
        }
        if (!export_astc_path.empty()) {
            astc_roundtrip_result exported;
            const latent_representation & export_latents = export_mode == "scalar" ? scalar_latents : additive_latents;
            if (!astc_roundtrip(export_latents.texels, rows, columns, format, nullptr, exported) ||
                !write_binary(export_astc_path, exported.compressed) ||
                !write_binary(export_reference_path, exported.texels) ||
                (!export_metadata_path.empty() && !write_export_metadata(
                    export_metadata_path, format, export_mode.c_str(), rows, columns,
                    export_latents.decoder, exported.compressed.size(), exported.texels.size()))) {
                std::fprintf(stderr, "ASTC latent export failed\n");
                return 1;
            }
            std::printf("latent-export format=%s mode=%s bytes=%zu texels=%zu astc=%s reference=%s metadata=%s scale-l=%.8g scale-a=%.8g offset=%.8g\n",
                        format.name, export_mode.c_str(), exported.compressed.size(), exported.texels.size(),
                        export_astc_path.c_str(), export_reference_path.c_str(),
                        export_metadata_path.empty() ? "" : export_metadata_path.c_str(),
                        export_latents.decoder.scale_l, export_latents.decoder.scale_a,
                        export_latents.decoder.offset);
        }
        if (export_only) continue;
        if (!residual_basis_name.empty()) {
            const latent_representation structured = make_additive_latents_with_basis(
                weights, minimum, range, rows, columns, format.block_width,
                selected_basis, kDefaultCoarseLevels);
            const std::string case_name = "luminance-alpha-" + residual_basis_name;
            if (!std::isfinite(run_case(case_name.c_str(), weights, structured,
                                        rows, columns, format, inputs))) {
                std::fprintf(stderr, "ASTC structured residual smoke failed\n");
                return 1;
            }
        }
        if (activation_alpha_sweep) {
            const latent_representation activation_optimal = make_activation_constant_latents(
                weights, minimum, range, rows, columns, format.block_width,
                calibration_inputs, false, false, kDefaultCoarseLevels);
            const latent_representation shard_gated = make_activation_constant_latents(
                weights, minimum, range, rows, columns, format.block_width,
                calibration_inputs, true, false, kDefaultCoarseLevels);
            const latent_representation confidence_gated = make_activation_constant_latents(
                weights, minimum, range, rows, columns, format.block_width,
                calibration_inputs, true, true, kDefaultCoarseLevels);
            if (!std::isfinite(run_case("alpha-block-mean", weights, block_latents,
                                        rows, columns, format, inputs)) ||
                !std::isfinite(run_case("alpha-activation-optimal", weights, activation_optimal,
                                        rows, columns, format, inputs)) ||
                !std::isfinite(run_case("alpha-activation-shard-gated", weights, shard_gated,
                                        rows, columns, format, inputs)) ||
                !std::isfinite(run_case("alpha-activation-confidence-gated", weights, confidence_gated,
                                        rows, columns, format, inputs))) {
                std::fprintf(stderr, "ASTC activation Alpha sweep failed\n");
                return 1;
            }
        }
        if (decode_loop_alpha_sweep &&
            !decode_loop_alpha_search(weights, block_latents, rows, columns, format,
                                      calibration_inputs, validation_inputs, inputs, false,
                                      decode_loop_log_path, decode_loop_payloads_path, decode_loop_reference_path,
                                      validation_payload_path, validation_reference_path, validation_metadata_path,
                                      neutral_payload_path, neutral_reference_path, neutral_metadata_path,
                                      row_strip_log_path,
                                      candidate_threads, row_strip_select, row_strip_chunked,
                                      row_strip_diagnostics, persistent_worker_contexts, false,
                                      encoder_search, neural_candidate_limit)) {
            std::fprintf(stderr, "ASTC decode-in-the-loop Alpha sweep failed; use whole ASTC blocks\n");
            return 1;
        }
        if (scalar_anchored_gauge_sweep) {
            latent_representation gauge_latents = scalar_latents;
            gauge_latents.decoder = { range * 0.5, range * 0.5, minimum };
            if (!decode_loop_alpha_search(weights, gauge_latents, rows, columns, format,
                                          calibration_inputs, validation_inputs, inputs, true,
                                          decode_loop_log_path, decode_loop_payloads_path, decode_loop_reference_path,
                                          validation_payload_path, validation_reference_path, validation_metadata_path,
                                          neutral_payload_path, neutral_reference_path, neutral_metadata_path,
                                          row_strip_log_path,
                                          candidate_threads, row_strip_select, row_strip_chunked,
                                          row_strip_diagnostics, persistent_worker_contexts, false,
                                          encoder_search, neural_candidate_limit, false, 0, false, false,
                                          pv_alternate)) {
                std::fprintf(stderr, "ASTC scalar-anchored gauge sweep failed\n");
                return 1;
            }
        }
        if (scalar_anchored_c_delta_sweep) {
            latent_representation c_delta_latents = scalar_latents;
            c_delta_latents.decoder = { range * 0.5, range * 0.5, minimum };
            if (!decode_loop_alpha_search(weights, c_delta_latents, rows, columns, format,
                                          calibration_inputs, validation_inputs, inputs, true,
                                          decode_loop_log_path, decode_loop_payloads_path, decode_loop_reference_path,
                                          validation_payload_path, validation_reference_path, validation_metadata_path,
                                          neutral_payload_path, neutral_reference_path, neutral_metadata_path,
                                          row_strip_log_path,
                                          candidate_threads, row_strip_select, row_strip_chunked,
                                          row_strip_diagnostics, persistent_worker_contexts, true,
                                          encoder_search, neural_candidate_limit)) {
                std::fprintf(stderr, "ASTC scalar-anchored c+delta sweep failed\n");
                return 1;
            }
        }
        if (weight_grid_gauge_sweep) {
            latent_representation gauge_latents = scalar_latents;
            gauge_latents.decoder = { range * 0.5, range * 0.5, minimum };
            if (!decode_loop_alpha_search(weights, gauge_latents, rows, columns, format,
                                          calibration_inputs, validation_inputs, inputs, true,
                                          decode_loop_log_path, decode_loop_payloads_path, decode_loop_reference_path,
                                          validation_payload_path, validation_reference_path, validation_metadata_path,
                                          neutral_payload_path, neutral_reference_path, neutral_metadata_path,
                                          row_strip_log_path,
                                          candidate_threads, row_strip_select, row_strip_chunked,
                                          row_strip_diagnostics, persistent_worker_contexts, false,
                                          encoder_search, neural_candidate_limit, true, 0,
                                          pv_lite_grid_sweep, pv_lite_coarse_grid_sweep)) {
                std::fprintf(stderr, "ASTC weight-grid gauge sweep failed\n");
                return 1;
            }
        }
        if (few_level_weight_grid_gauge_sweep) {
            for (const uint32_t source_levels : { 3u, 5u }) {
                latent_representation gauge_latents = make_scalar_latents(
                    weights, minimum, range, source_levels);
                gauge_latents.decoder = { gauge_latents.decoder.scale_l * 0.5,
                                          gauge_latents.decoder.scale_l * 0.5,
                                          gauge_latents.decoder.offset };
                if (!decode_loop_alpha_search(weights, gauge_latents, rows, columns, format,
                                              calibration_inputs, validation_inputs, inputs, true,
                                              decode_loop_log_path, decode_loop_payloads_path,
                                              decode_loop_reference_path,
                                              validation_payload_path, validation_reference_path, validation_metadata_path,
                                              neutral_payload_path, neutral_reference_path, neutral_metadata_path,
                                              row_strip_log_path,
                                              candidate_threads, row_strip_select, row_strip_chunked,
                                              row_strip_diagnostics, persistent_worker_contexts, false,
                                              encoder_search, neural_candidate_limit, true,
                                              source_levels)) {
                    std::fprintf(stderr, "ASTC few-level weight-grid gauge sweep failed\n");
                    return 1;
                }
            }
        }
        if (residual_basis_only) continue;
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
                                     nullptr, true)) ||
             !std::isfinite(run_case("luminance-alpha-block-residual-neural-rank", weights,
                                     block_latents, rows, columns, format, inputs,
                                     nullptr, true)))) {
            std::fprintf(stderr, "ASTC latent neural-rank smoke failed\n");
            return 1;
        }
        if (coordinate_select &&
            !run_coordinate_case(weights, additive_latents, rows, columns, format,
                                 calibration_inputs, inputs, coordinate_fast_candidate, coordinate_diverse,
                                 coordinate_regularized, selector_compare, candidate_sweep)) {
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
    if (!footprint.empty() && footprint != "4x4" && footprint != "5x5" &&
        footprint != "6x6" && footprint != "8x5" && footprint != "8x6" && footprint != "10x6" && footprint != "8x8" &&
        footprint != "10x8") {
        std::fprintf(stderr, "unsupported --footprint value: %s\n", footprint.c_str());
        return 2;
    }
    return 0;
}
