#include <astcenc.h>

#include "astc-vulkan-input.h"
#include "astc-vulkan-paired.h"
#include "astc-vulkan-paired-selector.h"
#include "astc-vulkan-pv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

#if defined(ASTC_VULKAN_PAIRED_D2_10X5)
constexpr uint32_t kBlockWidth = 10;
constexpr const char * kFootprintName = "10x5";
#else
constexpr uint32_t kBlockWidth = 8;
constexpr const char * kFootprintName = "8x5";
#endif
constexpr uint32_t kPhysicalBlockHeight = 5;
constexpr uint32_t kLogicalBlockHeight = kPhysicalBlockHeight * 2;

struct params {
    std::string model;
    std::string tensor;
    std::string trace;
    uint32_t rows = 20;
    uint32_t columns = 256;
    uint32_t calibration_samples = 5;
    uint32_t validation_samples = 2;
    uint32_t progress_every_blocks = 0;
    std::string report;
    bool structure_bank = false;
    bool pv_alternate = false;
};

struct decoded_block {
    std::array<uint8_t, 16> payload{};
    std::array<float, kLogicalBlockHeight * kBlockWidth> logical_weights{};
};

struct generated_candidate {
    decoded_block block;
    astc_vulkan_paired_layout layout = astc_vulkan_paired_layout::rg_b;
    astc_vulkan_paired_steering_factor steering{};
    astc_vulkan_paired_candidate_delta delta;
    bool pv_generated = false;
};

struct run_profile {
    double source_seconds = 0.0;
    double delta_seconds = 0.0;
    double selection_seconds = 0.0;
    double objective_seconds = 0.0;
    double pv_seconds = 0.0;
    uint64_t pv_trials = 0;
    uint64_t pv_accepts = 0;
};

#if defined(ASTC_VULKAN_PAIRED_NEURAL_ENCODER)
struct structure_bank_state {
    std::vector<std::array<unsigned int, 4>> modes;
    bool constrained = false;
};

void capture_structure_bank(void * user_data, unsigned int, unsigned int, unsigned int,
                            unsigned int partition_count, unsigned int partition_index,
                            unsigned int block_mode, int plane2_component) {
    auto & state = *static_cast<structure_bank_state *>(user_data);
    const std::array<unsigned int, 4> key{partition_count, partition_index, block_mode,
                                          static_cast<unsigned int>(plane2_component + 1)};
    if (std::find(state.modes.begin(), state.modes.end(), key) == state.modes.end()) {
        state.modes.push_back(key);
    }
}

bool filter_structure_bank(void * user_data, unsigned int partition_count,
                           unsigned int partition_index, unsigned int block_mode,
                           int plane2_component) {
    const auto & state = *static_cast<const structure_bank_state *>(user_data);
    if (!state.constrained) return true;
    const std::array<unsigned int, 4> key{partition_count, partition_index, block_mode,
                                          static_cast<unsigned int>(plane2_component + 1)};
    return std::find(state.modes.begin(), state.modes.end(), key) != state.modes.end();
}
#endif

bool parse_params(int argc, char ** argv, params & result) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 == argc) return false;
        const std::string value = argv[++index];
        if (option == "--model") result.model = value;
        else if (option == "--tensor") result.tensor = value;
        else if (option == "--trace") result.trace = value;
        else if (option == "--rows") result.rows = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--columns") result.columns = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--calibration-samples") result.calibration_samples = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--validation-samples") result.validation_samples = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--progress-every-blocks") result.progress_every_blocks = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--report") result.report = value;
        else if (option == "--structure-bank") result.structure_bank = value == "1" || value == "true";
        else if (option == "--pv-alternate") result.pv_alternate = value == "1" || value == "true";
        else return false;
    }
    return !result.model.empty() && !result.tensor.empty() && !result.trace.empty() &&
           result.rows != 0 && result.columns != 0 && result.calibration_samples != 0;
}

class block_codec {
public:
    struct timing {
        double encode_seconds = 0.0;
        double decode_seconds = 0.0;
        uint64_t roundtrips = 0;
    };

    block_codec(bool neural_backend, astc_vulkan_paired_layout layout,
                const block_codec * shared_parent = nullptr, bool structure_bank = false) : layout_(layout),
                                                               decoded_scratch_(kBlockWidth * kPhysicalBlockHeight * 4) {
        astcenc_config config{};
        if (astcenc_config_init(ASTCENC_PRF_LDR, kBlockWidth, kPhysicalBlockHeight, 1,
                                ASTCENC_PRE_THOROUGH, 0, &config) == ASTCENC_SUCCESS) {
#if defined(ASTC_VULKAN_PAIRED_NEURAL_ENCODER)
            if (neural_backend) {
                config.flags |= ASTCENC_FLG_MAP_NEURAL_D2;
                config.neural_d2_layout = layout == astc_vulkan_paired_layout::rg_b ? 0u : 1u;
                if (structure_bank) {
                    config.structure_callback = capture_structure_bank;
                    config.structure_callback_user_data = &structure_bank_;
                    config.structure_filter = filter_structure_bank;
                    config.structure_filter_user_data = &structure_bank_;
                }
            }
#else
            if (neural_backend) return;
#endif
#if defined(ASTC_VULKAN_PAIRED_NEURAL_ENCODER)
            const astcenc_error shared_result = astcenc_context_alloc(&config, 1, &context_,
                shared_parent == nullptr ? nullptr : shared_parent->context_);
            if (shared_result == ASTCENC_SUCCESS) {
                ready_ = true;
            } else if (shared_parent != nullptr) {
                // The custom semantic layout does not change ASTC lookup tables, but retain a
                // standalone fallback if a future astcenc revision tightens parent matching.
                ready_ = astcenc_context_alloc(&config, 1, &context_, nullptr) == ASTCENC_SUCCESS;
            }
#else
            ready_ = astcenc_context_alloc(&config, 1, &context_) == ASTCENC_SUCCESS;
#endif
        }
    }

    ~block_codec() { if (context_ != nullptr) astcenc_context_free(context_); }
    block_codec(const block_codec &) = delete;
    block_codec & operator=(const block_codec &) = delete;

    bool ready() const { return ready_; }
    const timing & timings() const { return timings_; }

    void begin_block() {
#if defined(ASTC_VULKAN_PAIRED_NEURAL_ENCODER)
        structure_bank_.modes.clear();
        structure_bank_.constrained = false;
#endif
    }

    void commit_structure_bank() {
#if defined(ASTC_VULKAN_PAIRED_NEURAL_ENCODER)
        structure_bank_.constrained = true;
#endif
    }

    bool roundtrip(const std::vector<float> & source, decoded_block & result) {
        if (!ready_ || source.size() != kBlockWidth * kPhysicalBlockHeight * 4) return false;
        void * source_slice = const_cast<float *>(source.data());
        astcenc_image source_image{kBlockWidth, kPhysicalBlockHeight, 1, ASTCENC_TYPE_F32, &source_slice};
        const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
        const auto encode_start = std::chrono::steady_clock::now();
        if (astcenc_compress_image(context_, &source_image, &swizzle, result.payload.data(),
                                   result.payload.size(), 0) != ASTCENC_SUCCESS) return false;
        timings_.encode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - encode_start).count();
        void * decoded_slice = decoded_scratch_.data();
        astcenc_image decoded_image{kBlockWidth, kPhysicalBlockHeight, 1, ASTCENC_TYPE_F32, &decoded_slice};
        const auto decode_start = std::chrono::steady_clock::now();
        if (astcenc_decompress_image(context_, result.payload.data(), result.payload.size(),
                                     &decoded_image, &swizzle, 0) != ASTCENC_SUCCESS) return false;
        timings_.decode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
        ++timings_.roundtrips;
        for (uint32_t texel_y = 0; texel_y < kPhysicalBlockHeight; ++texel_y) {
            for (uint32_t x = 0; x < kBlockWidth; ++x) {
                const size_t offset = (static_cast<size_t>(texel_y) * kBlockWidth + x) * 4;
                const astc_vulkan_rgba_texel texel{decoded_scratch_[offset], decoded_scratch_[offset + 1],
                                                   decoded_scratch_[offset + 2], decoded_scratch_[offset + 3]};
                result.logical_weights[(2 * texel_y) * kBlockWidth + x] =
                    astc_vulkan_paired_weight(texel, 0, layout_);
                result.logical_weights[(2 * texel_y + 1) * kBlockWidth + x] =
                    astc_vulkan_paired_weight(texel, 1, layout_);
            }
        }
        return true;
    }

private:
    astc_vulkan_paired_layout layout_ = astc_vulkan_paired_layout::rg_b;
    astcenc_context * context_ = nullptr;
    std::vector<float> decoded_scratch_;
    timing timings_{};
    bool ready_ = false;
#if defined(ASTC_VULKAN_PAIRED_NEURAL_ENCODER)
    structure_bank_state structure_bank_;
#endif
};

float normalized_weight(const ggml_vk_astc_loaded_matrix & matrix, uint32_t row, uint32_t column,
                        float minimum, float range) {
    if (row >= matrix.rows || column >= matrix.columns) return 0.5f;
    const float value = matrix.values[static_cast<size_t>(row) * matrix.columns + column];
    return std::clamp((value - minimum) / (range > 0.0f ? range : 1.0f), 0.0f, 1.0f);
}

void fill_source_block_with_steering(std::vector<float> & source,
                       const ggml_vk_astc_loaded_matrix & matrix, uint32_t row0, uint32_t column0,
                       uint32_t rows, uint32_t columns, float minimum, float range,
                       astc_vulkan_paired_layout layout,
                       const std::function<float(float, float)> & steering_value) {
    if (source.size() != kBlockWidth * kPhysicalBlockHeight * 4) {
        source.resize(kBlockWidth * kPhysicalBlockHeight * 4);
    }
    for (uint32_t y = 0; y < kPhysicalBlockHeight; ++y) {
        for (uint32_t x_index = 0; x_index < kBlockWidth; ++x_index) {
            const uint32_t logical_row0 = row0 + 2 * y;
            const uint32_t logical_row1 = logical_row0 + 1;
            const uint32_t column = column0 + x_index;
            const float x = 2.0f * static_cast<float>(x_index) / (kBlockWidth - 1) - 1.0f;
            const float y_value = 2.0f * static_cast<float>(y) / (kPhysicalBlockHeight - 1) - 1.0f;
            const float alpha = std::clamp(0.5f + steering_value(x, y_value), 0.0f, 1.0f);
            const float q0 = logical_row0 < rows && column < columns ?
                normalized_weight(matrix, logical_row0, column, minimum, range) : 0.5f;
            const float q1 = logical_row1 < rows && column < columns ?
                normalized_weight(matrix, logical_row1, column, minimum, range) : 0.5f;
            const auto texel = astc_vulkan_make_paired_texel(q0, q1, alpha, layout);
            const size_t offset = (static_cast<size_t>(y) * kBlockWidth + x_index) * 4;
            source[offset] = texel.r;
            source[offset + 1] = texel.g;
            source[offset + 2] = texel.b;
            source[offset + 3] = texel.a;
        }
    }
}

void fill_source_block(std::vector<float> & source, const ggml_vk_astc_loaded_matrix & matrix,
                       uint32_t row0, uint32_t column0, uint32_t rows, uint32_t columns,
                       float minimum, float range, astc_vulkan_paired_layout layout,
                       const astc_vulkan_paired_steering_factor & steering) {
    fill_source_block_with_steering(source, matrix, row0, column0, rows, columns, minimum, range,
        layout, [&steering](float x, float y) {
            return steering.amplitude * astc_vulkan_paired_steering_basis_value(steering.basis, x, y);
        });
}

void write_block(std::vector<float> & target, const decoded_block & block, uint32_t block_row,
                 uint32_t block_column, uint32_t rows, uint32_t columns) {
    for (uint32_t y = 0; y < kLogicalBlockHeight; ++y) {
        const uint32_t row = block_row * kLogicalBlockHeight + y;
        if (row >= rows) continue;
        for (uint32_t x = 0; x < kBlockWidth; ++x) {
            const uint32_t column = block_column * kBlockWidth + x;
            if (column >= columns) continue;
            target[static_cast<size_t>(row) * columns + column] = block.logical_weights[y * kBlockWidth + x];
        }
    }
}

std::vector<double> output_error(const ggml_vk_astc_loaded_matrix & matrix,
                                 const ggml_vk_astc_activation_trace & trace,
                                 const std::vector<float> & decoded, uint32_t rows,
                                 uint32_t columns, uint32_t sample_offset, uint32_t samples,
                                 float minimum, float range) {
    std::vector<double> result(static_cast<size_t>(samples) * rows);
    for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t row = 0; row < rows; ++row) {
            double value = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                const float source = normalized_weight(matrix, row, column, minimum, range);
                const float difference = (source - decoded[static_cast<size_t>(row) * columns + column]) * range;
                value += difference * trace.values[static_cast<size_t>(sample + sample_offset) * trace.columns + column];
            }
            result[static_cast<size_t>(sample) * rows + row] = value;
        }
    }
    return result;
}

double mse(const std::vector<double> & error) {
    if (error.empty()) return NAN;
    double total = 0.0;
    for (double value : error) total += value * value;
    return total / static_cast<double>(error.size());
}

void assign_delta(std::vector<double> & delta, const decoded_block & candidate,
                  const decoded_block & baseline, const ggml_vk_astc_activation_trace & trace,
                  uint32_t block_row, uint32_t block_column, uint32_t rows, uint32_t columns,
                  uint32_t sample_offset, uint32_t samples, float range) {
    delta.assign(static_cast<size_t>(samples) * rows, 0.0);
    for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t local_row = 0; local_row < kLogicalBlockHeight; ++local_row) {
            const uint32_t row = block_row * kLogicalBlockHeight + local_row;
            if (row >= rows) continue;
            double output_delta = 0.0;
            for (uint32_t x = 0; x < kBlockWidth; ++x) {
                const uint32_t column = block_column * kBlockWidth + x;
                if (column >= columns) continue;
                const float weight_delta = (candidate.logical_weights[local_row * kBlockWidth + x] -
                                            baseline.logical_weights[local_row * kBlockWidth + x]) * range;
                output_delta += weight_delta * trace.values[static_cast<size_t>(sample + sample_offset) * trace.columns + column];
            }
            delta[static_cast<size_t>(sample) * rows + row] = output_delta;
        }
    }
}

// Local activation objective used only to seed the bounded D2 PV search. The
// final decision is still made by the shared paired selector, so this does
// not change the deployed objective or commit order.
double block_activation_objective(const ggml_vk_astc_loaded_matrix & matrix,
                                  const ggml_vk_astc_activation_trace & trace,
                                  const decoded_block & candidate, uint32_t block_row,
                                  uint32_t block_column, uint32_t rows, uint32_t columns,
                                  uint32_t sample_offset, uint32_t samples, float minimum,
                                  float range) {
    double loss = 0.0;
    uint64_t terms = 0;
    for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t local_row = 0; local_row < kLogicalBlockHeight; ++local_row) {
            const uint32_t row = block_row * kLogicalBlockHeight + local_row;
            if (row >= rows) continue;
            double output = 0.0;
            double reference = 0.0;
            for (uint32_t x = 0; x < kBlockWidth; ++x) {
                const uint32_t column = block_column * kBlockWidth + x;
                if (column >= columns) continue;
                const float activation = trace.values[static_cast<size_t>(sample + sample_offset) * trace.columns + column];
                const float source = normalized_weight(matrix, row, column, minimum, range) * range;
                const float decoded = candidate.logical_weights[local_row * kBlockWidth + x] * range;
                reference += source * activation;
                output += decoded * activation;
            }
            const double error = reference - output;
            loss += error * error;
            ++terms;
        }
    }
    return terms == 0 ? std::numeric_limits<double>::infinity() : loss / static_cast<double>(terms);
}

bool same_candidate(const generated_candidate & lhs, const generated_candidate & rhs) {
    return lhs.layout == rhs.layout && lhs.block.payload == rhs.block.payload;
}

} // namespace

int main(int argc, char ** argv) {
    params options;
    if (!parse_params(argc, argv, options)) {
        std::fprintf(stderr, "usage: %s --model model.gguf --tensor name --trace input.trace "
                             "[--rows N --columns N --calibration-samples N --validation-samples N "
                             "--progress-every-blocks N --report path [--structure-bank 1] "
                             "[--pv-alternate 1]\n", argv[0]);
        return 2;
    }
    ggml_vk_astc_loaded_matrix matrix;
    ggml_vk_astc_activation_trace trace;
    std::string error;
    if (!ggml_vk_astc_load_gguf_matrix(options.model, options.tensor, matrix, error) ||
        !ggml_vk_astc_load_activation_trace(options.trace, trace, error) ||
        options.rows > matrix.rows || options.columns > matrix.columns || options.columns > trace.columns ||
        options.calibration_samples + options.validation_samples >= trace.samples) {
        std::fprintf(stderr, "paired selection smoke input error: %s\n", error.c_str());
        return 1;
    }

    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint32_t row = 0; row < options.rows; ++row) for (uint32_t column = 0; column < options.columns; ++column) {
        const float value = matrix.values[static_cast<size_t>(row) * matrix.columns + column];
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const float range = maximum - minimum;
    const uint32_t blocks_x = (options.columns + kBlockWidth - 1) / kBlockWidth;
    const uint32_t blocks_y = (options.rows + kLogicalBlockHeight - 1) / kLogicalBlockHeight;
    const uint32_t validation_offset = options.calibration_samples;
    const uint32_t holdout_offset = validation_offset + options.validation_samples;
    const uint32_t holdout_samples = trace.samples - holdout_offset;

    constexpr bool neural_backend =
#if defined(ASTC_VULKAN_PAIRED_NEURAL_ENCODER)
        true;
#else
        false;
#endif
    std::vector<std::vector<generated_candidate>> generated(static_cast<size_t>(blocks_x) * blocks_y);
    std::vector<float> baseline(static_cast<size_t>(options.rows) * options.columns);
    std::vector<float> source_scratch(kBlockWidth * kPhysicalBlockHeight * 4);
    const auto codebook = astc_vulkan_make_paired_steering_codebook();
    block_codec rg_b_codec(neural_backend, astc_vulkan_paired_layout::rg_b, nullptr, options.structure_bank);
    block_codec r_gb_codec(neural_backend, astc_vulkan_paired_layout::r_gb, &rg_b_codec, options.structure_bank);
    if (!rg_b_codec.ready() || !r_gb_codec.ready()) return 1;
    run_profile profile;
    uint64_t raw_candidates = 0;
    uint32_t completed_blocks = 0;
    const auto generation_start = std::chrono::steady_clock::now();
    for (uint32_t block_y = 0; block_y < blocks_y; ++block_y) {
        for (uint32_t block_x = 0; block_x < blocks_x; ++block_x) {
            auto & candidates = generated[static_cast<size_t>(block_y) * blocks_x + block_x];
            for (const auto layout : {astc_vulkan_paired_layout::rg_b, astc_vulkan_paired_layout::r_gb}) {
                block_codec & codec = layout == astc_vulkan_paired_layout::rg_b ? rg_b_codec : r_gb_codec;
                codec.begin_block();
                for (const auto & steering : codebook) {
                    generated_candidate candidate;
                    candidate.layout = layout;
                    candidate.steering = steering;
                    const auto source_start = std::chrono::steady_clock::now();
                    fill_source_block(source_scratch, matrix, block_y * kLogicalBlockHeight, block_x * kBlockWidth,
                        options.rows, options.columns, minimum, range, layout, steering);
                    profile.source_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - source_start).count();
                    if (!codec.roundtrip(source_scratch, candidate.block)) return 1;
                    if (steering.basis == astc_vulkan_paired_steering_basis::neutral &&
                        steering.amplitude == 0.0f) {
                        codec.commit_structure_bank();
                    }
                    candidate.delta.payload = candidate.block.payload;
                    ++raw_candidates;
                    bool duplicate = false;
                    for (const auto & existing : candidates) duplicate = duplicate || same_candidate(existing, candidate);
                    if (!duplicate) candidates.push_back(std::move(candidate));
                }
                if (options.pv_alternate) {
                    // A deliberately small PV family: two continuous steering
                    // coordinates (x/y ramps), projected through the exact
                    // paired ASTC codec on every trial. This is a candidate
                    // generator only; global selection remains unchanged.
                    decoded_block pv_block;
                    std::vector<float> pv_source;
                    const auto pv_start = std::chrono::steady_clock::now();
                    astc_vulkan_pv_result pv_result;
                    const bool pv_ok = astc_vulkan_pv_alternate_batched(
                        {0.0f, 0.0f}, {0.25f, 0.25f}, 2,
                        [&](const std::vector<float> & coefficients, std::vector<float> & deployed) {
                            if (coefficients.size() != 2) return false;
                            fill_source_block_with_steering(
                                pv_source, matrix, block_y * kLogicalBlockHeight,
                                block_x * kBlockWidth, options.rows, options.columns,
                                minimum, range, layout,
                                [&coefficients](float x, float y) {
                                    return coefficients[0] * x + coefficients[1] * y;
                                });
                            if (!codec.roundtrip(pv_source, pv_block)) return false;
                            deployed.assign(pv_block.logical_weights.begin(), pv_block.logical_weights.end());
                            return true;
                        },
                        [&](const std::vector<std::vector<float>> & deployed,
                            std::vector<double> & scores) {
                            scores.clear();
                            for (const auto & values : deployed) {
                                if (values.size() != pv_block.logical_weights.size()) return false;
                                decoded_block trial = pv_block;
                                std::copy(values.begin(), values.end(), trial.logical_weights.begin());
                                scores.push_back(block_activation_objective(
                                    matrix, trace, trial, block_y, block_x, options.rows,
                                    options.columns, 0, options.calibration_samples, minimum, range));
                            }
                            return true;
                        }, pv_result);
                    profile.pv_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - pv_start).count();
                    profile.pv_trials += static_cast<uint64_t>(pv_result.iterations) * 4u + 1u;
                    profile.pv_accepts += pv_result.accepted_steps;
                    if (pv_ok && pv_result.deployed.size() == pv_block.logical_weights.size()) {
                        // Re-project the accepted point so the payload and
                        // decoded weights are exactly those being evaluated.
                        if (pv_result.continuous.size() == 2 &&
                            [&]() {
                                fill_source_block_with_steering(
                                    pv_source, matrix, block_y * kLogicalBlockHeight,
                                    block_x * kBlockWidth, options.rows, options.columns,
                                    minimum, range, layout,
                                    [&pv_result](float x, float y) {
                                        return pv_result.continuous[0] * x + pv_result.continuous[1] * y;
                                    });
                                return codec.roundtrip(pv_source, pv_block);
                            }()) {
                            generated_candidate candidate;
                            candidate.layout = layout;
                            candidate.steering = {pv_result.continuous[0],
                                astc_vulkan_paired_steering_basis::x_ramp};
                            candidate.block = pv_block;
                            candidate.pv_generated = true;
                            bool duplicate = false;
                            for (const auto & existing : candidates) duplicate = duplicate || same_candidate(existing, candidate);
                            if (!duplicate) candidates.push_back(std::move(candidate));
                        }
                    }
                }
            }
            // A deterministic neutral RG/B candidate is mandatory at index zero.
            const auto neutral = std::find_if(candidates.begin(), candidates.end(), [](const generated_candidate & candidate) {
                return candidate.layout == astc_vulkan_paired_layout::rg_b &&
                       candidate.steering.basis == astc_vulkan_paired_steering_basis::neutral;
            });
            if (neutral == candidates.end()) return 1;
            std::iter_swap(candidates.begin(), neutral);
            write_block(baseline, candidates.front().block, block_y, block_x, options.rows, options.columns);
            ++completed_blocks;
            if (options.progress_every_blocks != 0 &&
                (completed_blocks % options.progress_every_blocks == 0 || completed_blocks == blocks_x * blocks_y)) {
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - generation_start).count();
                std::printf("paired-select progress backend=%s blocks=%u/%u candidates=%llu elapsed=%.1fs\n",
                            neural_backend ? "neural-d2" : "standard", completed_blocks, blocks_x * blocks_y,
                            static_cast<unsigned long long>(raw_candidates), seconds);
                std::fflush(stdout);
            }
        }
    }

    const auto initial_objective_start = std::chrono::steady_clock::now();
    const auto initial_calibration = output_error(matrix, trace, baseline, options.rows, options.columns,
        0, options.calibration_samples, minimum, range);
    const auto initial_validation = output_error(matrix, trace, baseline, options.rows, options.columns,
        validation_offset, options.validation_samples, minimum, range);
    profile.objective_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - initial_objective_start).count();
    for (uint32_t block_y = 0; block_y < blocks_y; ++block_y) for (uint32_t block_x = 0; block_x < blocks_x; ++block_x) {
        auto & candidates = generated[static_cast<size_t>(block_y) * blocks_x + block_x];
        const decoded_block & neutral = candidates.front().block;
        for (auto & candidate : candidates) {
            const auto delta_start = std::chrono::steady_clock::now();
            assign_delta(candidate.delta.calibration_delta, candidate.block, neutral, trace, block_y, block_x,
                         options.rows, options.columns, 0, options.calibration_samples, range);
            assign_delta(candidate.delta.validation_delta, candidate.block, neutral, trace, block_y, block_x,
                         options.rows, options.columns, validation_offset, options.validation_samples, range);
            profile.delta_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - delta_start).count();
        }
    }
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> selector_candidates(generated.size());
    for (size_t block = 0; block < generated.size(); ++block) {
        for (const auto & candidate : generated[block]) selector_candidates[block].push_back(candidate.delta);
    }
    astc_vulkan_paired_selection_result selection;
    const astc_vulkan_paired_selector_config config{options.rows, options.calibration_samples, options.validation_samples};
    const auto selection_start = std::chrono::steady_clock::now();
    if (!astc_vulkan_select_paired_candidates(config, initial_calibration, initial_validation,
                                               selector_candidates, selection)) return 1;
    profile.selection_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - selection_start).count();

    std::vector<float> selected = baseline;
    for (uint32_t block_y = 0; block_y < blocks_y; ++block_y) for (uint32_t block_x = 0; block_x < blocks_x; ++block_x) {
        const size_t block = static_cast<size_t>(block_y) * blocks_x + block_x;
        write_block(selected, generated[block][selection.validation_selected_candidates[block]].block,
                    block_y, block_x, options.rows, options.columns);
    }
    const auto final_objective_start = std::chrono::steady_clock::now();
    const auto baseline_holdout = output_error(matrix, trace, baseline, options.rows, options.columns,
        holdout_offset, holdout_samples, minimum, range);
    const auto selected_holdout = output_error(matrix, trace, selected, options.rows, options.columns,
        holdout_offset, holdout_samples, minimum, range);
    profile.objective_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - final_objective_start).count();
    uint64_t unique_candidates = 0;
    for (const auto & candidates : generated) unique_candidates += candidates.size();
    const double neutral_calibration_mse = mse(initial_calibration);
    const double neutral_validation_mse = mse(initial_validation);
    const double neutral_holdout_mse = mse(baseline_holdout);
    const double selected_calibration_mse = selection.calibration_residual_loss / (options.calibration_samples * options.rows);
    const double selected_validation_mse = selection.validation_residual_loss / (options.validation_samples * options.rows);
    const double selected_holdout_mse = mse(selected_holdout);
    std::printf("paired-select D2_%s rows=%u columns=%u blocks=%u raw=%llu unique=%llu\n",
                kFootprintName,
                options.rows, options.columns, blocks_x * blocks_y,
                static_cast<unsigned long long>(raw_candidates), static_cast<unsigned long long>(unique_candidates));
    std::printf("paired-select neutral calibration-mse=%.8g validation-mse=%.8g holdout-mse=%.8g\n",
                neutral_calibration_mse, neutral_validation_mse, neutral_holdout_mse);
    std::printf("paired-select selected commits=%zu validation-prefix=%u calibration-mse=%.8g validation-mse=%.8g holdout-mse=%.8g\n",
                selection.commits.size(), selection.validation_prefix,
                selected_calibration_mse, selected_validation_mse, selected_holdout_mse);
    const auto & rg_b_timing = rg_b_codec.timings();
    const auto & r_gb_timing = r_gb_codec.timings();
    std::printf("paired-select profile source=%.3fs encode=%.3fs decode=%.3fs delta=%.3fs selection=%.3fs objective=%.3fs roundtrips=%llu\n",
                profile.source_seconds, rg_b_timing.encode_seconds + r_gb_timing.encode_seconds,
                rg_b_timing.decode_seconds + r_gb_timing.decode_seconds, profile.delta_seconds,
                profile.selection_seconds, profile.objective_seconds,
                static_cast<unsigned long long>(rg_b_timing.roundtrips + r_gb_timing.roundtrips));
    std::printf("paired-select pv enabled=%s trials=%llu accepts=%llu seconds=%.3fs\n",
                options.pv_alternate ? "true" : "false",
                static_cast<unsigned long long>(profile.pv_trials),
                static_cast<unsigned long long>(profile.pv_accepts), profile.pv_seconds);
    if (!options.report.empty()) {
        std::ofstream report(options.report);
        if (!report) return 1;
        report << "backend=" << (neural_backend ? "neural-d2" : "standard") << '\n'
               << "footprint=" << kFootprintName << '\n'
               << "rows=" << options.rows << '\n'
               << "columns=" << options.columns << '\n'
               << "blocks=" << blocks_x * blocks_y << '\n'
               << "raw_candidates=" << raw_candidates << '\n'
               << "unique_candidates=" << unique_candidates << '\n'
               << "neutral_calibration_mse=" << neutral_calibration_mse << '\n'
               << "neutral_validation_mse=" << neutral_validation_mse << '\n'
               << "neutral_holdout_mse=" << neutral_holdout_mse << '\n'
               << "commits=" << selection.commits.size() << '\n'
               << "validation_prefix=" << selection.validation_prefix << '\n'
               << "selected_calibration_mse=" << selected_calibration_mse << '\n'
               << "selected_validation_mse=" << selected_validation_mse << '\n'
               << "selected_holdout_mse=" << selected_holdout_mse << '\n'
               << "profile_source_seconds=" << profile.source_seconds << '\n'
               << "profile_encode_seconds=" << rg_b_timing.encode_seconds + r_gb_timing.encode_seconds << '\n'
               << "profile_decode_seconds=" << rg_b_timing.decode_seconds + r_gb_timing.decode_seconds << '\n'
               << "profile_delta_seconds=" << profile.delta_seconds << '\n'
               << "profile_selection_seconds=" << profile.selection_seconds << '\n'
               << "profile_objective_seconds=" << profile.objective_seconds << '\n'
               << "pv_alternate=" << (options.pv_alternate ? 1 : 0) << '\n'
               << "pv_trials=" << profile.pv_trials << '\n'
               << "pv_accepts=" << profile.pv_accepts << '\n'
               << "profile_pv_seconds=" << profile.pv_seconds << '\n'
               << "roundtrips=" << rg_b_timing.roundtrips + r_gb_timing.roundtrips << '\n';
        if (!report) return 1;
    }
    return 0;
}
