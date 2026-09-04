#include <astcenc.h>

#include "astc-vulkan-input.h"
#include "astc-vulkan-objective.h"
#include "astc-vulkan-paired.h"
#include "astc-vulkan-paired-layout.h"
#include "astc-vulkan-paired-selector.h"
#include "astc-vulkan-pv.h"
#include "astc-vulkan-yaqa.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <utility>
#include <vector>

namespace {

#if defined(ASTC_VULKAN_PAIRED_D2_10X5)
constexpr uint32_t kBlockWidth = 10;
constexpr const char * kFootprintName = "10x5";
constexpr astc_vulkan_footprint kFootprint = astc_vulkan_footprint::k10x5;
#else
constexpr uint32_t kBlockWidth = 8;
constexpr const char * kFootprintName = "8x5";
constexpr astc_vulkan_footprint kFootprint = astc_vulkan_footprint::k8x5;
#endif
constexpr uint32_t kPhysicalBlockHeight = 5;
constexpr uint32_t kLogicalBlockHeight = kPhysicalBlockHeight * 2;

enum class d2_channel_weight_profile : uint8_t {
    legacy,
    balanced_alpha_025,
    balanced_alpha_050,
};

enum class d2_alpha_source_kind : uint8_t {
    geometric,
    q0,
    q1,
    mean,
    difference,
};

enum class d2_basis_selection : uint8_t {
    direct,
    common_difference,
};

struct d2_source_candidate {
    astc_vulkan_paired_steering_factor steering{};
    d2_alpha_source_kind alpha_source = d2_alpha_source_kind::geometric;
};

struct params {
    std::string model;
    std::string tensor;
    std::string trace;
    std::string output_trace;
    uint32_t rows = 20;
    uint32_t columns = 256;
    uint32_t calibration_samples = 5;
    uint32_t validation_samples = 2;
    uint32_t progress_every_blocks = 0;
    std::string report;
    std::string export_payload;
    std::string export_layout;
    bool structure_bank = false;
    bool pv_alternate = false;
    bool row_strip_chunked = false;
    d2_channel_weight_profile channel_weights = d2_channel_weight_profile::legacy;
    bool source_derived_alpha = false;
    d2_basis_selection paired_basis = d2_basis_selection::direct;
    astc_vulkan_objective objective = astc_vulkan_objective::activation;
};

struct decoded_block {
    std::array<uint8_t, 16> payload{};
    std::array<float, kLogicalBlockHeight * kBlockWidth> logical_weights{};
    astcenc_block_info info{};
    bool info_valid = false;
};

struct generated_candidate {
    decoded_block block;
    astc_vulkan_paired_layout layout = astc_vulkan_paired_layout::rg_b;
    astc_vulkan_paired_steering_factor steering{};
    d2_alpha_source_kind alpha_source = d2_alpha_source_kind::geometric;
    astc_vulkan_paired_basis basis = astc_vulkan_paired_basis::direct;
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

const char * channel_weight_profile_name(d2_channel_weight_profile profile) {
    switch (profile) {
        case d2_channel_weight_profile::legacy: return "legacy";
        case d2_channel_weight_profile::balanced_alpha_025: return "balanced-a025";
        case d2_channel_weight_profile::balanced_alpha_050: return "balanced-a050";
    }
    return "unknown";
}

const char * paired_basis_selection_name(d2_basis_selection selection) {
    switch (selection) {
        case d2_basis_selection::direct: return "direct";
        case d2_basis_selection::common_difference: return "common-difference";
    }
    return "unknown";
}

bool parse_channel_weight_profile(const std::string & value,
                                  d2_channel_weight_profile & profile) {
    if (value == "legacy") profile = d2_channel_weight_profile::legacy;
    else if (value == "balanced-a025") profile = d2_channel_weight_profile::balanced_alpha_025;
    else if (value == "balanced-a050") profile = d2_channel_weight_profile::balanced_alpha_050;
    else return false;
    return true;
}

bool parse_paired_basis_selection(const std::string & value,
                                  d2_basis_selection & selection) {
    if (value == "direct") selection = d2_basis_selection::direct;
    else if (value == "common-difference") selection = d2_basis_selection::common_difference;
    else return false;
    return true;
}

std::vector<astc_vulkan_paired_basis> selected_paired_bases(d2_basis_selection selection) {
    if (selection == d2_basis_selection::direct) return {astc_vulkan_paired_basis::direct};
    // The direct RG/B neutral block is the scalar-anchored reference for every
    // selector run. Common/difference is therefore an added legal candidate
    // family, never a replacement baseline.
    if (selection == d2_basis_selection::common_difference) {
        return {astc_vulkan_paired_basis::direct, astc_vulkan_paired_basis::common_difference};
    }
    return {astc_vulkan_paired_basis::direct};
}

bool parse_params(int argc, char ** argv, params & result) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 == argc) return false;
        const std::string value = argv[++index];
        if (option == "--model") result.model = value;
        else if (option == "--tensor") result.tensor = value;
        else if (option == "--trace") result.trace = value;
        else if (option == "--output-trace") result.output_trace = value;
        else if (option == "--rows") result.rows = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--columns") result.columns = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--calibration-samples") result.calibration_samples = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--validation-samples") result.validation_samples = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--progress-every-blocks") result.progress_every_blocks = static_cast<uint32_t>(std::stoul(value));
        else if (option == "--report") result.report = value;
        else if (option == "--export-payload") result.export_payload = value;
        else if (option == "--export-layout") result.export_layout = value;
        else if (option == "--structure-bank") result.structure_bank = value == "1" || value == "true";
        else if (option == "--pv-alternate") result.pv_alternate = value == "1" || value == "true";
        else if (option == "--row-strip-chunked") result.row_strip_chunked = value == "1" || value == "true";
        else if (option == "--channel-weights") {
            if (!parse_channel_weight_profile(value, result.channel_weights)) return false;
        }
        else if (option == "--source-derived-alpha") result.source_derived_alpha = value == "1" || value == "true";
        else if (option == "--paired-basis") {
            if (!parse_paired_basis_selection(value, result.paired_basis)) return false;
        }
        else if (option == "--objective") {
            if (value == "activation") result.objective = astc_vulkan_objective::activation;
            else if (value == "yaqa") result.objective = astc_vulkan_objective::two_sided_trace;
            else return false;
        }
        else return false;
    }
    return !result.model.empty() && !result.tensor.empty() && !result.trace.empty() &&
           result.rows != 0 && result.columns != 0 && result.calibration_samples != 0 &&
           (result.objective != astc_vulkan_objective::two_sided_trace || !result.output_trace.empty());
}

class block_codec {
public:
    struct timing {
        double encode_seconds = 0.0;
        double decode_seconds = 0.0;
        uint64_t roundtrips = 0;
    };

    block_codec(bool neural_backend, astc_vulkan_paired_layout layout,
                d2_channel_weight_profile channel_weights,
                const block_codec * shared_parent = nullptr, bool structure_bank = false) : layout_(layout),
                                                               decoded_scratch_(kBlockWidth * kPhysicalBlockHeight * 4) {
        astcenc_config config{};
        if (astcenc_config_init(ASTCENC_PRF_LDR, kBlockWidth, kPhysicalBlockHeight, 1,
                                ASTCENC_PRE_THOROUGH, 0, &config) == ASTCENC_SUCCESS) {
            apply_channel_weights(config, layout, channel_weights);
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

    bool roundtrip(const std::vector<float> & source, decoded_block & result,
                   astc_vulkan_paired_basis basis = astc_vulkan_paired_basis::direct) {
        if (!ready_ || source.size() != kBlockWidth * kPhysicalBlockHeight * 4) return false;
        void * source_slice = const_cast<float *>(source.data());
        astcenc_image source_image{kBlockWidth, kPhysicalBlockHeight, 1, ASTCENC_TYPE_F32, &source_slice};
        const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
        const auto encode_start = std::chrono::steady_clock::now();
        if (astcenc_compress_image(context_, &source_image, &swizzle, result.payload.data(),
                                   result.payload.size(), 0) != ASTCENC_SUCCESS) return false;
        timings_.encode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - encode_start).count();
        result.info_valid = astcenc_get_block_info(context_, result.payload.data(), &result.info) == ASTCENC_SUCCESS;
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
                    astc_vulkan_paired_weight(texel, 0, layout_, basis);
                result.logical_weights[(2 * texel_y + 1) * kBlockWidth + x] =
                    astc_vulkan_paired_weight(texel, 1, layout_, basis);
            }
        }
        return true;
    }

private:
    static void apply_channel_weights(astcenc_config & config,
                                      astc_vulkan_paired_layout layout,
                                      d2_channel_weight_profile profile) {
        if (profile == d2_channel_weight_profile::legacy) return;
        const float alpha_weight = profile == d2_channel_weight_profile::balanced_alpha_025 ?
            0.25f : 0.50f;
        if (layout == astc_vulkan_paired_layout::rg_b) {
            config.cw_r_weight = 0.5f;
            config.cw_g_weight = 0.5f;
            config.cw_b_weight = 1.0f;
        } else {
            config.cw_r_weight = 1.0f;
            config.cw_g_weight = 0.5f;
            config.cw_b_weight = 0.5f;
        }
        config.cw_a_weight = alpha_weight;
    }

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

std::vector<d2_source_candidate> make_source_candidates(bool source_derived_alpha) {
    const auto geometric = astc_vulkan_make_paired_steering_codebook();
    std::vector<d2_source_candidate> result;
    if (!source_derived_alpha) {
        result.reserve(geometric.size());
        for (const auto & steering : geometric) result.push_back({steering, d2_alpha_source_kind::geometric});
        return result;
    }

    // Keep exactly eleven probes per layout: neutral and the three lowest-order
    // signed geometric bases, plus four source-derived Alpha signals. This is a
    // replacement experiment, not a wider candidate search.
    for (const auto & steering : geometric) {
        if (steering.basis == astc_vulkan_paired_steering_basis::x_plus_y ||
            steering.basis == astc_vulkan_paired_steering_basis::x_minus_y) continue;
        result.push_back({steering, d2_alpha_source_kind::geometric});
    }
    result.push_back({{}, d2_alpha_source_kind::q0});
    result.push_back({{}, d2_alpha_source_kind::q1});
    result.push_back({{}, d2_alpha_source_kind::mean});
    result.push_back({{}, d2_alpha_source_kind::difference});
    return result;
}

float source_alpha_value(const d2_source_candidate & candidate, float q0, float q1,
                         float x, float y) {
    switch (candidate.alpha_source) {
        case d2_alpha_source_kind::geometric:
            return 0.5f + candidate.steering.amplitude *
                astc_vulkan_paired_steering_basis_value(candidate.steering.basis, x, y);
        case d2_alpha_source_kind::q0: return q0;
        case d2_alpha_source_kind::q1: return q1;
        case d2_alpha_source_kind::mean: return 0.5f * (q0 + q1);
        case d2_alpha_source_kind::difference: return 0.5f + 0.5f * (q0 - q1);
    }
    return 0.5f;
}

void fill_source_block_with_steering(std::vector<float> & source,
                       const ggml_vk_astc_loaded_matrix & matrix, uint32_t row0, uint32_t column0,
                       uint32_t rows, uint32_t columns, float minimum, float range,
                       astc_vulkan_paired_layout layout,
                       astc_vulkan_paired_basis basis,
                       const std::function<float(float, float, float, float)> & steering_value) {
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
            const float q0 = logical_row0 < rows && column < columns ?
                normalized_weight(matrix, logical_row0, column, minimum, range) : 0.5f;
            const float q1 = logical_row1 < rows && column < columns ?
                normalized_weight(matrix, logical_row1, column, minimum, range) : 0.5f;
            const float alpha = std::clamp(steering_value(q0, q1, x, y_value), 0.0f, 1.0f);
            const auto texel = astc_vulkan_make_paired_texel(q0, q1, alpha, layout, basis);
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
                       astc_vulkan_paired_basis basis, const d2_source_candidate & candidate) {
    fill_source_block_with_steering(source, matrix, row0, column0, rows, columns, minimum, range,
        layout, basis, [&candidate](float q0, float q1, float x, float y) {
            return source_alpha_value(candidate, q0, q1, x, y);
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
    return lhs.layout == rhs.layout && lhs.basis == rhs.basis && lhs.block.payload == rhs.block.payload;
}

struct yaqa_candidate_delta {
    std::vector<double> calibration;
    std::vector<double> validation;
};

bool make_yaqa_trace_window(const ggml_vk_astc_activation_trace & input,
                            const ggml_vk_astc_activation_trace & output,
                            uint32_t sample_offset, uint32_t samples,
                            uint32_t rows, uint32_t columns,
                            std::vector<float> & input_window,
                            std::vector<float> & output_window) {
    if (sample_offset + samples > input.samples || sample_offset + samples > output.samples ||
        columns > input.columns || rows > output.columns || samples == 0) return false;
    input_window.resize(static_cast<size_t>(samples) * columns);
    output_window.resize(static_cast<size_t>(samples) * rows);
    for (uint32_t sample = 0; sample < samples; ++sample) {
        std::copy_n(input.values.data() + static_cast<size_t>(sample_offset + sample) * input.columns,
                    columns, input_window.data() + static_cast<size_t>(sample) * columns);
        std::copy_n(output.values.data() + static_cast<size_t>(sample_offset + sample) * output.columns,
                    rows, output_window.data() + static_cast<size_t>(sample) * rows);
    }
    return true;
}

std::vector<double> yaqa_residual(const ggml_vk_astc_loaded_matrix & matrix,
                                  const std::vector<float> & decoded, uint32_t rows,
                                  uint32_t columns, float minimum, float range,
                                  const std::vector<float> & input_window,
                                  const std::vector<float> & output_window,
                                  uint32_t samples) {
    std::vector<double> projected(static_cast<size_t>(rows) * samples, 0.0);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t sample = 0; sample < samples; ++sample) {
            double value = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                const float error = (normalized_weight(matrix, row, column, minimum, range) -
                    decoded[static_cast<size_t>(row) * columns + column]) * range;
                value += error * input_window[static_cast<size_t>(sample) * columns + column];
            }
            projected[static_cast<size_t>(row) * samples + sample] = value;
        }
    }
    std::vector<double> residual(static_cast<size_t>(samples) * samples, 0.0);
    for (uint32_t output_sample = 0; output_sample < samples; ++output_sample) {
        for (uint32_t input_sample = 0; input_sample < samples; ++input_sample) {
            double value = 0.0;
            for (uint32_t row = 0; row < rows; ++row) {
                value += output_window[static_cast<size_t>(output_sample) * rows + row] *
                    projected[static_cast<size_t>(row) * samples + input_sample];
            }
            residual[static_cast<size_t>(output_sample) * samples + input_sample] = value;
        }
    }
    return residual;
}

double yaqa_trace_score_from_decoded(const ggml_vk_astc_loaded_matrix & matrix,
                                     const std::vector<float> & decoded, uint32_t rows,
                                     uint32_t columns, float minimum, float range,
                                     const std::vector<float> & input_window,
                                     const std::vector<float> & output_window,
                                     uint32_t samples) {
    std::vector<float> error(static_cast<size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row) for (uint32_t column = 0; column < columns; ++column) {
        error[static_cast<size_t>(row) * columns + column] =
            (normalized_weight(matrix, row, column, minimum, range) -
             decoded[static_cast<size_t>(row) * columns + column]) * range;
    }
    return astc_vulkan_yaqa_trace_score(error, rows, columns, input_window, output_window, samples);
}

bool matches_yaqa_oracle(const ggml_vk_astc_loaded_matrix & matrix,
                         const std::vector<float> & decoded, uint32_t rows,
                         uint32_t columns, float minimum, float range,
                         const std::vector<float> & input_window,
                         const std::vector<float> & output_window,
                         uint32_t samples, const std::vector<double> & residual) {
    double incremental_score = 0.0;
    for (const double value : residual) incremental_score += value * value;
    const double oracle_score = yaqa_trace_score_from_decoded(matrix, decoded, rows, columns, minimum, range,
                                                               input_window, output_window, samples);
    return std::isfinite(oracle_score) && std::fabs(incremental_score - oracle_score) <=
        1e-8 * std::max(1.0, std::fabs(oracle_score));
}

std::vector<double> yaqa_delta_for_block(const decoded_block & candidate,
                                          const decoded_block & baseline,
                                          uint32_t block_row, uint32_t block_column,
                                          uint32_t rows, uint32_t columns, float range,
                                          const std::vector<float> & input_window,
                                          const std::vector<float> & output_window,
                                          uint32_t samples) {
    std::vector<double> projected(static_cast<size_t>(kLogicalBlockHeight) * samples, 0.0);
    for (uint32_t local_row = 0; local_row < kLogicalBlockHeight; ++local_row) {
        const uint32_t row = block_row * kLogicalBlockHeight + local_row;
        if (row >= rows) continue;
        for (uint32_t sample = 0; sample < samples; ++sample) {
            double value = 0.0;
            for (uint32_t x = 0; x < kBlockWidth; ++x) {
                const uint32_t column = block_column * kBlockWidth + x;
                if (column >= columns) continue;
                const float decoded_delta = (candidate.logical_weights[local_row * kBlockWidth + x] -
                    baseline.logical_weights[local_row * kBlockWidth + x]) * range;
                value += decoded_delta * input_window[static_cast<size_t>(sample) * columns + column];
            }
            projected[static_cast<size_t>(local_row) * samples + sample] = value;
        }
    }
    std::vector<double> result(static_cast<size_t>(samples) * samples, 0.0);
    for (uint32_t output_sample = 0; output_sample < samples; ++output_sample) {
        for (uint32_t input_sample = 0; input_sample < samples; ++input_sample) {
            double value = 0.0;
            for (uint32_t local_row = 0; local_row < kLogicalBlockHeight; ++local_row) {
                const uint32_t row = block_row * kLogicalBlockHeight + local_row;
                if (row >= rows) continue;
                value += output_window[static_cast<size_t>(output_sample) * rows + row] *
                    projected[static_cast<size_t>(local_row) * samples + input_sample];
            }
            result[static_cast<size_t>(output_sample) * samples + input_sample] = value;
        }
    }
    return result;
}

double squared_norm(const std::vector<double> & values) {
    double result = 0.0;
    for (const double value : values) result += value * value;
    return result;
}

double residual_gain(const std::vector<double> & residual, const std::vector<double> & delta) {
    double result = 0.0;
    for (size_t index = 0; index < residual.size(); ++index) {
        result += 2.0 * residual[index] * delta[index] - delta[index] * delta[index];
    }
    return result;
}

void subtract_delta(std::vector<double> & residual, const std::vector<double> & delta) {
    for (size_t index = 0; index < residual.size(); ++index) residual[index] -= delta[index];
}

bool select_yaqa_candidates(const std::vector<double> & initial_calibration,
                            const std::vector<double> & initial_validation,
                            const std::vector<std::vector<yaqa_candidate_delta>> & candidates,
                            astc_vulkan_paired_selection_result & result) {
    if (initial_calibration.empty() || initial_validation.empty() || candidates.empty()) return false;
    const size_t calibration_size = initial_calibration.size();
    const size_t validation_size = initial_validation.size();
    for (const auto & block : candidates) {
        if (block.empty() || block.front().calibration.size() != calibration_size ||
            block.front().validation.size() != validation_size ||
            std::any_of(block.front().calibration.begin(), block.front().calibration.end(),
                        [](double value) { return value != 0.0; }) ||
            std::any_of(block.front().validation.begin(), block.front().validation.end(),
                        [](double value) { return value != 0.0; })) return false;
        for (const auto & candidate : block) {
            if (candidate.calibration.size() != calibration_size ||
                candidate.validation.size() != validation_size) return false;
        }
    }
    result = {};
    result.calibration_selected_candidates.assign(candidates.size(), 0);
    std::vector<double> calibration_residual = initial_calibration;
    std::vector<bool> committed(candidates.size(), false);
    while (true) {
        uint32_t best_block = 0, best_candidate = 0;
        double best_gain = 0.0;
        bool found = false;
        for (uint32_t block = 0; block < candidates.size(); ++block) {
            if (committed[block]) continue;
            for (uint32_t candidate = 1; candidate < candidates[block].size(); ++candidate) {
                const double gain = residual_gain(calibration_residual, candidates[block][candidate].calibration);
                if (gain > best_gain || (gain == best_gain && found &&
                    (block < best_block || (block == best_block && candidate < best_candidate)))) {
                    best_block = block;
                    best_candidate = candidate;
                    best_gain = gain;
                    found = true;
                }
            }
        }
        if (!found || !(best_gain > 0.0)) break;
        subtract_delta(calibration_residual, candidates[best_block][best_candidate].calibration);
        committed[best_block] = true;
        result.calibration_selected_candidates[best_block] = best_candidate;
        result.commits.push_back({best_block, best_candidate, best_gain});
    }
    result.calibration_residual_loss = squared_norm(calibration_residual);
    std::vector<double> validation_residual = initial_validation;
    result.validation_selected_candidates.assign(candidates.size(), 0);
    result.validation_residual_loss = squared_norm(validation_residual);
    for (uint32_t index = 0; index < result.commits.size(); ++index) {
        const auto & commit = result.commits[index];
        subtract_delta(validation_residual, candidates[commit.block][commit.candidate].validation);
        const double loss = squared_norm(validation_residual);
        if (loss < result.validation_residual_loss) {
            result.validation_residual_loss = loss;
            result.validation_prefix = index + 1;
        }
    }
    for (uint32_t index = 0; index < result.validation_prefix; ++index) {
        const auto & commit = result.commits[index];
        result.validation_selected_candidates[commit.block] = commit.candidate;
    }
    return true;
}

// // select one strip at a time, keeping the candidate working set proportional
// to the strip width instead of the full tensor.  Validation stopping is
// applied independently per strip; this is the exact separable objective and
// is deliberately reported as such in the artifact metadata.
bool run_row_strip_chunked(const params & options,
                           const ggml_vk_astc_loaded_matrix & matrix,
                           const ggml_vk_astc_activation_trace & trace,
                           float minimum, float range, bool neural_backend) {
    const uint32_t blocks_x = (options.columns + kBlockWidth - 1) / kBlockWidth;
    const uint32_t blocks_y = (options.rows + kLogicalBlockHeight - 1) / kLogicalBlockHeight;
    const uint32_t validation_offset = options.calibration_samples;
    const uint32_t holdout_offset = validation_offset + options.validation_samples;
    const uint32_t holdout_samples = trace.samples - holdout_offset;
    const auto codebook = make_source_candidates(options.source_derived_alpha);
    std::vector<float> neutral(static_cast<size_t>(options.rows) * options.columns, 0.0f);
    std::vector<float> selected(neutral.size(), 0.0f);
    const size_t block_count = static_cast<size_t>(blocks_x) * blocks_y;
    std::vector<uint8_t> selected_payload(block_count * 16u, 0);
    std::vector<uint32_t> selected_layout(static_cast<size_t>(astc_vulkan_paired_layout_word_count(
        kFootprint, options.columns, options.rows)), 0);
    uint64_t unique_candidates = 0, raw_candidates = 0;
    uint64_t accepted = 0, peak_candidates = 0;
    uint64_t selected_dual_planes = 0, selected_semantic_dual_planes = 0, selected_alpha_dual_planes = 0;
    constexpr uint32_t worker_count = 4;
    std::vector<std::unique_ptr<block_codec>> worker_rg_b;
    std::vector<std::unique_ptr<block_codec>> worker_r_gb;
    for (uint32_t worker = 0; worker < worker_count; ++worker) {
        worker_rg_b.emplace_back(std::make_unique<block_codec>(neural_backend, astc_vulkan_paired_layout::rg_b,
            options.channel_weights));
        worker_r_gb.emplace_back(std::make_unique<block_codec>(neural_backend, astc_vulkan_paired_layout::r_gb,
            options.channel_weights));
    }
    for (const auto & codec : worker_rg_b) if (!codec->ready()) return false;
    for (const auto & codec : worker_r_gb) if (!codec->ready()) return false;
    const auto start = std::chrono::steady_clock::now();

    for (uint32_t strip = 0; strip < blocks_y; ++strip) {
        const uint32_t row0 = strip * kLogicalBlockHeight;
        const uint32_t strip_rows = std::min(kLogicalBlockHeight, options.rows - row0);
        std::vector<std::vector<generated_candidate>> generated(blocks_x);
        std::vector<float> strip_neutral(static_cast<size_t>(strip_rows) * options.columns, 0.0f);
        std::atomic<uint32_t> next_block{0};
        std::atomic<bool> generation_failed{false};
        std::atomic<uint64_t> strip_raw{0}, strip_unique{0};
        auto worker = [&](uint32_t worker_index) {
            std::vector<float> source_scratch(kBlockWidth * kPhysicalBlockHeight * 4);
            for (;;) {
                const uint32_t block_x = next_block.fetch_add(1, std::memory_order_relaxed);
                if (block_x >= blocks_x || generation_failed.load(std::memory_order_relaxed)) return;
                auto & candidates = generated[block_x];
                for (const auto layout : {astc_vulkan_paired_layout::rg_b, astc_vulkan_paired_layout::r_gb}) {
                    block_codec & codec = layout == astc_vulkan_paired_layout::rg_b ? *worker_rg_b[worker_index] : *worker_r_gb[worker_index];
                    for (const auto basis : selected_paired_bases(options.paired_basis)) {
                        codec.begin_block();
                        for (const auto & steering : codebook) {
                            generated_candidate candidate;
                            candidate.layout = layout;
                            candidate.steering = steering.steering;
                            candidate.alpha_source = steering.alpha_source;
                            candidate.basis = basis;
                            fill_source_block(source_scratch, matrix, row0, block_x * kBlockWidth,
                                options.rows, options.columns, minimum, range, layout, basis, steering);
                            if (!codec.roundtrip(source_scratch, candidate.block, basis)) { generation_failed.store(true); return; }
                            candidate.delta.payload = candidate.block.payload;
                            ++strip_raw;
                            bool duplicate = false;
                            for (const auto & existing : candidates) duplicate = duplicate || same_candidate(existing, candidate);
                            if (!duplicate) candidates.push_back(std::move(candidate));
                        }
                    }
                }
                auto neutral_it = std::find_if(candidates.begin(), candidates.end(), [](const generated_candidate & candidate) {
                    return candidate.layout == astc_vulkan_paired_layout::rg_b &&
                           candidate.basis == astc_vulkan_paired_basis::direct &&
                           candidate.alpha_source == d2_alpha_source_kind::geometric &&
                           candidate.steering.basis == astc_vulkan_paired_steering_basis::neutral &&
                           candidate.steering.amplitude == 0.0f;
                });
                if (neutral_it == candidates.end()) { generation_failed.store(true); return; }
                std::iter_swap(candidates.begin(), neutral_it);
                const decoded_block & block = candidates.front().block;
                for (uint32_t local_row = 0; local_row < strip_rows; ++local_row) for (uint32_t x = 0; x < kBlockWidth; ++x) {
                    const uint32_t column = block_x * kBlockWidth + x;
                    if (column < options.columns) strip_neutral[static_cast<size_t>(local_row) * options.columns + column] =
                        block.logical_weights[local_row * kBlockWidth + x];
                }
                strip_unique += candidates.size();
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (uint32_t worker_index = 0; worker_index < worker_count; ++worker_index) workers.emplace_back(worker, worker_index);
        for (auto & thread : workers) thread.join();
        if (generation_failed.load(std::memory_order_relaxed)) return false;
        raw_candidates += strip_raw.load();
        unique_candidates += strip_unique.load();
        peak_candidates = std::max<uint64_t>(peak_candidates, strip_unique.load());
        for (uint32_t local_row = 0; local_row < strip_rows; ++local_row) {
            std::copy_n(strip_neutral.data() + static_cast<size_t>(local_row) * options.columns,
                        options.columns, neutral.data() + static_cast<size_t>(row0 + local_row) * options.columns);
        }

        auto make_delta = [&](const generated_candidate & candidate, uint32_t sample_offset, uint32_t samples) {
            std::vector<double> delta(static_cast<size_t>(samples) * strip_rows, 0.0);
            const decoded_block & base = generated[0].front().block; // overwritten below per block
            (void) base;
            return delta;
        };
        std::vector<double> initial_cal(static_cast<size_t>(options.calibration_samples) * strip_rows, 0.0);
        std::vector<double> initial_val(static_cast<size_t>(options.validation_samples) * strip_rows, 0.0);
        for (uint32_t sample = 0; sample < options.calibration_samples + options.validation_samples; ++sample) {
            const uint32_t sample_offset = sample < options.calibration_samples ? 0 : validation_offset;
            const uint32_t local_sample = sample < options.calibration_samples ? sample : sample - options.calibration_samples;
            auto & output = sample < options.calibration_samples ? initial_cal : initial_val;
            for (uint32_t local_row = 0; local_row < strip_rows; ++local_row) {
                double value = 0.0;
                for (uint32_t column = 0; column < options.columns; ++column) {
                    const float source = normalized_weight(matrix, row0 + local_row, column, minimum, range);
                    const float decoded = strip_neutral[static_cast<size_t>(local_row) * options.columns + column];
                    value += (source - decoded) * range * trace.values[static_cast<size_t>(sample_offset + local_sample) * trace.columns + column];
                }
                output[static_cast<size_t>(local_sample) * strip_rows + local_row] = value;
            }
        }

        std::vector<std::vector<astc_vulkan_paired_candidate_delta>> selector_candidates(blocks_x);
        for (uint32_t block_x = 0; block_x < blocks_x; ++block_x) {
            const auto & base = generated[block_x].front().block;
            for (auto & candidate : generated[block_x]) {
                candidate.delta.calibration_delta.assign(static_cast<size_t>(options.calibration_samples) * strip_rows, 0.0);
                candidate.delta.validation_delta.assign(static_cast<size_t>(options.validation_samples) * strip_rows, 0.0);
                for (uint32_t sample = 0; sample < options.calibration_samples + options.validation_samples; ++sample) {
                    const uint32_t sample_offset = sample < options.calibration_samples ? 0 : validation_offset;
                    const uint32_t local_sample = sample < options.calibration_samples ? sample : sample - options.calibration_samples;
                    auto & output = sample < options.calibration_samples ? candidate.delta.calibration_delta : candidate.delta.validation_delta;
                    for (uint32_t local_row = 0; local_row < strip_rows; ++local_row) {
                        double value = 0.0;
                        for (uint32_t x = 0; x < kBlockWidth; ++x) {
                            const uint32_t column = block_x * kBlockWidth + x;
                            if (column >= options.columns) continue;
                            value += (candidate.block.logical_weights[local_row * kBlockWidth + x] -
                                      base.logical_weights[local_row * kBlockWidth + x]) * range *
                                     trace.values[static_cast<size_t>(sample_offset + local_sample) * trace.columns + column];
                        }
                        output[static_cast<size_t>(local_sample) * strip_rows + local_row] = value;
                    }
                }
                selector_candidates[block_x].push_back(candidate.delta);
            }
        }
        astc_vulkan_paired_selection_result selection;
        const astc_vulkan_paired_selector_config config{strip_rows, options.calibration_samples, options.validation_samples};
        if (!astc_vulkan_select_paired_candidates(config, initial_cal, initial_val, selector_candidates, selection)) return false;
        accepted += selection.commits.size();
        for (uint32_t block_x = 0; block_x < blocks_x; ++block_x) {
            const auto & base = generated[block_x].front().block;
            const uint32_t chosen = selection.validation_selected_candidates[block_x];
            const auto & chosen_candidate = generated[block_x][chosen];
            const auto & block = chosen_candidate.block;
            if (block.info_valid && block.info.is_dual_plane_block) {
                ++selected_dual_planes;
                const unsigned int semantic_singleton = chosen_candidate.layout == astc_vulkan_paired_layout::rg_b ? 2u : 0u;
                if (block.info.dual_plane_component == semantic_singleton) ++selected_semantic_dual_planes;
                if (block.info.dual_plane_component == 3u) ++selected_alpha_dual_planes;
            }
            for (uint32_t local_row = 0; local_row < strip_rows; ++local_row) for (uint32_t x = 0; x < kBlockWidth; ++x) {
                const uint32_t column = block_x * kBlockWidth + x;
                if (column >= options.columns) continue;
                neutral[static_cast<size_t>(row0 + local_row) * options.columns + column] = base.logical_weights[local_row * kBlockWidth + x];
                selected[static_cast<size_t>(row0 + local_row) * options.columns + column] = block.logical_weights[local_row * kBlockWidth + x];
            }
            const size_t block_index = static_cast<size_t>(strip) * blocks_x + block_x;
            std::copy(block.payload.begin(), block.payload.end(), selected_payload.begin() + block_index * 16u);
            if (!astc_vulkan_paired_layout_set(selected_layout, block_index, generated[block_x][chosen].layout)) return false;
        }
        if ((strip + 1) % 8 == 0 || strip + 1 == blocks_y) {
            std::printf("paired-select chunked progress strips=%u/%u accepted=%llu elapsed=%.1fs\n", strip + 1, blocks_y,
                        static_cast<unsigned long long>(accepted), std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            std::fflush(stdout);
        }
    }

    const auto hold_error = output_error(matrix, trace, selected, options.rows, options.columns, holdout_offset, holdout_samples, minimum, range);
    const auto neutral_hold = output_error(matrix, trace, neutral, options.rows, options.columns, holdout_offset, holdout_samples, minimum, range);
    const double selected_holdout = mse(hold_error), neutral_holdout = mse(neutral_hold);
    if ((!options.export_payload.empty()) != (!options.export_layout.empty())) return false;
    if (!options.export_payload.empty() && options.paired_basis != d2_basis_selection::direct) {
        std::fprintf(stderr, "paired selection cannot export common/difference without a versioned basis map\n");
        return false;
    }
    if (!options.export_payload.empty()) {
        std::ofstream payload(options.export_payload, std::ios::binary | std::ios::trunc);
        std::ofstream layout(options.export_layout, std::ios::binary | std::ios::trunc);
        if (!payload || !layout) return false;
        payload.write(reinterpret_cast<const char *>(selected_payload.data()), selected_payload.size());
        layout.write(reinterpret_cast<const char *>(selected_layout.data()), static_cast<std::streamsize>(selected_layout.size() * sizeof(uint32_t)));
        if (!payload.good() || !layout.good()) return false;
    }
    std::printf("paired-select chunked D2_%s rows=%u columns=%u channel-weights=%s source-alpha=%s basis=%s blocks=%zu raw=%llu unique=%llu peak-candidates=%llu accepted=%llu dual-plane=%llu semantic-plane=%llu alpha-plane=%llu neutral-holdout=%.8g selected-holdout=%.8g\n",
                kFootprintName, options.rows, options.columns,
                channel_weight_profile_name(options.channel_weights),
                options.source_derived_alpha ? "derived-replace-diagonals" : "geometric-v1",
                paired_basis_selection_name(options.paired_basis), block_count,
                static_cast<unsigned long long>(raw_candidates), static_cast<unsigned long long>(unique_candidates),
                static_cast<unsigned long long>(peak_candidates), static_cast<unsigned long long>(accepted),
                static_cast<unsigned long long>(selected_dual_planes),
                static_cast<unsigned long long>(selected_semantic_dual_planes),
                static_cast<unsigned long long>(selected_alpha_dual_planes), neutral_holdout, selected_holdout);
    if (!options.report.empty()) {
        std::ofstream report(options.report);
        if (!report) return false;
        report << "backend=" << (neural_backend ? "neural-d2" : "standard") << '\n'
               << "footprint=" << kFootprintName << '\n'
               << "rows=" << options.rows << '\n'
               << "columns=" << options.columns << '\n'
               << "channel_weights=" << channel_weight_profile_name(options.channel_weights) << '\n'
               << "source_alpha=" << (options.source_derived_alpha ? "derived-replace-diagonals" : "geometric-v1") << '\n'
               << "paired_basis=" << paired_basis_selection_name(options.paired_basis) << '\n'
               << "blocks=" << block_count << '\n'
               << "selection_scope=row-strip-independent\n"
               << "raw_candidates=" << raw_candidates << '\n'
               << "unique_candidates=" << unique_candidates << '\n'
               << "peak_candidates=" << peak_candidates << '\n'
               << "accepted=" << accepted << '\n'
               << "selected_dual_planes=" << selected_dual_planes << '\n'
               << "selected_semantic_dual_planes=" << selected_semantic_dual_planes << '\n'
               << "selected_alpha_dual_planes=" << selected_alpha_dual_planes << '\n'
               << "neutral_holdout_mse=" << neutral_holdout << '\n'


               << "selected_holdout_mse=" << selected_holdout << '\n';
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    params options;
    if (!parse_params(argc, argv, options)) {
        std::fprintf(stderr, "usage: %s --model model.gguf --tensor name --trace input.trace "
                             "[--rows N --columns N --calibration-samples N --validation-samples N "
                             "--progress-every-blocks N --report path [--structure-bank 1] "
                             "[--pv-alternate 1] [--row-strip-chunked 1] "
                             "[--channel-weights legacy|balanced-a025|balanced-a050] "
                             "[--source-derived-alpha 1] "
                             "[--paired-basis direct|common-difference] "
                             "[--objective activation|yaqa --output-trace output.trace] "
                             "[--export-payload path --export-layout path]\n", argv[0]);
        return 2;
    }
    ggml_vk_astc_loaded_matrix matrix;
    ggml_vk_astc_activation_trace trace, output_trace;
    std::string error;
    if (!ggml_vk_astc_load_gguf_matrix(options.model, options.tensor, matrix, error) ||
        !ggml_vk_astc_load_activation_trace(options.trace, trace, error) ||
        options.rows > matrix.rows || options.columns > matrix.columns || options.columns > trace.columns ||
        options.calibration_samples + options.validation_samples >= trace.samples) {
        std::fprintf(stderr, "paired selection smoke input error: %s\n", error.c_str());
        return 1;
    }
    if (options.objective == astc_vulkan_objective::two_sided_trace &&
        (!ggml_vk_astc_load_activation_trace(options.output_trace, output_trace, error) ||
         output_trace.samples != trace.samples || output_trace.columns < options.rows)) {
        std::fprintf(stderr, "paired selection YAQA output trace error: %s\n", error.c_str());
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
    if (options.row_strip_chunked && options.objective != astc_vulkan_objective::activation) {
        std::fprintf(stderr, "paired selection YAQA requires global selection; disable --row-strip-chunked\n");
        return 1;
    }
    if (options.row_strip_chunked) {
        return run_row_strip_chunked(options, matrix, trace, minimum, range, neural_backend) ? 0 : 1;
    }
    std::vector<std::vector<generated_candidate>> generated(static_cast<size_t>(blocks_x) * blocks_y);
    std::vector<float> baseline(static_cast<size_t>(options.rows) * options.columns);
    std::vector<float> source_scratch(kBlockWidth * kPhysicalBlockHeight * 4);
    const auto codebook = make_source_candidates(options.source_derived_alpha);
    block_codec rg_b_codec(neural_backend, astc_vulkan_paired_layout::rg_b, options.channel_weights,
                           nullptr, options.structure_bank);
    block_codec r_gb_codec(neural_backend, astc_vulkan_paired_layout::r_gb, options.channel_weights,
                           &rg_b_codec, options.structure_bank);
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
                for (const auto basis : selected_paired_bases(options.paired_basis)) {
                    codec.begin_block();
                    for (const auto & steering : codebook) {
                        generated_candidate candidate;
                        candidate.layout = layout;
                        candidate.steering = steering.steering;
                        candidate.alpha_source = steering.alpha_source;
                        candidate.basis = basis;
                        const auto source_start = std::chrono::steady_clock::now();
                        fill_source_block(source_scratch, matrix, block_y * kLogicalBlockHeight, block_x * kBlockWidth,
                            options.rows, options.columns, minimum, range, layout, basis, steering);
                        profile.source_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - source_start).count();
                        if (!codec.roundtrip(source_scratch, candidate.block, basis)) return 1;
                        if (basis == astc_vulkan_paired_basis::direct &&
                            steering.steering.basis == astc_vulkan_paired_steering_basis::neutral &&
                            steering.steering.amplitude == 0.0f &&
                            steering.alpha_source == d2_alpha_source_kind::geometric) {
                            codec.commit_structure_bank();
                        }
                        candidate.delta.payload = candidate.block.payload;
                        ++raw_candidates;
                        bool duplicate = false;
                        for (const auto & existing : candidates) duplicate = duplicate || same_candidate(existing, candidate);
                        if (!duplicate) candidates.push_back(std::move(candidate));
                    }
                }
                if (options.pv_alternate && options.paired_basis == d2_basis_selection::direct) {
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
                                minimum, range, layout, astc_vulkan_paired_basis::direct,
                                [&coefficients](float, float, float x, float y) {
                                    return 0.5f + coefficients[0] * x + coefficients[1] * y;
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
                                    minimum, range, layout, astc_vulkan_paired_basis::direct,
                                    [&pv_result](float, float, float x, float y) {
                                        return 0.5f + pv_result.continuous[0] * x +
                                            pv_result.continuous[1] * y;
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
                       candidate.basis == astc_vulkan_paired_basis::direct &&
                       candidate.alpha_source == d2_alpha_source_kind::geometric &&
                       candidate.steering.basis == astc_vulkan_paired_steering_basis::neutral &&
                       candidate.steering.amplitude == 0.0f;
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
    astc_vulkan_paired_selection_result selection;
    const auto selection_start = std::chrono::steady_clock::now();
    if (options.objective == astc_vulkan_objective::activation) {
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
        const astc_vulkan_paired_selector_config config{options.rows, options.calibration_samples, options.validation_samples};
        if (!astc_vulkan_select_paired_candidates(config, initial_calibration, initial_validation,
                                                   selector_candidates, selection)) return 1;
    } else {
        std::vector<float> calibration_input, calibration_output, validation_input, validation_output;
        if (!make_yaqa_trace_window(trace, output_trace, 0, options.calibration_samples, options.rows, options.columns,
                                    calibration_input, calibration_output) ||
            !make_yaqa_trace_window(trace, output_trace, validation_offset, options.validation_samples,
                                    options.rows, options.columns, validation_input, validation_output)) return 1;
        const auto yaqa_calibration = yaqa_residual(matrix, baseline, options.rows, options.columns, minimum, range,
                                                    calibration_input, calibration_output, options.calibration_samples);
        const auto yaqa_validation = yaqa_residual(matrix, baseline, options.rows, options.columns, minimum, range,
                                                   validation_input, validation_output, options.validation_samples);
        if (!matches_yaqa_oracle(matrix, baseline, options.rows, options.columns, minimum, range,
                                 calibration_input, calibration_output, options.calibration_samples, yaqa_calibration) ||
            !matches_yaqa_oracle(matrix, baseline, options.rows, options.columns, minimum, range,
                                 validation_input, validation_output, options.validation_samples, yaqa_validation)) return 1;
        std::vector<std::vector<yaqa_candidate_delta>> selector_candidates(generated.size());
        for (uint32_t block_y = 0; block_y < blocks_y; ++block_y) for (uint32_t block_x = 0; block_x < blocks_x; ++block_x) {
            auto & candidates = generated[static_cast<size_t>(block_y) * blocks_x + block_x];
            const decoded_block & neutral = candidates.front().block;
            auto & deltas = selector_candidates[static_cast<size_t>(block_y) * blocks_x + block_x];
            deltas.reserve(candidates.size());
            for (const auto & candidate : candidates) {
                yaqa_candidate_delta delta;
                delta.calibration = yaqa_delta_for_block(candidate.block, neutral, block_y, block_x,
                    options.rows, options.columns, range, calibration_input, calibration_output,
                    options.calibration_samples);
                delta.validation = yaqa_delta_for_block(candidate.block, neutral, block_y, block_x,
                    options.rows, options.columns, range, validation_input, validation_output,
                    options.validation_samples);
                deltas.push_back(std::move(delta));
            }
        }
        if (!select_yaqa_candidates(yaqa_calibration, yaqa_validation, selector_candidates, selection)) return 1;
    }
    profile.selection_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - selection_start).count();

    std::vector<float> selected = baseline;
    uint64_t selected_dual_planes = 0, selected_semantic_dual_planes = 0, selected_alpha_dual_planes = 0;
    for (uint32_t block_y = 0; block_y < blocks_y; ++block_y) for (uint32_t block_x = 0; block_x < blocks_x; ++block_x) {
        const size_t block = static_cast<size_t>(block_y) * blocks_x + block_x;
        const auto & chosen_candidate = generated[block][selection.validation_selected_candidates[block]];
        if (chosen_candidate.block.info_valid && chosen_candidate.block.info.is_dual_plane_block) {
            ++selected_dual_planes;
            const unsigned int semantic_singleton = chosen_candidate.layout == astc_vulkan_paired_layout::rg_b ? 2u : 0u;
            if (chosen_candidate.block.info.dual_plane_component == semantic_singleton) ++selected_semantic_dual_planes;
            if (chosen_candidate.block.info.dual_plane_component == 3u) ++selected_alpha_dual_planes;
        }
        write_block(selected, chosen_candidate.block,
                    block_y, block_x, options.rows, options.columns);
    }
    const auto final_objective_start = std::chrono::steady_clock::now();
    const auto baseline_holdout = output_error(matrix, trace, baseline, options.rows, options.columns,
        holdout_offset, holdout_samples, minimum, range);
    const auto selected_holdout = output_error(matrix, trace, selected, options.rows, options.columns,
        holdout_offset, holdout_samples, minimum, range);
    double neutral_holdout_objective = mse(baseline_holdout);
    double selected_holdout_objective = mse(selected_holdout);
    if (options.objective == astc_vulkan_objective::two_sided_trace) {
        std::vector<float> holdout_input, holdout_output;
        if (!make_yaqa_trace_window(trace, output_trace, holdout_offset, holdout_samples,
                                    options.rows, options.columns, holdout_input, holdout_output)) return 1;
        neutral_holdout_objective = squared_norm(yaqa_residual(matrix, baseline, options.rows, options.columns,
            minimum, range, holdout_input, holdout_output, holdout_samples));
        selected_holdout_objective = squared_norm(yaqa_residual(matrix, selected, options.rows, options.columns,
            minimum, range, holdout_input, holdout_output, holdout_samples));
    }
    profile.objective_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - final_objective_start).count();
    uint64_t unique_candidates = 0;
    for (const auto & candidates : generated) unique_candidates += candidates.size();
    const double neutral_calibration_mse = mse(initial_calibration);
    const double neutral_validation_mse = mse(initial_validation);
    const double neutral_holdout_mse = mse(baseline_holdout);
    const double selected_calibration_mse = options.objective == astc_vulkan_objective::activation ?
        selection.calibration_residual_loss / (options.calibration_samples * options.rows) : selection.calibration_residual_loss;
    const double selected_validation_mse = options.objective == astc_vulkan_objective::activation ?
        selection.validation_residual_loss / (options.validation_samples * options.rows) : selection.validation_residual_loss;
    const double selected_holdout_mse = selected_holdout_objective;
    if ((!options.export_payload.empty()) != (!options.export_layout.empty())) return 1;
    if (!options.export_payload.empty() && options.paired_basis != d2_basis_selection::direct) {
        std::fprintf(stderr, "paired selection cannot export common/difference without a versioned basis map\n");
        return 1;
    }
    if (!options.export_payload.empty()) {
        std::vector<uint8_t> payload(generated.size() * 16u);
        std::vector<uint32_t> layout_words(static_cast<size_t>(astc_vulkan_paired_layout_word_count(
            kFootprint, options.columns, options.rows)), 0);
        for (size_t block = 0; block < generated.size(); ++block) {
            const auto & selected_candidate = generated[block][selection.validation_selected_candidates[block]];
            std::copy(selected_candidate.block.payload.begin(), selected_candidate.block.payload.end(),
                      payload.begin() + block * 16u);
            if (!astc_vulkan_paired_layout_set(layout_words, block, selected_candidate.layout)) return 1;
        }
        std::ofstream payload_file(options.export_payload, std::ios::binary | std::ios::trunc);
        std::ofstream layout_file(options.export_layout, std::ios::binary | std::ios::trunc);
        if (!payload_file || !layout_file) return 1;
        payload_file.write(reinterpret_cast<const char *>(payload.data()), payload.size());
        layout_file.write(reinterpret_cast<const char *>(layout_words.data()),
                          static_cast<std::streamsize>(layout_words.size() * sizeof(uint32_t)));
        if (!payload_file.good() || !layout_file.good()) return 1;
    }
    std::printf("paired-select D2_%s rows=%u columns=%u objective=%s channel-weights=%s source-alpha=%s basis=%s blocks=%u raw=%llu unique=%llu dual-plane=%llu semantic-plane=%llu alpha-plane=%llu\n",
                kFootprintName,
                options.rows, options.columns, astc_vulkan_objective_name(options.objective), channel_weight_profile_name(options.channel_weights),
                options.source_derived_alpha ? "derived-replace-diagonals" : "geometric-v1",
                paired_basis_selection_name(options.paired_basis), blocks_x * blocks_y,
                static_cast<unsigned long long>(raw_candidates), static_cast<unsigned long long>(unique_candidates),
                static_cast<unsigned long long>(selected_dual_planes),
                static_cast<unsigned long long>(selected_semantic_dual_planes),
                static_cast<unsigned long long>(selected_alpha_dual_planes));
    std::printf("paired-select neutral activation calibration-mse=%.8g validation-mse=%.8g holdout-mse=%.8g\n",
                neutral_calibration_mse, neutral_validation_mse, neutral_holdout_mse);
    std::printf("paired-select selected objective=%s commits=%zu validation-prefix=%u calibration-score=%.8g validation-score=%.8g holdout-score=%.8g\n",
                astc_vulkan_objective_name(options.objective),
                selection.commits.size(), selection.validation_prefix,
                selected_calibration_mse, selected_validation_mse, selected_holdout_mse);
    std::printf("paired-select objective holdout neutral-score=%.8g selected-score=%.8g\n",
                neutral_holdout_objective, selected_holdout_objective);
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
               << "channel_weights=" << channel_weight_profile_name(options.channel_weights) << '\n'
               << "source_alpha=" << (options.source_derived_alpha ? "derived-replace-diagonals" : "geometric-v1") << '\n'
               << "paired_basis=" << paired_basis_selection_name(options.paired_basis) << '\n'
               << "objective=" << astc_vulkan_objective_name(options.objective) << '\n'
               << "output_trace=" << (options.output_trace.empty() ? "" : options.output_trace) << '\n'
               << "blocks=" << blocks_x * blocks_y << '\n'
               << "raw_candidates=" << raw_candidates << '\n'
               << "unique_candidates=" << unique_candidates << '\n'
               << "neutral_calibration_mse=" << neutral_calibration_mse << '\n'
               << "neutral_validation_mse=" << neutral_validation_mse << '\n'
               << "neutral_holdout_mse=" << neutral_holdout_mse << '\n'
               << "commits=" << selection.commits.size() << '\n'
               << "validation_prefix=" << selection.validation_prefix << '\n'
               << "selected_calibration_objective=" << selected_calibration_mse << '\n'
               << "selected_validation_objective=" << selected_validation_mse << '\n'
               << "selected_holdout_objective=" << selected_holdout_mse << '\n'
               << "selected_holdout_activation_mse=" << mse(selected_holdout) << '\n'
               << "neutral_holdout_objective=" << neutral_holdout_objective << '\n'
               << "selected_dual_planes=" << selected_dual_planes << '\n'
               << "selected_semantic_dual_planes=" << selected_semantic_dual_planes << '\n'
               << "selected_alpha_dual_planes=" << selected_alpha_dual_planes << '\n'
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
