#pragma once

#include "agent/adaptation/flydelta/flydelta-basis.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// CPU-only direction candidates built from already host-certified repair
// deltas. This component does not run inference and does not promote or
// activate a direction.
enum class common_flydelta_direction_kind {
    raw_repair,
    normalized_trimmed_mean,
    diagonal_whitened_mean,
    token_margin_direction,
    execution_boundary_prototype,
};

const char * common_flydelta_direction_kind_name(
        common_flydelta_direction_kind kind);

struct common_flydelta_contrast_sample {
    common_flydelta_behavior_delta delta;
    common_flydelta_intervention_credit credit;
};

// Generic forward-only decision material. The caller owns tokenization and
// unembedding; this contract only receives the two comparable output rows.
struct common_flydelta_token_margin_material {
    std::vector<float> positive_output_row;
    std::vector<float> negative_output_row;
};

// Generic host/model boundary material. A positive and negative capture may
// represent tool choice, planning, research, structured output or another
// host-verifiable behavior; no tool-specific semantics are stored here.
struct common_flydelta_boundary_sample {
    bool positive = false;
    std::vector<float> values;
};

struct common_flydelta_direction_search_config {
    int schema_version = 1;
    size_t dimension = 0;
    int32_t layer_index = -1;
    size_t min_samples = 2;
    size_t max_samples = 32;
    float min_median_alignment = 0.25f;
    float trim_fraction = 0.20f;
    float variance_ridge = 0.001f;
    common_adaptation_evidence_source source = common_adaptation_evidence_source::tool_repair;
    std::string behavior_key;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
};

struct common_flydelta_direction_candidate {
    int schema_version = 1;
    common_flydelta_direction_kind kind = common_flydelta_direction_kind::raw_repair;
    int32_t layer_index = -1;
    std::vector<float> values;
    size_t source_samples = 0;
    size_t retained_samples = 0;
    float median_alignment = 0.0f;
};

bool common_flydelta_direction_search_config_validate(
        const common_flydelta_direction_search_config & config,
        std::string & error);
bool common_flydelta_direction_candidate_validate(
        const common_flydelta_direction_candidate & candidate,
        size_t expected_dimension,
        std::string & error);

// Builds up to three CPU-cheap WHAT candidates:
//   raw_repair              one normalized sample, used as a control;
//   normalized_trimmed_mean robust mean of centrally aligned samples;
//   diagonal_whitened_mean  the same mean weighted by inverse per-dimension
//                           variance plus variance_ridge.
//
// Only HELPED, learning-eligible samples are accepted. The host must already
// have certified the repair relation; this function never infers correctness.
// If there are too few compatible samples, the raw control is still returned
// and the aggregate candidates are omitted.
bool common_flydelta_build_direction_candidates(
        const common_flydelta_direction_search_config & config,
        const std::vector<common_flydelta_contrast_sample> & samples,
        std::vector<common_flydelta_direction_candidate> & candidates,
        std::string & error);

// Builds one direct direction from a positive-vs-negative output margin.
// Output rows are supplied by the host/model adapter so this module remains
// independent of tokenizer and llama.cpp internals.
bool common_flydelta_build_token_margin_candidate(
        const common_flydelta_direction_search_config & config,
        const common_flydelta_token_margin_material & material,
        common_flydelta_direction_candidate & candidate,
        std::string & error);

// Builds one robust, normalized prototype direction from multiple positive
// and negative boundary captures. This is intentionally the same candidate
// format used by repair directions and is not restricted to tool use.
bool common_flydelta_build_boundary_prototype_candidate(
        const common_flydelta_direction_search_config & config,
        const std::vector<common_flydelta_boundary_sample> & samples,
        common_flydelta_direction_candidate & candidate,
        std::string & error);
