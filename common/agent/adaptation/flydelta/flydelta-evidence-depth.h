#pragma once

#include "agent/adaptation/flydelta/flydelta-direction-search.h"

#include <cstddef>
#include <string>
#include <vector>

// Search depth is a bounded worker budget, not a second learning algorithm.
// The underlying direction, overlay and host-verification seams are shared by
// all three levels.
enum class common_flydelta_search_depth {
    bootstrap,
    shallow,
    deep,
};

const char * common_flydelta_search_depth_name(common_flydelta_search_depth depth);

struct common_flydelta_evidence_depth_config {
    int schema_version = 1;
    size_t min_shallow_samples = 2;
    size_t min_deep_samples = 6;
    size_t max_samples = 32;
    float rank_relative_tolerance = 0.10f;
    float min_median_alignment = 0.25f;
    float max_condition_number = 100.0f;
};

struct common_flydelta_evidence_depth_result {
    common_flydelta_search_depth depth = common_flydelta_search_depth::bootstrap;
    // Only HELPED samples with explicit learning eligibility contribute to
    // these evidence-depth statistics.
    size_t compatible_samples = 0;
    // Compatible non-HARMED observations retained for experimental search,
    // but excluded from evidence-depth rank and capacity.
    size_t experimental_samples = 0;
    size_t incompatible_samples = 0;
    size_t effective_rank = 0;
    float stable_rank = 0.0f;
    float median_alignment = 0.0f;
    float condition_number = 0.0f;
    bool basis_condition_ok = false;
    bool geometry_stable = false;
    bool shallow_ready = false;
    bool deep_ready = false;
};

// A budget profile keeps Bootstrap/Shallow/Deep as policies over the common
// search pipeline. It is intentionally descriptive; the model callback still
// owns execution and host verification.
struct common_flydelta_search_budget {
    common_flydelta_search_depth depth = common_flydelta_search_depth::bootstrap;
    size_t max_region_trials = 4;
    size_t max_coefficient_trials = 0;
    size_t full_generation_top_k = 1;
    bool build_aggregate_directions = false;
    bool require_decision_margin = false;
    bool allow_tfo_lite = false;
    bool include_opposite_control = false;
};

bool common_flydelta_search_budget_validate(
        const common_flydelta_search_budget & budget,
        std::string & error);

common_flydelta_search_budget common_flydelta_search_budget_for_depth(
        common_flydelta_search_depth depth);

bool common_flydelta_evidence_depth_config_validate(
        const common_flydelta_evidence_depth_config & config,
        std::string & error);

// Assesses only compatible, HELPED and learning-eligible contrast samples for
// one direction identity. UNKNOWN and NEUTRAL remain visible as experimental
// sample counts, but can never open Shallow/Deep by themselves. The rank is
// computed from the small sample Gram matrix, so the cost is bounded by sample
// count and does not require a dimension-sized covariance matrix.
bool common_flydelta_assess_evidence_depth(
        const common_flydelta_direction_search_config & identity,
        const common_flydelta_evidence_depth_config & config,
        const std::vector<common_flydelta_contrast_sample> & samples,
        common_flydelta_evidence_depth_result & result,
        std::string & error);
