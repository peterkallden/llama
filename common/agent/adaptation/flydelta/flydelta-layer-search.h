#pragma once

#include "agent/adaptation/flydelta/flydelta-experiment.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Layer search is a host-side experiment planner. It ranks already captured
// UNKNOWN diagnostics and creates a bounded coarse-to-fine plan; it does not
// run inference, compose an overlay or create learning evidence.
struct common_flydelta_layer_diagnostic {
    uint32_t layer_index = 0;
    float cosine = 0.0f;
    float progress = 0.0f;
    float leakage = 0.0f;
    float shift_norm = 0.0f;
};

enum class common_flydelta_layer_search_candidate_source {
    diagnostic_singleton,
    neighborhood_expansion,
};

const char * common_flydelta_layer_search_candidate_source_name(
        common_flydelta_layer_search_candidate_source source);

struct common_flydelta_layer_search_config {
    int schema_version = 1;
    size_t max_regions = 2;
    size_t max_singletons = 4;
    size_t max_neighborhoods = 4;
    size_t max_candidates = 8;
    uint32_t min_region_separation = 2;
    float min_cosine = 0.3f;
    float total_scale = 0.01f;
};

struct common_flydelta_layer_candidate {
    int schema_version = 1;
    std::vector<uint32_t> layer_indices;
    uint32_t anchor_layer_index = 0;
    float diagnostic_score = 0.0f;
    float total_scale = 0.0f;
    float per_layer_scale = 0.0f;
    common_flydelta_layer_search_candidate_source source =
        common_flydelta_layer_search_candidate_source::diagnostic_singleton;
};

struct common_flydelta_layer_search_plan {
    int schema_version = 1;
    std::vector<common_flydelta_layer_candidate> singleton_candidates;
    std::vector<common_flydelta_layer_candidate> neighborhood_candidates;
};

bool common_flydelta_layer_search_config_validate(
        const common_flydelta_layer_search_config & config,
        std::string & error);
bool common_flydelta_layer_candidate_validate(
        const common_flydelta_layer_candidate & candidate,
        std::string & error);
bool common_flydelta_layer_search_plan_validate(
        const common_flydelta_layer_search_plan & plan,
        const common_flydelta_layer_search_config & config,
        std::string & error);

// Builds singleton candidates around at most max_regions separated local
// maxima. Neighborhood candidates are only adjacent pairs around those
// anchors and are intended to run after singleton verification. The supplied
// available layers must be the host-captured/basis-compatible layer set.
bool common_flydelta_build_layer_search_plan(
        const std::vector<common_flydelta_layer_diagnostic> & diagnostics,
        const std::vector<uint32_t> & available_layers,
        const common_flydelta_layer_search_config & config,
        common_flydelta_layer_search_plan & plan,
        std::string & error);

struct common_flydelta_layer_search_trial {
    common_flydelta_layer_candidate candidate;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
    float quality_delta = 0.0f;
    bool executed = false;
    bool verifier_known = false;
    std::string evidence_ref;
};

struct common_flydelta_layer_search_selection {
    bool selected = false;
    common_flydelta_layer_candidate candidate;
    float score = 0.0f;
    size_t trial_index = 0;
};

bool common_flydelta_layer_search_trial_validate(
        const common_flydelta_layer_search_trial & trial,
        std::string & error);

// Only a host-verified HELPED trial may be selected. UNKNOWN and NEUTRAL are
// diagnostic/retention results and can never become learning or promotion
// input through this helper.
bool common_flydelta_select_layer_candidate(
        const common_flydelta_layer_search_plan & plan,
        const std::vector<common_flydelta_layer_search_trial> & trials,
        common_flydelta_layer_search_selection & selection,
        std::string & error);
