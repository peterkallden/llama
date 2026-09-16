#pragma once

#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"
#include "agent/adaptation/flydelta/flydelta-layer-search.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// A bounded layer x scale scan. It is deliberately separate from the
// diagnostic layer planner: a weak probe must not exclude a layer before its
// first small scale ladder has been tried.
struct common_flydelta_intervention_region_search_config {
    int schema_version = 1;
    std::vector<uint32_t> available_layers;
    std::vector<float> scales;
    size_t max_singleton_layers = 4;
    size_t max_neighborhoods = 4;
    size_t max_trials = 32;
    size_t max_stalled_scales = 2;
    float min_cosine = 0.3f;
    float max_leakage = 1.0f;
    float max_shift_norm = 1.0f;
};

bool common_flydelta_intervention_region_search_config_validate(
        const common_flydelta_intervention_region_search_config & config,
        std::string & error);

struct common_flydelta_intervention_region_candidate {
    int schema_version = 1;
    std::vector<uint32_t> layer_indices;
    uint32_t anchor_layer_index = 0;
    float total_scale = 0.0f;
    float per_layer_scale = 0.0f;
    common_flydelta_layer_search_candidate_source source =
        common_flydelta_layer_search_candidate_source::diagnostic_singleton;
};

bool common_flydelta_intervention_region_candidate_validate(
        const common_flydelta_intervention_region_candidate & candidate,
        std::string & error);

struct common_flydelta_intervention_region_trial {
    common_flydelta_intervention_region_candidate candidate;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
    float quality_delta = 0.0f;
    common_flydelta_decision_margin margin;
    bool executed = false;
    bool verifier_known = false;
    bool geometry_available = false;
    common_flydelta_representation_diagnostics geometry;
    bool promising = false;
    bool safe_to_continue = false;
    std::string evidence_ref;
};

bool common_flydelta_intervention_region_trial_validate(
        const common_flydelta_intervention_region_trial & trial,
        std::string & error);

struct common_flydelta_intervention_region_selection {
    bool selected = false;
    size_t trial_index = 0;
    float score = 0.0f;
};

using common_flydelta_intervention_region_search_runner = std::function<bool(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_intervention_region_candidate * candidate,
        bool apply_overlay,
        common_flydelta_counterfactual_trial & trial,
        common_flydelta_decision_margin & margin,
        common_flydelta_representation_diagnostics & geometry,
        bool & geometry_available,
        std::string & error)>;

// Runs baseline, singleton layer/scale arms, then adjacent pairs only around
// singleton regions with useful signal. Diagnostics rank/refine the search;
// only host-verified HELPED can select a region.
bool common_flydelta_run_intervention_region_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_intervention_region_search_config & config,
        const common_flydelta_intervention_region_search_runner & runner,
        std::vector<common_flydelta_intervention_region_trial> & trials,
        common_flydelta_intervention_region_selection & selection,
        std::string & error);
