#pragma once

#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Whirlpool is a bounded, derivative-free WHERE search. It uses a small set
// of probes around a discrete layer centre, estimates which probe is more
// promising from the existing diagnostics, then recentres and shrinks the
// neighbourhood. It does not build directions, update DeltaMemory or make a
// host verdict.
struct common_flydelta_whirlpool_search_config {
    int schema_version = 1;
    std::vector<uint32_t> available_layers;
    // Dense discovery/previous search anchors are priors, not evidence. They
    // are used only to choose the initial centre and are never promoted.
    std::vector<uint32_t> seed_layers;
    std::vector<common_flydelta_layer_diagnostic> layer_diagnostics;
    float total_scale = 0.02f;
    size_t max_rounds = 2;
    size_t probes_per_round = 4;
    size_t max_trials = 8;
    uint32_t initial_radius = 2;
    float shrink_factor = 0.5f;
    float margin_weight = 1.0f;
    float geometry_weight = 0.25f;
    float leakage_penalty = 0.10f;
    float max_leakage = 1.0f;
    float max_shift_norm = 1.0f;
    float min_cosine = 0.3f;
    // Optional common dose regulator. It may request one explicit lower retry
    // after an unsafe probe; it never selects a layer or creates evidence.
    bool use_dose_controller = true;
    size_t max_dose_retries = 1;
    common_flydelta_dose_policy dose_policy;
};

bool common_flydelta_whirlpool_search_config_validate(
        const common_flydelta_whirlpool_search_config & config,
        std::string & error);

struct common_flydelta_whirlpool_round_trace {
    size_t round = 0;
    uint32_t centre_before = 0;
    uint32_t radius_before = 0;
    std::vector<uint32_t> probed_layers;
    uint32_t best_probe_layer = 0;
    float best_probe_score = 0.0f;
    uint32_t centre_after = 0;
    uint32_t radius_after = 0;
};

struct common_flydelta_whirlpool_trace {
    int schema_version = 1;
    size_t model_evaluations = 0;
    size_t best_trial_index = static_cast<size_t>(-1);
    float best_search_score = 0.0f;
    uint32_t final_centre = 0;
    uint32_t final_radius = 0;
    std::vector<common_flydelta_whirlpool_round_trace> rounds;
};

bool common_flydelta_whirlpool_trace_validate(
        const common_flydelta_whirlpool_trace & trace,
        const common_flydelta_whirlpool_search_config & config,
        size_t trial_count,
        std::string & error);

using common_flydelta_whirlpool_search_runner = std::function<bool(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_intervention_region_candidate * candidate,
        common_flydelta_counterfactual_trial & trial,
        common_flydelta_decision_margin & margin,
        common_flydelta_representation_diagnostics & geometry,
        bool & geometry_available,
        std::string & error)>;

// Runs a bounded layer-only Whirlpool search. The baseline is evaluated once;
// every other arm uses the existing candidate/overlay runner. UNKNOWN and
// NEUTRAL may guide later probes through diagnostics, but only a host-known
// HELPED trial is returned as selected.
bool common_flydelta_run_whirlpool_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_whirlpool_search_config & config,
        const common_flydelta_whirlpool_search_runner & runner,
        std::vector<common_flydelta_intervention_region_trial> & trials,
        common_flydelta_intervention_region_selection & selection,
        common_flydelta_whirlpool_trace & trace,
        std::string & error);
