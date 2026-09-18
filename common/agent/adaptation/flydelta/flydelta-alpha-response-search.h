#pragma once

#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-decision-margin.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"
#include "agent/adaptation/flydelta/flydelta-dose-controller.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

enum class common_flydelta_alpha_response_status {
    inconclusive,
    helped,
    saturated,
    safety_limited,
    budget_limited,
    upper_bound_reached,
};

const char * common_flydelta_alpha_response_status_name(
        common_flydelta_alpha_response_status status);

// A bounded one-dimensional response search for an already selected rank-1
// direction/region. This is deliberately a low-level primitive: it consumes
// one bounded slice and does not decide Bootstrap/Shallow/Deep transitions.
// Margin is the preferred utility; geometry is the safe fallback when a
// decision pair is unavailable.
struct common_flydelta_alpha_response_search_config {
    int schema_version = 1;
    float seed_scale = 0.02f;
    // Expansion is deliberately faster than golden-section refinement. The
    // latter is used only after an observed response interval exists.
    float growth_factor = 2.0f;
    float max_scale = 0.64f;
    size_t max_expansion_trials = 5;
    size_t max_zoom_trials = 2;
    size_t max_min_effective_trials = 2;
    // Avoid stopping on one noisy utility reversal during exploration.
    size_t max_expansion_non_improving = 2;
    float utility_epsilon = 0.0001f;
    float max_leakage = 1.0f;
    float max_shift_norm = 1.0f;
    float min_cosine = 0.0f;
    float leakage_penalty = 0.10f;
    bool use_dose_controller = true;
    size_t max_dose_retries = 1;
    common_flydelta_dose_policy dose_policy;
};

struct common_flydelta_alpha_response_trial {
    float scale = 0.0f;
    float requested_scale = 0.0f;
    common_flydelta_dose_action dose_action = common_flydelta_dose_action::reject;
    float relative_dose = 0.0f;
    bool dose_evaluated = false;
    bool dose_safety_limited = false;
    std::string dose_reason;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
    common_flydelta_counterfactual_trial counterfactual;
    common_flydelta_decision_margin margin;
    common_flydelta_representation_diagnostics geometry;
    bool geometry_available = false;
    bool safe_to_continue = false;
    bool margin_available = false;
    float margin_delta_total = 0.0f;
    float margin_delta_normalized = 0.0f;
    float utility = 0.0f;
    bool refinement = false;
};

struct common_flydelta_alpha_response_selection {
    bool selected = false;
    float scale = 0.0f;
    float utility = 0.0f;
    // Best decision signal observed by the bounded response search. This is
    // kept separately from utility because leakage-penalized utility is not
    // the same quantity as the fixture-relative teacher-forced margin.
    float best_margin_delta_total = 0.0f;
    float best_margin_delta_normalized = 0.0f;
    bool best_margin_available = false;
    size_t trial_index = 0;
    common_flydelta_alpha_response_status response_status =
        common_flydelta_alpha_response_status::inconclusive;
    float last_scale = 0.0f;
    float last_requested_scale = 0.0f;
    float last_executed_scale = 0.0f;
    float last_relative_dose = 0.0f;
    common_flydelta_dose_action last_dose_action = common_flydelta_dose_action::reject;
    bool last_dose_evaluated = false;
    bool last_dose_safety_limited = false;
    float last_utility = 0.0f;
    float utility_slope = 0.0f;
    float max_reachable_scale = 0.0f;
    bool range_not_exhausted = false;
    bool minimum_effective_available = false;
    float minimum_effective_scale = 0.0f;
    size_t minimum_effective_trial_index = 0;
};

using common_flydelta_alpha_response_runner = std::function<bool(
        const common_flydelta_experiment_fixture & fixture,
        float scale,
        bool apply_overlay,
        common_flydelta_counterfactual_trial & trial,
        common_flydelta_decision_margin & margin,
        common_flydelta_representation_diagnostics & geometry,
        bool & geometry_available,
        std::string & error)>;

bool common_flydelta_alpha_response_search_config_validate(
        const common_flydelta_alpha_response_search_config & config,
        std::string & error);
bool common_flydelta_alpha_response_trial_validate(
        const common_flydelta_alpha_response_trial & trial,
        std::string & error);

bool common_flydelta_run_alpha_response_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_alpha_response_search_config & config,
        const common_flydelta_alpha_response_runner & runner,
        std::vector<common_flydelta_alpha_response_trial> & trials,
        common_flydelta_alpha_response_selection & selection,
        std::string & error);
