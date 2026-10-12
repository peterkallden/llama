#pragma once

#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-dose-controller.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// Generic host-side scale search for an already selected overlay direction or
// layer mask. Geometry is a safety/prioritization signal only; host
// verification remains the sole source of HELPED.
struct common_flydelta_scale_search_config {
    int schema_version = 1;
    float initial_scale = 0.02f;
    float growth_factor = 2.0f;
    float max_scale = 0.32f;
    size_t max_geometric_trials = 4;
    size_t max_refinement_trials = 1;
    float min_cosine = 0.3f;
    float max_leakage = 1.0f;
    float max_shift_norm = 1.0f;
    float saturation_epsilon = 0.00001f;
    // When enabled, initial_scale/growth_factor/max_scale are relative to a
    // measured positive/negative activation separation. The runner receives
    // the resolved, bounded scale while trials retain both values.
    bool separation_calibrated = false;
    float reference_separation = 1.0f;
    float max_resolved_scale = 1.0f;
    bool use_dose_controller = true;
    size_t max_dose_retries = 1;
    common_flydelta_dose_policy dose_policy;
};

struct common_flydelta_scale_geometry {
    bool available = false;
    float cosine = 0.0f;
    float progress = 0.0f;
    float leakage = 0.0f;
    float shift_norm = 0.0f;
};

struct common_flydelta_scale_trial {
    float scale = 0.0f;
    float requested_scale = 0.0f;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
    float quality_delta = 0.0f;
    bool executed = false;
    bool verifier_known = false;
    bool geometry_available = false;
    bool safe_to_escalate = false;
    bool refinement = false;
    bool separation_calibrated = false;
    bool scale_clamped = false;
    float dose_requested_strength = 0.0f;
    float dose_executed_strength = 0.0f;
    common_flydelta_dose_action dose_action = common_flydelta_dose_action::reject;
    float relative_dose = 0.0f;
    bool dose_evaluated = false;
    bool dose_safety_limited = false;
    std::string dose_reason;
    common_flydelta_scale_geometry geometry;
    std::string evidence_ref;
};

struct common_flydelta_scale_selection {
    bool selected = false;
    float scale = 0.0f;
    float score = 0.0f;
    size_t trial_index = 0;
    float requested_scale = 0.0f;
};

bool common_flydelta_scale_search_config_validate(
        const common_flydelta_scale_search_config & config,
        std::string & error);

// Resolves a relative scale against a measured layer separation. The result
// is bounded by max_resolved_scale; clamping is reported so a caller can stop
// geometric escalation rather than repeat the same arm.
bool common_flydelta_resolve_scale(
        const common_flydelta_scale_search_config & config,
        float requested_scale,
        float & resolved_scale,
        bool & clamped,
        std::string & error);
bool common_flydelta_scale_trial_validate(
        const common_flydelta_scale_trial & trial,
        std::string & error);

using common_flydelta_scale_search_runner = std::function<bool(
        const common_flydelta_experiment_fixture & fixture,
        float scale,
        bool apply_overlay,
        common_flydelta_counterfactual_trial & trial,
        common_flydelta_scale_geometry & geometry,
        std::string & error)>;

// Runs one baseline, then a bounded geometric scale sequence. Escalation stops
// when geometry is unavailable/unsafe. After the first verified HELPED arm,
// at most max_refinement_trials midpoint(s) are tested between that arm and
// the latest smaller attempted scale. The helper never turns geometry into a
// verdict or learning signal.
bool common_flydelta_run_scale_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_scale_search_config & config,
        const common_flydelta_scale_search_runner & runner,
        std::vector<common_flydelta_scale_trial> & trials,
        common_flydelta_scale_selection & selection,
        std::string & error);
