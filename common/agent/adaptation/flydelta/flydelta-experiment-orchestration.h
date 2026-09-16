#pragma once

#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"

#include <cstddef>
#include <string>

// A WHERE result is not evidence. This reference-only continuation bridges
// bounded region search and later per-behavior WHAT aggregation. UNKNOWN arms
// with useful diagnostics may be retained here; promotion still needs HELPED.
struct common_flydelta_search_continuation {
    int schema_version = 1;
    size_t direction_index = 0;
    size_t region_trial_index = 0;
    common_flydelta_intervention_region_candidate region;
    float search_score = 0.0f;
    bool host_helped = false;
};

enum class common_flydelta_experiment_phase {
    bootstrap,
    shallow_controls,
    deep_controls,
};

const char * common_flydelta_experiment_phase_name(
        common_flydelta_experiment_phase phase);

// The host resolves compatible deltas for continuation.region's layer, then
// runs this bounded plan. This carries no learning or promotion authority.
struct common_flydelta_experiment_plan {
    common_flydelta_search_continuation continuation;
    common_flydelta_search_depth depth = common_flydelta_search_depth::bootstrap;
    common_flydelta_experiment_phase phase = common_flydelta_experiment_phase::bootstrap;
    common_flydelta_search_budget budget;
    size_t required_compatible_directions = 1;
    bool require_decision_margin = false;
    bool run_rank_two_controls_first = false;
    bool allow_tfo_lite_after_controls = false;
};

// Selects the best diagnostic WHERE arm. HELPED is preferred; otherwise the
// best safe/promising UNKNOWN or NEUTRAL arm wins by search score.
bool common_flydelta_select_search_continuation(
        const common_flydelta_search_pipeline_result & pipeline,
        common_flydelta_search_continuation & continuation,
        std::string & error);

// Turns per-layer compatible evidence into Bootstrap/Shallow/Deep work.
// Shallow/Deep always run rank-two controls before any coefficient optimizer.
bool common_flydelta_plan_search_continuation(
        const common_flydelta_search_continuation & continuation,
        const common_flydelta_evidence_depth_result & evidence_depth,
        common_flydelta_experiment_plan & plan,
        std::string & error);
