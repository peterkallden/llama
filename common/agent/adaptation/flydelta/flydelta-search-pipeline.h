#pragma once

#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-layer-search.h"
#include "agent/adaptation/flydelta/flydelta-scale-search.h"
#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"
#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// This is the host-side composition point for the existing FlyDelta searches.
// It deliberately owns no model context, overlay, artifact store or learning
// promotion. The runner remains responsible for fresh inference contexts and
// host verification.
struct common_flydelta_search_pipeline_config {
    int schema_version = 1;
    size_t dimension = 0;
    size_t max_directions = 8;
    common_flydelta_layer_search_config layer;
    common_flydelta_scale_search_config scale;
    // The region scan is the normal bounded WHERE x HOW MUCH phase. The
    // legacy layer planner remains available as an explicit fallback while
    // callers migrate their pipeline configuration.
    bool use_intervention_region_search = true;
    size_t region_max_singleton_layers = 4;
    size_t region_max_neighborhoods = 4;
    size_t region_max_trials = 32;
    size_t region_max_stalled_scales = 2;
};

struct common_flydelta_search_pipeline_direction {
    common_flydelta_direction_candidate direction;
    std::vector<common_flydelta_layer_diagnostic> layer_diagnostics;
    std::vector<uint32_t> available_layers;
};

struct common_flydelta_search_pipeline_layer_result {
    common_flydelta_layer_candidate candidate;
    std::vector<common_flydelta_scale_trial> scale_trials;
    common_flydelta_scale_selection scale_selection;
    common_flydelta_counterfactual_trial representative_trial;
};

struct common_flydelta_search_pipeline_direction_result {
    common_flydelta_direction_candidate direction;
    common_flydelta_layer_search_plan layer_plan;
    std::vector<common_flydelta_layer_search_trial> layer_trials;
    std::vector<common_flydelta_search_pipeline_layer_result> layer_results;
    common_flydelta_layer_search_selection layer_selection;
    std::vector<common_flydelta_intervention_region_trial> region_trials;
    common_flydelta_intervention_region_selection region_selection;
};

struct common_flydelta_search_pipeline_selection {
    bool selected = false;
    bool intervention_region = false;
    size_t direction_index = 0;
    size_t layer_result_index = 0;
    size_t region_trial_index = 0;
    float scale = 0.0f;
    float score = 0.0f;
};

struct common_flydelta_search_pipeline_result {
    std::vector<common_flydelta_search_pipeline_direction_result> directions;
    common_flydelta_search_pipeline_selection selection;
};

bool common_flydelta_search_pipeline_config_validate(
        const common_flydelta_search_pipeline_config & config,
        std::string & error);
bool common_flydelta_search_pipeline_direction_validate(
        const common_flydelta_search_pipeline_direction & direction,
        const common_flydelta_search_pipeline_config & config,
        std::string & error);

// The runner owns direction-to-overlay composition. A null layer candidate
// with apply_overlay=false is the baseline. For a non-null layer candidate,
// scale is the total intervention budget; the runner is responsible for
// applying it across the candidate's layer mask. Geometry is diagnostic only.
using common_flydelta_search_pipeline_runner = std::function<bool(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_direction_candidate & direction,
        const common_flydelta_layer_candidate * layer,
        float scale,
        bool apply_overlay,
        common_flydelta_counterfactual_trial & trial,
        common_flydelta_decision_margin & margin,
        common_flydelta_scale_geometry & geometry,
        std::string & error)>;

// Composes the existing searches without replacing them:
// direction -> layer plan/trials -> scale search per layer candidate.
// Only host-verified HELPED scale trials populate selection. UNKNOWN and
// NEUTRAL trials remain in the result for lifecycle/refinement handling, while
// geometry never creates learning evidence.
bool common_flydelta_run_search_pipeline(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_search_pipeline_config & config,
        const std::vector<common_flydelta_search_pipeline_direction> & directions,
        const common_flydelta_search_pipeline_runner & runner,
        common_flydelta_search_pipeline_result & result,
        std::string & error);

// Appends every executed scale arm as an experimental lifecycle result. The
// caller supplies the immutable experimental artifact id; this helper only
// records search state and never admits, activates or promotes that artifact.
// A host-classified UNKNOWN is valid here: host_verified means that the host
// evaluated the arm, not that it could prove the behavior.
bool common_flydelta_append_search_pipeline_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_search_pipeline_result & result,
        const std::string & experimental_artifact_id,
        std::string & error);
