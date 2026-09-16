#pragma once

#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"
#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"

#include <cstddef>
#include <string>
#include <vector>

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
    // Evidence capacity only. This is not runtime permission: UtilityGate
    // must still approve TFO after rank-two controls.
    bool tfo_lite_permitted_by_evidence = false;
    bool tfo_lite_requires_utility_gate = false;
};

// Evidence controls search capacity; this policy controls whether observed
// subspace utility has earned more model work inside that capacity. It carries
// no host outcome or learning credit.
struct common_flydelta_utility_gate_config {
    int schema_version = 1;
    float shallow_enter_margin = 0.0f;
    float deep_enter_margin = 0.0f;
    float tfo_enter_margin = 0.0f;
    size_t shallow_enter_observations = 1;
    size_t deep_enter_observations = 1;
    size_t tfo_enter_observations = 1;
    size_t exit_nonqualifying_observations = 2;
    float min_cosine = 0.3f;
    float max_leakage = 1.0f;
    float max_shift_norm = 1.0f;
};

struct common_flydelta_subspace_utility_observation {
    bool safe_to_continue = false;
    bool decision_margin_available = false;
    float decision_margin_delta = 0.0f;
    bool geometry_available = false;
    common_flydelta_representation_diagnostics geometry;
};

struct common_flydelta_utility_history {
    size_t qualifying_streak = 0;
    size_t nonqualifying_streak = 0;
};

enum class common_flydelta_utility_gate_action {
    stop,
    retain,
    // A useful rank-one signal may justify a small local WHERE/HOW MUCH
    // refinement even when evidence capacity does not yet permit Shallow.
    refine_bootstrap,
    escalate_shallow,
    escalate_deep,
    allow_tfo_lite,
};

struct common_flydelta_utility_gate_decision {
    bool utility_qualified = false;
    common_flydelta_utility_gate_action action = common_flydelta_utility_gate_action::retain;
    common_flydelta_utility_history history;
};

const char * common_flydelta_utility_gate_action_name(
        common_flydelta_utility_gate_action action);
bool common_flydelta_utility_gate_config_validate(
        const common_flydelta_utility_gate_config & config,
        std::string & error);
bool common_flydelta_subspace_utility_observation_validate(
        const common_flydelta_subspace_utility_observation & observation,
        std::string & error);
bool common_flydelta_decide_subspace_utility(
        const common_flydelta_utility_gate_config & config,
        common_flydelta_search_depth max_allowed_depth,
        common_flydelta_experiment_phase current_phase,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_utility_gate_decision & decision,
        std::string & error);

// BootstrapZoom remains strictly rank-one. It refines alpha and a local layer
// profile around a Whirlpool/region seed, but does not introduce a second WHAT
// direction, create learning credit, or alter evidence depth.
enum class common_flydelta_bootstrap_zoom_phase {
    alpha_zoom,
    profile_zoom,
    sign_control,
};

const char * common_flydelta_bootstrap_zoom_phase_name(
        common_flydelta_bootstrap_zoom_phase phase);

struct common_flydelta_bootstrap_zoom_config {
    int schema_version = 1;
    size_t max_extra_model_trials = 8;
    std::vector<float> alpha_multipliers = {0.5f, 1.0f, 1.5f};
    bool include_triplet_profile = true;
    bool include_opposite_sign_control = true;
    float min_margin_improvement = 0.0001f;
};

// Coefficients are profile weights, one per layer, and must have unit L2
// energy. The caller uses each layer's own compatible rank-one direction d_L:
//   delta_h_L = total_scale * layer_weights[L] * d_L.
struct common_flydelta_bootstrap_zoom_candidate {
    int schema_version = 1;
    common_flydelta_bootstrap_zoom_phase phase =
        common_flydelta_bootstrap_zoom_phase::alpha_zoom;
    std::vector<uint32_t> layer_indices;
    std::vector<float> layer_weights;
    float total_scale = 0.0f;
    bool opposite_sign_control = false;
};

bool common_flydelta_bootstrap_zoom_config_validate(
        const common_flydelta_bootstrap_zoom_config & config,
        std::string & error);
bool common_flydelta_bootstrap_zoom_candidate_validate(
        const common_flydelta_bootstrap_zoom_candidate & candidate,
        std::string & error);

// Deterministic, bounded proposal helpers. Alpha probes are emitted first.
// Profile probes use normalized layer weights, so singleton/pair/triplet arms
// consume comparable total intervention energy.
bool common_flydelta_propose_bootstrap_alpha_zoom(
        uint32_t anchor_layer,
        float base_scale,
        const common_flydelta_bootstrap_zoom_config & config,
        std::vector<common_flydelta_bootstrap_zoom_candidate> & candidates,
        std::string & error);
bool common_flydelta_propose_bootstrap_profile_zoom(
        const std::vector<uint32_t> & local_layers,
        uint32_t anchor_layer,
        float selected_scale,
        const common_flydelta_bootstrap_zoom_config & config,
        std::vector<common_flydelta_bootstrap_zoom_candidate> & candidates,
        std::string & error);

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
