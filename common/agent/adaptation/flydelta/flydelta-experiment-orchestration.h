#pragma once

#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"
#include "agent/adaptation/flydelta/flydelta-alpha-response-search.h"
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

// Bootstrap refinement remains one bounded production phase. This kind tells
// the host which rank-one refinement primitive the next slice should execute;
// it does not create a new worker lane or change evidence capacity.
enum class common_flydelta_bootstrap_refinement_kind {
    bootstrap_zoom,
    adaptive_alpha,
};

const char * common_flydelta_bootstrap_refinement_kind_name(
        common_flydelta_bootstrap_refinement_kind kind);

const char * common_flydelta_experiment_phase_name(
        common_flydelta_experiment_phase phase);

// The host resolves compatible deltas for continuation.region's layer, then
// runs this bounded plan. This carries no learning or promotion authority.
struct common_flydelta_experiment_plan {
    common_flydelta_search_continuation continuation;
    // Evidence depth is the maximum permitted search depth. The current
    // phase always advances sequentially under UtilityGate control.
    common_flydelta_search_depth depth = common_flydelta_search_depth::bootstrap;
    common_flydelta_experiment_phase phase = common_flydelta_experiment_phase::bootstrap;
    common_flydelta_bootstrap_refinement_kind bootstrap_refinement =
        common_flydelta_bootstrap_refinement_kind::bootstrap_zoom;
    common_flydelta_search_budget budget;
    size_t required_compatible_directions = 1;
    bool require_decision_margin = false;
    bool run_rank_two_controls_first = false;
    // Evidence capacity only. This is not runtime permission: UtilityGate
    // must still approve TFO after rank-two controls.
    bool tfo_lite_permitted_by_evidence = false;
    bool tfo_lite_requires_utility_gate = false;
    // Set only after Deep controls have earned a separate UtilityGate action.
    bool run_tfo_lite = false;
    // Set for the bounded experimental escape from a rank-one plateau. This
    // does not change evidence depth and does not imply a Shallow transition.
    bool run_orthogonal_search = false;
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
    // Optional summary emitted by AdaptiveAlphaSearch. It is search utility,
    // never host evidence or learning credit.
    bool alpha_response_available = false;
    common_flydelta_alpha_response_status alpha_response_status =
        common_flydelta_alpha_response_status::inconclusive;
    bool alpha_range_not_exhausted = false;
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
    // Bootstrap has plateaued, so inspect a second, search-derived axis. The
    // resulting rank-2 space remains experimental until host verification.
    orthogonal_search,
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

// Cross-slice action returned to the host scheduler. This is policy output,
// not an instruction for the evaluator to recurse into another phase.
enum class common_flydelta_next_action {
    stop,
    retain,
    run_bootstrap,
    refine_bootstrap,
    run_orthogonal_search,
    run_shallow_controls,
    run_deep_controls,
    recenter_surface,
    run_representation_augmentation,
    prepare_concept_material,
    run_concept_synthesis,
    allow_tfo_lite,
};

const char * common_flydelta_next_action_name(common_flydelta_next_action action);

struct common_flydelta_slice_orchestration_result {
    common_flydelta_next_action next_action = common_flydelta_next_action::retain;
    common_flydelta_utility_gate_decision utility;
    common_flydelta_experiment_plan plan;
    bool plan_advanced = false;
    std::string reason;
};

// Rank-one plateau detection is a search gate. It never changes evidence
// rank; it only decides whether a locally useful Bootstrap search deserves an
// experimental orthogonal probe.
struct common_flydelta_rank1_plateau_config {
    int schema_version = 1;
    size_t minimum_bootstrap_arms = 6;
    size_t required_plateau_rounds = 2;
    float minimum_useful_margin = 0.0f;
    float maximum_recent_gain_ratio = 0.075f;
    uint32_t maximum_region_span = 3;
    float shallow_rank_threshold = 1.5f;
};

struct common_flydelta_rank1_plateau_round {
    bool safe_to_continue = false;
    size_t evaluated_arms = 0;
    uint32_t anchor_layer = 0;
    float best_margin_delta = 0.0f;
};

enum class common_flydelta_rank1_plateau_action {
    continue_bootstrap,
    refine_bootstrap,
    orthogonal_search,
};

struct common_flydelta_rank1_plateau_result {
    bool eligible = false;
    bool plateau = false;
    common_flydelta_rank1_plateau_action action =
        common_flydelta_rank1_plateau_action::continue_bootstrap;
    size_t safe_arm_count = 0;
    size_t plateau_streak = 0;
    float best_margin_delta = 0.0f;
    float recent_gain_ratio = 0.0f;
    uint32_t minimum_anchor_layer = 0;
    uint32_t maximum_anchor_layer = 0;
};

bool common_flydelta_rank1_plateau_config_validate(
        const common_flydelta_rank1_plateau_config & config,
        std::string & error);
bool common_flydelta_evaluate_rank1_plateau(
        const common_flydelta_rank1_plateau_config & config,
        float effective_rank,
        const std::vector<common_flydelta_rank1_plateau_round> & rounds,
        common_flydelta_rank1_plateau_result & result,
        std::string & error);

// Converts the plateau gate's search-only result into the common UtilityGate
// decision consumed by the plan transition helper. This adapter never grants
// learning or promotion credit.
bool common_flydelta_decide_rank1_plateau_utility(
        const common_flydelta_rank1_plateau_result & plateau,
        common_flydelta_utility_gate_decision & decision,
        std::string & error);

// A Bootstrap arm descriptor is the actual bounded intervention in a common
// flattened layer/profile space. Keeping the descriptor here, rather than a
// score-only summary, prevents an apparent orthogonal signal from being
// inferred from alpha values that all lie on the same rank-one ray.
struct common_flydelta_orthogonal_search_arm {
    std::vector<float> intervention;
    // Decision utility is preferred whenever enough arms carry it.  It is
    // deliberately optional: a rank-one plateau may still expose a coherent
    // geometric response before a behavior-specific margin becomes useful.
    bool decision_margin_available = false;
    float decision_margin_delta = 0.0f;
    // The caller supplies a dimensionless directional response derived from
    // compatible diagnostics. Relative dose is kept separate as a safety and
    // response-presence signal. This arm may only open one experimental
    // orthogonal probe; it never supplies learning credit or permission for
    // further rank-two expenditure.
    bool geometric_response_available = false;
    float geometric_response = 0.0f;
    bool safe_to_continue = false;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
};

enum class common_flydelta_orthogonal_response_signal {
    none,
    decision_margin,
    geometric_response,
};

const char * common_flydelta_orthogonal_response_signal_name(
        common_flydelta_orthogonal_response_signal signal);

struct common_flydelta_orthogonal_search_config {
    int schema_version = 1;
    size_t minimum_arms = 5;
    size_t maximum_arms = 32;
    float minimum_residual_norm = 0.0001f;
    float minimum_fit_quality = 0.1f;
    float ridge = 0.001f;
    // Admission for a geometric-response fallback. These are intentionally
    // safety bounds only; decision utility remains the gate for further
    // rank-two expenditure.
    float minimum_cosine = 0.3f;
    float maximum_leakage = 1.0f;
    float maximum_shift_norm = 1.0f;
    // Relative dose indicates that an intervention moved the representation;
    // this separate penalty keeps leakage from being mistaken for useful
    // orthogonal response when no teacher-forced margin is available.
    float geometric_leakage_penalty = 0.10f;
};

struct common_flydelta_orthogonal_search_result {
    bool available = false;
    bool experimental_only = true;
    size_t source_arm_count = 0;
    float residual_norm = 0.0f;
    float fit_quality = 0.0f;
    common_flydelta_orthogonal_response_signal response_signal =
        common_flydelta_orthogonal_response_signal::none;
    std::vector<float> direction;
};

// Reference-free preparation owned by the common layer. A host/model adapter
// supplies only the persisted BootstrapZoom state, then uses this typed input
// to run one bounded orthogonal validation slice on fresh contexts.
struct common_flydelta_orthogonal_search_input {
    std::vector<uint32_t> local_layers;
    std::vector<float> rank1_intervention;
    std::vector<common_flydelta_orthogonal_search_arm> arms;
};

bool common_flydelta_orthogonal_search_config_validate(
        const common_flydelta_orthogonal_search_config & config,
        std::string & error);
bool common_flydelta_build_orthogonal_search_direction(
        const common_flydelta_orthogonal_search_config & config,
        const std::vector<float> & rank1_direction,
        const std::vector<common_flydelta_orthogonal_search_arm> & arms,
        common_flydelta_orthogonal_search_result & result,
        std::string & error);
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
    adaptive_alpha,
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

// One host-evaluated BootstrapZoom arm. This is experimental search state:
// UNKNOWN may be retained when its diagnostic signal is useful, but it has
// no learning or promotion authority.
struct common_flydelta_bootstrap_zoom_trial {
    common_flydelta_bootstrap_zoom_candidate candidate;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
    bool host_evaluated = false;
    // The verifier-known flag remains separate: UNKNOWN is valid search state.
    bool verifier_known = false;
    bool margin_available = false;
    float margin_delta = 0.0f;
    bool diagnostics_available = false;
    common_flydelta_representation_diagnostics diagnostics;
};

struct common_flydelta_bootstrap_zoom_selection {
    bool selected = false;
    size_t trial_index = 0;
    float search_score = 0.0f;
};

// Reference-safe resume state for the real worker. It contains only search
// position and diagnostics; activation tensors and raw model material stay in
// their existing stores. The host persists this state under state_ref and may
// attach the reference to the next queued job.
struct common_flydelta_bootstrap_zoom_state {
    int schema_version = 1;
    std::string state_ref;
    std::string behavior_key;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    common_flydelta_bootstrap_zoom_phase phase =
        common_flydelta_bootstrap_zoom_phase::alpha_zoom;
    common_flydelta_bootstrap_refinement_kind refinement_kind =
        common_flydelta_bootstrap_refinement_kind::bootstrap_zoom;
    uint32_t anchor_layer = 0;
    float selected_scale = 0.0f;
    float best_margin_delta = 0.0f;
    float best_search_score = 0.0f;
    size_t extra_model_trials = 0;
    size_t next_candidate_index = 0;
    // Search-surface lineage is not evidence or promotion state. It lets a
    // worker resume an orthogonal/search-derived surface without inflating
    // host-certified evidence rank.
    uint32_t surface_revision = 1;
    uint32_t parent_surface_revision = 0;
    size_t search_rank = 1;
    float evidence_rank = 1.0f;
    std::string surface_origin = "bootstrap_rank1";
    std::string parent_surface_ref;
    std::vector<uint32_t> local_layers;
    // The best safe experimental arm so far. This is persisted with resume
    // state so a later worker slice can retain/refine the same candidate.
    std::vector<common_flydelta_bootstrap_zoom_trial> completed_trials;
    // Rank-two controls are kept separately from rank-one BootstrapZoom
    // trials. They describe the next experimental surface and never become
    // learning evidence by being present here.
    std::vector<common_flydelta_bootstrap_zoom_trial> surface_trials;
    common_flydelta_bootstrap_zoom_selection selection;
    // The low-level AdaptiveAlpha result is summarized here so the next
    // bounded worker slice can be orchestrated without replaying prior arms.
    bool alpha_response_available = false;
    common_flydelta_alpha_response_selection alpha_response;
};

bool common_flydelta_bootstrap_zoom_state_validate(
        const common_flydelta_bootstrap_zoom_state & state,
        std::string & error);

// Projects bounded host-evaluated BootstrapZoom arms into the common
// orthogonal-search input and marks their safety eligibility. It carries
// neither prompts nor activations and does not decide evidence depth,
// lifecycle, or the next action.
bool common_flydelta_prepare_orthogonal_search_input(
        const common_flydelta_orthogonal_search_config & config,
        const common_flydelta_bootstrap_zoom_state & state,
        common_flydelta_orthogonal_search_input & input,
        std::string & error);

bool common_flydelta_bootstrap_zoom_config_validate(
        const common_flydelta_bootstrap_zoom_config & config,
        std::string & error);
bool common_flydelta_bootstrap_zoom_candidate_validate(
        const common_flydelta_bootstrap_zoom_candidate & candidate,
        std::string & error);
bool common_flydelta_bootstrap_zoom_trial_validate(
        const common_flydelta_bootstrap_zoom_trial & trial,
        std::string & error);
bool common_flydelta_bootstrap_zoom_selection_validate(
        const common_flydelta_bootstrap_zoom_selection & selection,
        size_t trial_count,
        std::string & error);

// Chooses a retained experimental arm. HELPED wins when present; otherwise a
// host-classified safe UNKNOWN/NEUTRAL arm with the strongest margin signal
// wins. This selection does not award learning credit or promotion.
bool common_flydelta_select_bootstrap_zoom_trial(
        const std::vector<common_flydelta_bootstrap_zoom_trial> & trials,
        common_flydelta_bootstrap_zoom_selection & selection,
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

// Advances one bounded phase after the host has evaluated that phase's arms.
// Evidence depth limits the maximum phase; utility determines whether the
// next phase earns model budget. BootstrapZoom is a same-phase refinement,
// Shallow controls must precede Deep, and Deep controls must precede TFO.
bool common_flydelta_advance_experiment_plan(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_decision & utility,
        common_flydelta_experiment_plan & next,
        bool & advanced,
        std::string & error);

// Applies EvidenceGate + UtilityGate to one completed bounded slice. It
// never invokes model execution; the caller persists the returned plan and
// lets the host scheduler decide whether to enqueue next_action.
bool common_flydelta_orchestrate_search_slice(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error);
