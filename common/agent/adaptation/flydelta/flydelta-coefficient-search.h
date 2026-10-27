#pragma once

#include "agent/adaptation/flydelta/flydelta-concept.h"
#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-decision-margin.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"
#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"
#include "agent/adaptation/flydelta/flydelta-dose-controller.h"
#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"
#include "agent/adaptation/flydelta/experiment/flydelta-outcome-guidance.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

// A small per-layer low-rank basis. The basis is orthonormalized by the host
// adapter, while the search below remains independent of llama.cpp internals.
struct common_flydelta_low_rank_basis {
    int schema_version = 1;
    size_t dimension = 0;
    int32_t layer_index = -1;
    std::vector<std::vector<float>> vectors;
};

// Logical row reservations are separate from physical starts/completions.
// The host uses the existing lifecycle journal before entering inference.
struct common_flydelta_search_execution_budget {
    size_t limit = 256;
    size_t used = 0;
    size_t physical_rows_started = 0;
    size_t physical_rows_completed = 0;
    size_t physical_rows_unresolved = 0;
    size_t max_physical_attempts = 3;
    std::function<bool(const std::vector<std::string> &, const std::string &,
            std::string &, std::string &)> begin_execution;
    std::function<bool(const std::string &, size_t, bool, std::string &)> finish_execution;
};

bool common_flydelta_reserve_search_evaluations(
        common_flydelta_search_execution_budget & budget, size_t count, std::string & error);

// Stable identity shared by arm preparation, guidance and lifecycle records.
std::string common_flydelta_coefficient_basis_identity(
        const common_flydelta_low_rank_basis & basis);

// Reconstruct the complete applied intervention from the immutable basis and
// executed coefficients. No previous overlay is included in this result.
bool common_flydelta_reconstruct_intervention(
        const common_flydelta_low_rank_basis & basis,
        const std::string & expected_basis_identity,
        const std::vector<float> & applied_coefficients,
        std::vector<float> & intervention,
        std::string & error);

bool common_flydelta_low_rank_basis_validate(
        const common_flydelta_low_rank_basis & basis,
        size_t max_rank,
        std::string & error);

// Builds a bounded orthonormal basis from compatible direction candidates.
// This is intentionally Gram-Schmidt rather than a heavy trainer: rank one
// candidates (raw/trimmed/whitened/margin) and small rank-N subspaces share
// the same runtime representation.
bool common_flydelta_build_low_rank_basis(
        size_t dimension,
        size_t max_rank,
        const std::vector<common_flydelta_direction_candidate> & candidates,
        common_flydelta_low_rank_basis & basis,
        std::string & error);

// Builds the bounded rank-two basis for one typed prefer/avoid proposal.
// This is the experimental path for negative support; it does not change the
// stricter semantic-basis reuse admission below.
bool common_flydelta_build_paired_intervention_basis(
        const common_flydelta_paired_intervention_proposal & proposal,
        const common_flydelta_direction_candidate & prefer,
        const common_flydelta_direction_candidate & avoid,
        common_flydelta_low_rank_basis & basis,
        std::string & error);

// Semantic basis reuse is a strict admission layer in front of the existing
// Gram-Schmidt basis builder.  The resolver only composes descriptors that
// match the same concept/behavior/model/capture/Oracle identity and are
// already host-verified.  It does not create evidence, activate a basis or
// alter coefficient/TFO search.
struct common_flydelta_semantic_basis_query {
    int schema_version = 1;
    common_adaptation_evidence_source source =
        common_adaptation_evidence_source::tool_repair;
    std::string concept_key;
    std::string behavior_key;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string capture_layout_revision;
    std::string scope_fingerprint;
    std::string oracle_ref;
    std::string oracle_revision;
    size_t max_rank = 2;
};

bool common_flydelta_semantic_basis_query_validate(
        const common_flydelta_semantic_basis_query & query,
        std::string & error);

struct common_flydelta_semantic_direction_candidate {
    common_flydelta_direction_candidate direction;
    common_flydelta_synthesis_candidate_descriptor descriptor;
};

bool common_flydelta_resolve_semantic_basis(
        const common_flydelta_semantic_basis_query & query,
        const std::vector<common_flydelta_semantic_direction_candidate> & candidates,
        common_flydelta_low_rank_basis & basis,
        std::vector<size_t> & selected_candidate_indices,
        std::string & error);

// Coefficient search remains a host-side experiment strategy. The default
// coordinate stencil is deliberately retained; tfo_lite is an optional,
// bounded population search for mixed coefficients inside the same basis.
enum class common_flydelta_coefficient_search_strategy {
    coordinate,
    tfo_lite,
};

const char * common_flydelta_coefficient_search_strategy_name(
        common_flydelta_coefficient_search_strategy strategy);

// Experimental calibration is host-owned and never grants activation authority.
struct common_flydelta_residual_calibration {
    int schema_version = 1;
    std::string reference;
    std::string model_profile_fingerprint;
    std::string capture_layout_revision;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string token_selection;
    common_flydelta_capture_position position = common_flydelta_capture_position::generation_boundary;
    int32_t token_index = -1;
    int32_t layer_index = -1;
    size_t sample_count = 0;
    size_t dimension = 0;
    std::vector<float> sample_norms;
    float median_norm = 0.0f;
    std::string intervention_site;
    std::string measurement_stage;
    std::string runtime_configuration_fingerprint;
    std::string runtime_layout_revision;
    std::string dataset_revision;
    std::string aggregation;
    std::vector<std::string> sequence_ids;
};

// One contribution per independent sequence. Within-sequence positions are
// reduced first, so long sequences never gain extra calibration weight.
bool common_flydelta_calibrate_residual_sequences(
    const std::vector<std::vector<common_flydelta_hidden_state_capture>> & sequences,
    const std::vector<std::string> & sequence_ids, int32_t layer_index,
    const std::string & reference, const std::string & token_selection,
    common_flydelta_residual_calibration & calibration, std::string & error);

bool common_flydelta_calibrate_residual_dose(
    const std::vector<common_flydelta_hidden_state_capture> & captures,
    int32_t layer_index, const std::string & reference,
    const std::string & token_selection,
    common_flydelta_residual_calibration & calibration, std::string & error);

bool common_flydelta_intervention_norm(
    const common_flydelta_low_rank_basis & basis,
    const std::vector<float> & coefficients, float & value, std::string & error);

enum class common_flydelta_coefficient_dose_mode {
    raw_coefficients,
    residual_fraction,
    direction_dose,
};

struct common_flydelta_coefficient_search_config {
    int schema_version = 1;
    common_flydelta_coefficient_search_strategy strategy =
        common_flydelta_coefficient_search_strategy::coordinate;
    float step = 0.05f;
    // Opt-in experiment only. Legacy fields retain their original semantics.
    common_flydelta_coefficient_dose_mode experimental_dose_mode =
        common_flydelta_coefficient_dose_mode::raw_coefficients;
    // Extra composition diagnostics are opt-in even for the raw control arm.
    bool record_intervention_norms = false;
    common_flydelta_residual_calibration residual_calibration;
    std::string injection_capture_layout_revision;
    std::string injection_token_selection;
    common_flydelta_capture_position injection_capture_position = common_flydelta_capture_position::generation_boundary;
    int32_t injection_token_index = -1;
    std::string injection_intervention_site;
    std::string injection_runtime_configuration_fingerprint;
    std::string injection_runtime_layout_revision;
    float residual_fraction = 0.02f;
    std::vector<float> residual_fraction_candidates = {0.01f, 0.02f, 0.04f};
    // Optional host-resolved search center. These are the complete physical
    // coefficients, not an additional overlay on top of a previous arm.
    // The immutable basis identity prevents resuming in a different space.
    std::vector<float> anchor_coefficients;
    std::string anchor_basis_identity;
    std::vector<std::vector<float>> evaluated_coefficients;
    size_t max_candidates = 16;
    // Backend batch wave limit for diagnostic/model-facing coefficient arms.
    // Zero lets the registered batch runner own the limit. This is an
    // execution bound only; it does not change the number or order of search
    // proposals and it does not make coefficient search concurrent.
    size_t max_batch_arms = 0;
    float max_l2_norm = 0.32f;
    // TFO-lite only. These limits are intentionally small so the strategy
    // remains a bounded experiment rather than a background optimizer.
    uint64_t seed = 0x464c5944454c5441ULL;
    size_t population_size = 4;
    size_t iterations = 2;
    float exploration_scale = 1.0f;
    float norm_penalty = 0.05f;
    // Applied only when the existing representation diagnostics are
    // available for an arm. No second leakage calculation is introduced.
    float leakage_penalty = 0.10f;
    bool use_dose_controller = true;
    size_t max_dose_retries = 1;
    common_flydelta_dose_policy dose_policy;
    // Host-populated geometry used only to form comparable guidance features.
    int32_t guidance_layer_index = -1;
    size_t guidance_model_layer_count = 0;
    std::string guidance_basis_identity;
};

bool common_flydelta_coefficient_search_config_validate(
        const common_flydelta_coefficient_search_config & config,
        size_t rank,
        std::string & error);

struct common_flydelta_coefficient_trial {
    std::string arm_id;
    std::string intervention_ref;
    std::string verifier_reason;
    std::string observed_decision_summary;
    std::string expected_decision_summary;
    std::string evidence_ref;
    std::vector<float> coefficients;
    std::vector<float> requested_coefficients;
    float requested_strength = 0.0f;
    float executed_strength = 0.0f;
    common_flydelta_dose_action dose_action = common_flydelta_dose_action::reject;
    // Legacy behavior-delta-relative displacement, not residual-relative dose.
    float relative_dose = 0.0f;
    bool intervention_norms_available = false;
    float requested_intervention_norm = 0.0f;
    float applied_intervention_norm = 0.0f;
    float requested_residual_fraction = 0.0f;
    float applied_residual_fraction = 0.0f;
    std::string residual_calibration_ref;
    bool dose_evaluated = false;
    bool dose_safety_limited = false;
    std::string dose_reason;
    common_flydelta_decision_margin margin;
    common_flydelta_margin_comparison margin_comparison;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
    // Host-observed partial progress is separate from the final verdict.
    common_flydelta_semantic_progress_observation semantic_progress_observation;
    common_flydelta_semantic_progress semantic_progress;
    float quality_delta = 0.0f;
    float search_fitness = 0.0f;
    bool geometry_available = false;
    common_flydelta_representation_diagnostics geometry;
    bool executed = false;
    bool host_evaluated = false;
    bool verifier_known = false;
    bool overlay_applied = false;
    // Forecast identity/features are frozen when the adviser ranks the
    // proposal. Later scoring must not rewrite the predictor inputs.
    std::string guidance_arm_id;
    std::string guidance_basis_id;
    int guidance_margin_sign = 0;
    size_t iteration = 0;
    size_t parent_trial_index = static_cast<size_t>(-1);
    std::string mutation_kind;
    // Original host observations retained for presentation and validated progress
    // projection. They never bypass search admission or lifecycle authority.
    std::optional<common_flydelta_counterfactual_trial> candidate_observation;
    std::optional<common_flydelta_counterfactual_trial> baseline_observation;
    std::string progress_parent_ref;
    std::optional<common_flydelta_progress_assessment> progress_parent;
};

bool common_flydelta_coefficient_trial_progress_evidence(
        const common_flydelta_coefficient_trial & trial,
        common_flydelta_progress_evidence & evidence, std::string & error);

struct common_flydelta_coefficient_selection {
    bool selected = false;
    size_t trial_index = 0;
    float score = 0.0f;
};

struct common_flydelta_intervention_region_trial;

void common_flydelta_coefficient_trial_set_observation(
    common_flydelta_coefficient_trial & target,
    const common_flydelta_counterfactual_trial & baseline,
    const common_flydelta_counterfactual_trial & candidate);

bool common_flydelta_coefficient_trial_to_region(
    const common_flydelta_coefficient_trial & source,
    const common_flydelta_low_rank_basis & basis,
    common_flydelta_intervention_region_trial & target,
    std::string & error);

// Generates a no-op plus a bounded +/- coordinate stencil. The caller may
// use the margins to choose which proposals deserve full generation. This
// helper never makes a host verdict and remains the default strategy.
bool common_flydelta_propose_low_rank_coefficients(
        const common_flydelta_coefficient_search_config & config,
        size_t rank,
        std::vector<std::vector<float>> & proposals,
        std::string & error);

// Basis-aware proposals preserve the no-op and the shared candidate budget.
bool common_flydelta_propose_calibrated_coefficients(
    const common_flydelta_experiment_fixture & fixture,
    const common_flydelta_low_rank_basis & basis,
    const common_flydelta_coefficient_search_config & config,
    std::vector<std::vector<float>> & proposals, std::string & error,
    std::vector<float> * requested_fractions = nullptr);

// Deterministic rank-two controls used before a coefficient optimizer. Every
// non-zero arm has equal L2 budget: [1,0], [0,1], [1,1]/sqrt(2), and optionally
// [1,-1]/sqrt(2), all scaled by config.step. The runner owns the no-op arm.
bool common_flydelta_propose_shallow_rank_two_controls(
        const common_flydelta_coefficient_search_config & config,
        bool include_opposite_control,
        std::vector<std::vector<float>> & proposals,
        std::string & error);

using common_flydelta_coefficient_search_runner = std::function<bool(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const std::vector<float> & coefficients,
        bool apply_overlay,
        common_flydelta_counterfactual_trial & trial,
        common_flydelta_decision_margin & margin,
        common_flydelta_representation_diagnostics & geometry,
        bool & geometry_available,
        std::string & error)>;

using common_flydelta_coefficient_search_batch_runner = std::function<bool(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const std::vector<std::vector<float>> & coefficients,
        std::vector<common_flydelta_counterfactual_trial> & trials,
        std::vector<common_flydelta_decision_margin> & margins,
        std::vector<common_flydelta_representation_diagnostics> & geometries,
        std::vector<bool> & geometry_available,
        std::string & error)>;

// Runs the cheap bounded coefficient stencil. Decision margin is diagnostic
// and may rank follow-up work, while only host-verified counterfactual
// classification can select a coefficient trial.
bool common_flydelta_run_low_rank_coefficient_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error);

// Batch-aware rank-N search. Coordinate arms and one TFO population are
// submitted as one bounded batch; dose regulation, fitness, selection and
// iteration transitions remain scalar control-plane decisions.
bool common_flydelta_run_low_rank_coefficient_search_batched(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & baseline_runner,
        const common_flydelta_coefficient_search_batch_runner & batch_runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error,
        const common_flydelta_outcome_guidance_adviser & adviser = {});

// Executes the unchanged coefficient proposal/ranking algorithm in two
// execution phases: all proposals use diagnostics first, then only the
// diagnostic frontier top-K uses generation and host verification. The
// returned trial list retains both evidence depths; selection is based only
// on the full-generation trials.
bool common_flydelta_run_low_rank_coefficient_search_batched_staged(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & diagnostic_runner,
        const common_flydelta_coefficient_search_batch_runner & diagnostic_batch_runner,
        const common_flydelta_coefficient_search_runner & full_generation_runner,
        const common_flydelta_coefficient_search_batch_runner & full_generation_batch_runner,
        size_t full_generation_top_k,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error,
        const common_flydelta_outcome_guidance_adviser & adviser = {});

// Records all executed coefficient arms as experimental lifecycle entries.
// The helper is deliberately reference-only: it never updates DeltaMemory,
// changes the active registry or turns a diagnostic margin into HELPED.
bool common_flydelta_append_coefficient_search_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const std::vector<common_flydelta_coefficient_trial> & trials,
        const std::string & experimental_artifact_id,
        std::string & error);
