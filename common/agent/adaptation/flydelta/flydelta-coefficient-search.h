#pragma once

#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-decision-margin.h"
#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"
#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"
#include "agent/adaptation/flydelta/flydelta-dose-controller.h"

#include <cstddef>
#include <cstdint>
#include <functional>
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

// Coefficient search remains a host-side experiment strategy. The default
// coordinate stencil is deliberately retained; tfo_lite is an optional,
// bounded population search for mixed coefficients inside the same basis.
enum class common_flydelta_coefficient_search_strategy {
    coordinate,
    tfo_lite,
};

const char * common_flydelta_coefficient_search_strategy_name(
        common_flydelta_coefficient_search_strategy strategy);

struct common_flydelta_coefficient_search_config {
    int schema_version = 1;
    common_flydelta_coefficient_search_strategy strategy =
        common_flydelta_coefficient_search_strategy::coordinate;
    float step = 0.05f;
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
};

bool common_flydelta_coefficient_search_config_validate(
        const common_flydelta_coefficient_search_config & config,
        size_t rank,
        std::string & error);

struct common_flydelta_coefficient_trial {
    std::vector<float> coefficients;
    std::vector<float> requested_coefficients;
    float requested_strength = 0.0f;
    float executed_strength = 0.0f;
    common_flydelta_dose_action dose_action = common_flydelta_dose_action::reject;
    float relative_dose = 0.0f;
    bool dose_evaluated = false;
    bool dose_safety_limited = false;
    std::string dose_reason;
    common_flydelta_decision_margin margin;
    common_flydelta_margin_comparison margin_comparison;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
    float quality_delta = 0.0f;
    float search_fitness = 0.0f;
    bool geometry_available = false;
    common_flydelta_representation_diagnostics geometry;
    bool executed = false;
    bool verifier_known = false;
    size_t iteration = 0;
    size_t parent_trial_index = static_cast<size_t>(-1);
    std::string mutation_kind;
};

struct common_flydelta_coefficient_selection {
    bool selected = false;
    size_t trial_index = 0;
    float score = 0.0f;
};

// Generates a no-op plus a bounded +/- coordinate stencil. The caller may
// use the margins to choose which proposals deserve full generation. This
// helper never makes a host verdict and remains the default strategy.
bool common_flydelta_propose_low_rank_coefficients(
        const common_flydelta_coefficient_search_config & config,
        size_t rank,
        std::vector<std::vector<float>> & proposals,
        std::string & error);

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
        std::string & error);

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
        std::string & error);

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
