#pragma once

#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"

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

struct common_flydelta_decision_margin {
    bool available = false;
    float positive_total_logprob = 0.0f;
    float negative_total_logprob = 0.0f;
    size_t positive_token_count = 0;
    size_t negative_token_count = 0;

    float total_delta() const {
        return positive_total_logprob - negative_total_logprob;
    }
    float normalized_delta() const;
};

bool common_flydelta_decision_margin_validate(
        const common_flydelta_decision_margin & margin,
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
    float max_l2_norm = 0.32f;
    // TFO-lite only. These limits are intentionally small so the strategy
    // remains a bounded experiment rather than a background optimizer.
    uint64_t seed = 0x464c5944454c5441ULL;
    size_t population_size = 4;
    size_t iterations = 2;
    float exploration_scale = 1.0f;
    float norm_penalty = 0.05f;
};

bool common_flydelta_coefficient_search_config_validate(
        const common_flydelta_coefficient_search_config & config,
        size_t rank,
        std::string & error);

struct common_flydelta_coefficient_trial {
    std::vector<float> coefficients;
    common_flydelta_decision_margin margin;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
    float quality_delta = 0.0f;
    float sequence_margin_delta = 0.0f;
    float search_fitness = 0.0f;
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

using common_flydelta_coefficient_search_runner = std::function<bool(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const std::vector<float> & coefficients,
        bool apply_overlay,
        common_flydelta_counterfactual_trial & trial,
        common_flydelta_decision_margin & margin,
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
