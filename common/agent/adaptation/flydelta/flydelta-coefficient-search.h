#pragma once

#include "agent/adaptation/flydelta/flydelta-direction-search.h"
#include "agent/adaptation/flydelta/flydelta-experiment.h"

#include <cstddef>
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

struct common_flydelta_coefficient_search_config {
    int schema_version = 1;
    float step = 0.05f;
    size_t max_candidates = 16;
    float max_l2_norm = 0.32f;
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
    bool executed = false;
    bool verifier_known = false;
};

struct common_flydelta_coefficient_selection {
    bool selected = false;
    size_t trial_index = 0;
    float score = 0.0f;
};

// Generates a no-op plus a bounded +/- coordinate stencil. The caller may
// use the margins to choose which proposals deserve full generation. This
// helper never makes a host verdict.
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
