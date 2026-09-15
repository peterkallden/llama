#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool finite(float value) {
    return std::isfinite(value);
}

float norm(const std::vector<float> & values) {
    float sum = 0.0f;
    for (const float value : values) sum += value * value;
    return std::sqrt(sum);
}

float dot(const std::vector<float> & left, const std::vector<float> & right) {
    float value = 0.0f;
    for (size_t i = 0; i < left.size(); ++i) value += left[i] * right[i];
    return value;
}

bool valid_outcome(common_flydelta_counterfactual_outcome outcome) {
    switch (outcome) {
        case common_flydelta_counterfactual_outcome::unknown:
        case common_flydelta_counterfactual_outcome::helped:
        case common_flydelta_counterfactual_outcome::neutral:
        case common_flydelta_counterfactual_outcome::harmed:
            return true;
    }
    return false;
}

} // namespace

bool common_flydelta_low_rank_basis_validate(
        const common_flydelta_low_rank_basis & basis,
        size_t max_rank,
        std::string & error) {
    error.clear();
    if (basis.schema_version != 1 || basis.dimension == 0 ||
            basis.dimension > (1U << 20) || basis.layer_index < 0 ||
            basis.vectors.empty() || basis.vectors.size() > max_rank) {
        error = "FlyDelta low-rank basis identity or bounds are invalid";
        return false;
    }
    for (const auto & vector : basis.vectors) {
        if (vector.size() != basis.dimension || norm(vector) <= std::numeric_limits<float>::epsilon()) {
            error = "FlyDelta low-rank basis vector is invalid";
            return false;
        }
        for (const float value : vector) {
            if (!finite(value)) {
                error = "FlyDelta low-rank basis contains a non-finite value";
                return false;
            }
        }
    }
    return true;
}

bool common_flydelta_build_low_rank_basis(
        size_t dimension,
        size_t max_rank,
        const std::vector<common_flydelta_direction_candidate> & candidates,
        common_flydelta_low_rank_basis & basis,
        std::string & error) {
    error.clear();
    basis = {};
    if (dimension == 0 || dimension > (1U << 20) || max_rank == 0 || max_rank > 16 ||
            candidates.empty() || candidates.size() > 256) {
        error = "FlyDelta low-rank basis input bounds are invalid";
        return false;
    }
    basis.dimension = dimension;
    basis.layer_index = candidates.front().layer_index;
    for (const auto & candidate : candidates) {
        if (!common_flydelta_direction_candidate_validate(candidate, dimension, error) ||
                candidate.layer_index != basis.layer_index) {
            if (error.empty()) error = "FlyDelta low-rank candidates are incompatible";
            return false;
        }
        if (basis.vectors.size() == max_rank) break;
        std::vector<float> vector = candidate.values;
        for (const auto & existing : basis.vectors) {
            const float projection = dot(vector, existing);
            for (size_t i = 0; i < vector.size(); ++i) vector[i] -= projection * existing[i];
        }
        const float vector_norm = norm(vector);
        if (!finite(vector_norm)) {
            error = "FlyDelta low-rank basis normalization failed";
            return false;
        }
        if (vector_norm <= 0.000001f) continue;
        for (float & value : vector) value /= vector_norm;
        basis.vectors.push_back(std::move(vector));
    }
    if (basis.vectors.empty()) {
        error = "FlyDelta low-rank candidates do not span a nonzero basis";
        return false;
    }
    return common_flydelta_low_rank_basis_validate(basis, max_rank, error);
}

float common_flydelta_decision_margin::normalized_delta() const {
    if (positive_token_count == 0 || negative_token_count == 0) return 0.0f;
    return positive_total_logprob / static_cast<float>(positive_token_count) -
        negative_total_logprob / static_cast<float>(negative_token_count);
}

bool common_flydelta_decision_margin_validate(
        const common_flydelta_decision_margin & margin,
        std::string & error) {
    error.clear();
    if (!margin.available) return true;
    if (!finite(margin.positive_total_logprob) || !finite(margin.negative_total_logprob) ||
            margin.positive_token_count == 0 || margin.negative_token_count == 0 ||
            !finite(margin.total_delta()) || !finite(margin.normalized_delta())) {
        error = "FlyDelta decision margin is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_coefficient_search_config_validate(
        const common_flydelta_coefficient_search_config & config,
        size_t rank,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || rank == 0 || rank > 16 || !finite(config.step) ||
            config.step <= 0.0f || config.step > 1.0f || config.max_candidates == 0 ||
            config.max_candidates > 64 || !finite(config.max_l2_norm) ||
            config.max_l2_norm <= 0.0f || config.max_l2_norm > 1.0f) {
        error = "FlyDelta coefficient search configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_propose_low_rank_coefficients(
        const common_flydelta_coefficient_search_config & config,
        size_t rank,
        std::vector<std::vector<float>> & proposals,
        std::string & error) {
    error.clear();
    proposals.clear();
    if (!common_flydelta_coefficient_search_config_validate(config, rank, error)) return false;
    proposals.push_back(std::vector<float>(rank, 0.0f));
    for (size_t index = 0; index < rank && proposals.size() + 1 < config.max_candidates; ++index) {
        std::vector<float> positive(rank, 0.0f);
        positive[index] = config.step;
        proposals.push_back(positive);
        if (proposals.size() >= config.max_candidates) break;
        positive[index] = -config.step;
        proposals.push_back(std::move(positive));
    }
    return true;
}

bool common_flydelta_run_low_rank_coefficient_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error) {
    error.clear();
    trials.clear();
    selection = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_low_rank_basis_validate(basis, 16, error) ||
            !common_flydelta_coefficient_search_config_validate(
                config, basis.vectors.size(), error) || !runner) {
        if (error.empty()) error = "FlyDelta coefficient search input is invalid";
        return false;
    }
    std::vector<std::vector<float>> proposals;
    if (!common_flydelta_propose_low_rank_coefficients(
            config, basis.vectors.size(), proposals, error)) return false;
    common_flydelta_counterfactual_trial baseline;
    common_flydelta_decision_margin baseline_margin;
    if (!runner(fixture, basis, proposals.front(), false, baseline, baseline_margin, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error)) return false;
    for (size_t index = 1; index < proposals.size(); ++index) {
        const auto & coefficients = proposals[index];
        if (norm(coefficients) > config.max_l2_norm) continue;
        common_flydelta_counterfactual_trial candidate;
        common_flydelta_decision_margin margin;
        if (!runner(fixture, basis, coefficients, true, candidate, margin, error) ||
                !common_flydelta_counterfactual_trial_validate(candidate, error) ||
                !common_flydelta_decision_margin_validate(margin, error)) return false;
        common_flydelta_coefficient_trial trial;
        trial.coefficients = coefficients;
        trial.margin = margin;
        trial.outcome = common_flydelta_classify_counterfactual(baseline, candidate);
        trial.quality_delta = candidate.quality - baseline.quality;
        trial.executed = candidate.executed;
        trial.verifier_known = baseline.verifier_known && candidate.verifier_known;
        trials.push_back(std::move(trial));
    }
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (!valid_outcome(trial.outcome) || trial.outcome != common_flydelta_counterfactual_outcome::helped ||
                !trial.executed || !trial.verifier_known) continue;
        if (!selection.selected || trial.quality_delta > selection.score ||
                (trial.quality_delta == selection.score &&
                 norm(trial.coefficients) < norm(trials[selection.trial_index].coefficients))) {
            selection.selected = true;
            selection.trial_index = index;
            selection.score = trial.quality_delta;
        }
    }
    return true;
}
