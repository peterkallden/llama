#include "agent/adaptation/flydelta/flydelta-deep-search.h"

#include <algorithm>
#include <cmath>

namespace {

bool finite(float value) {
    return std::isfinite(value);
}

float direction_score(const common_flydelta_deep_search_direction & value) {
    return value.decision_score_available
        ? value.decision_score : value.direction.median_alignment;
}

} // namespace

bool common_flydelta_deep_search_config_validate(
        const common_flydelta_deep_search_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.max_rank == 0 || config.max_rank > 4 ||
            config.max_directions == 0 || config.max_directions > 16 ||
            config.full_generation_top_k == 0 || config.full_generation_top_k > 16 ||
            config.max_rank > config.max_directions ||
            !common_flydelta_coefficient_search_config_validate(
                config.coefficients, config.max_rank, error)) {
        if (error.empty()) error = "FlyDelta deep search configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_select_deep_search_directions(
        const common_flydelta_deep_search_config & config,
        const std::vector<common_flydelta_deep_search_direction> & directions,
        std::vector<common_flydelta_deep_search_direction> & selected,
        std::string & error) {
    error.clear();
    selected.clear();
    if (!common_flydelta_deep_search_config_validate(config, error) ||
            directions.empty() || directions.size() > config.max_directions) {
        if (error.empty()) error = "FlyDelta deep search direction input is invalid";
        return false;
    }
    const int32_t layer = directions.front().direction.layer_index;
    const size_t dimension = directions.front().direction.values.size();
    for (const auto & value : directions) {
        if (!common_flydelta_direction_candidate_validate(
                value.direction, dimension, error) ||
                value.direction.layer_index != layer ||
                (value.decision_score_available && !finite(value.decision_score))) {
            if (error.empty()) error = "FlyDelta deep search directions are incompatible";
            return false;
        }
    }
    selected = directions;
    std::stable_sort(selected.begin(), selected.end(),
        [](const auto & left, const auto & right) {
            const float left_score = direction_score(left);
            const float right_score = direction_score(right);
            if (left_score != right_score) return left_score > right_score;
            return left.direction.kind < right.direction.kind;
        });
    if (selected.size() > config.max_rank) selected.resize(config.max_rank);
    return true;
}

bool common_flydelta_run_deep_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_deep_search_config & config,
        const std::vector<common_flydelta_deep_search_direction> & directions,
        const common_flydelta_coefficient_search_runner & diagnostic_runner,
        const common_flydelta_coefficient_search_runner & full_generation_runner,
        common_flydelta_deep_search_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_deep_search_config_validate(config, error) ||
            !diagnostic_runner || !full_generation_runner) {
        if (error.empty()) error = "FlyDelta deep search input is invalid";
        return false;
    }
    if (!common_flydelta_select_deep_search_directions(
            config, directions, result.selected_directions, error)) return false;
    std::vector<common_flydelta_direction_candidate> candidates;
    candidates.reserve(result.selected_directions.size());
    for (const auto & value : result.selected_directions) candidates.push_back(value.direction);
    if (!common_flydelta_build_low_rank_basis(
            candidates.front().values.size(), config.max_rank, candidates,
            result.basis, error)) return false;
    result.layer_index = result.basis.layer_index;
    std::vector<common_flydelta_coefficient_trial> diagnostic_trials;
    common_flydelta_coefficient_selection diagnostic_selection;
    if (!common_flydelta_run_low_rank_coefficient_search(
            fixture, result.basis, config.coefficients, diagnostic_runner,
            diagnostic_trials, diagnostic_selection, error)) return false;
    result.coefficient_trials = diagnostic_trials;

    common_flydelta_counterfactual_trial baseline;
    common_flydelta_decision_margin baseline_margin;
    common_flydelta_representation_diagnostics baseline_geometry;
    bool baseline_geometry_available = false;
    if (!full_generation_runner(
            fixture, result.basis, std::vector<float>(result.basis.vectors.size(), 0.0f),
            false, baseline, baseline_margin, baseline_geometry,
            baseline_geometry_available, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error)) return false;

    std::vector<size_t> ranking;
    for (size_t index = 0; index < diagnostic_trials.size(); ++index) {
        if (diagnostic_trials[index].executed) ranking.push_back(index);
    }
    std::stable_sort(ranking.begin(), ranking.end(), [&](size_t left, size_t right) {
        if (diagnostic_trials[left].search_fitness != diagnostic_trials[right].search_fitness) {
            return diagnostic_trials[left].search_fitness > diagnostic_trials[right].search_fitness;
        }
        return diagnostic_trials[left].coefficients.size() <
            diagnostic_trials[right].coefficients.size();
    });
    const size_t top_k = std::min(config.full_generation_top_k, ranking.size());
    for (size_t rank = 0; rank < top_k; ++rank) {
        const size_t diagnostic_index = ranking[rank];
        const auto & diagnostic = diagnostic_trials[diagnostic_index];
        common_flydelta_counterfactual_trial candidate;
        common_flydelta_decision_margin margin;
        common_flydelta_representation_diagnostics geometry;
        bool geometry_available = false;
        if (!full_generation_runner(
                fixture, result.basis, diagnostic.coefficients, true, candidate, margin,
                geometry, geometry_available, error) ||
                !common_flydelta_counterfactual_trial_validate(candidate, error) ||
                !common_flydelta_decision_margin_validate(margin, error)) return false;
        common_flydelta_coefficient_trial trial;
        trial.coefficients = diagnostic.coefficients;
        trial.margin = margin;
        trial.outcome = common_flydelta_classify_counterfactual(baseline, candidate);
        trial.quality_delta = candidate.quality - baseline.quality;
        trial.sequence_margin_delta = margin.available && baseline_margin.available
            ? margin.normalized_delta() - baseline_margin.normalized_delta() : 0.0f;
        trial.executed = candidate.executed;
        trial.verifier_known = baseline.verifier_known && candidate.verifier_known;
        trial.geometry_available = geometry_available;
        if (geometry_available &&
                !common_flydelta_representation_diagnostics_validate(geometry, error)) {
            return false;
        }
        trial.geometry = geometry;
        trial.iteration = diagnostic.iteration;
        trial.parent_trial_index = diagnostic_index;
        trial.mutation_kind = "full_generation_top_arm";
        trial.search_fitness = diagnostic.search_fitness;
        const size_t result_index = result.coefficient_trials.size();
        result.coefficient_trials.push_back(std::move(trial));
        const auto & full_trial = result.coefficient_trials.back();
        if (full_trial.outcome == common_flydelta_counterfactual_outcome::helped &&
                full_trial.executed && full_trial.verifier_known &&
                (!result.coefficient_selection.selected ||
                 full_trial.quality_delta > result.coefficient_selection.score)) {
            result.coefficient_selection.selected = true;
            result.coefficient_selection.trial_index = result_index;
            result.coefficient_selection.score = full_trial.quality_delta;
        }
    }
    return true;
}

bool common_flydelta_append_deep_search_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_deep_search_config & config,
        const common_flydelta_deep_search_result & result,
        const std::string & experimental_artifact_id,
        std::string & error) {
    error.clear();
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_deep_search_config_validate(config, error) ||
            result.schema_version != 1 || result.layer_index < 0 ||
            result.selected_directions.empty() ||
            result.selected_directions.size() > config.max_rank ||
            experimental_artifact_id.empty() || experimental_artifact_id.size() > 512) {
        if (error.empty()) error = "FlyDelta deep search lifecycle input is invalid";
        return false;
    }
    if (!common_flydelta_low_rank_basis_validate(result.basis, config.max_rank, error) ||
            result.basis.layer_index != result.layer_index ||
            result.basis.vectors.size() != result.selected_directions.size()) {
        if (error.empty()) error = "FlyDelta deep search lifecycle basis is invalid";
        return false;
    }
    return common_flydelta_append_coefficient_search_lifecycle(
        store, context, fixture, result.basis, config.coefficients,
        result.coefficient_trials, experimental_artifact_id, error);
}
