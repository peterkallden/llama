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
        const common_flydelta_coefficient_search_runner & runner,
        common_flydelta_deep_search_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_deep_search_config_validate(config, error) || !runner) {
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
    return common_flydelta_run_low_rank_coefficient_search(
        fixture, result.basis, config.coefficients, runner,
        result.coefficient_trials, result.coefficient_selection, error);
}
