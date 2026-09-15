#include "agent/adaptation/flydelta/flydelta-search-pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool valid_dimension(size_t value) {
    return value > 0 && value <= (1U << 20);
}

bool better_selection(float score, size_t layer_count, float scale,
        const common_flydelta_search_pipeline_selection & current,
        const common_flydelta_search_pipeline_result & result) {
    if (!current.selected || score != current.score) return !current.selected || score > current.score;
    const auto & selected_direction = result.directions[current.direction_index];
    const auto & selected_layer = selected_direction.layer_results[current.layer_result_index];
    if (layer_count != selected_layer.candidate.layer_indices.size()) {
        return layer_count < selected_layer.candidate.layer_indices.size();
    }
    return scale < current.scale;
}

} // namespace

bool common_flydelta_search_pipeline_config_validate(
        const common_flydelta_search_pipeline_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || !valid_dimension(config.dimension) ||
            config.max_directions == 0 || config.max_directions > 32 ||
            !common_flydelta_layer_search_config_validate(config.layer, error) ||
            !common_flydelta_scale_search_config_validate(config.scale, error)) {
        if (error.empty()) error = "FlyDelta search pipeline configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_search_pipeline_direction_validate(
        const common_flydelta_search_pipeline_direction & direction,
        const common_flydelta_search_pipeline_config & config,
        std::string & error) {
    error.clear();
    if (!common_flydelta_direction_candidate_validate(
            direction.direction, config.dimension, error) ||
            direction.available_layers.empty() ||
            !std::is_sorted(direction.available_layers.begin(), direction.available_layers.end()) ||
            std::adjacent_find(direction.available_layers.begin(), direction.available_layers.end()) !=
                direction.available_layers.end() || direction.available_layers.front() == 0) {
        if (error.empty()) error = "FlyDelta search pipeline direction is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_run_search_pipeline(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_search_pipeline_config & config,
        const std::vector<common_flydelta_search_pipeline_direction> & directions,
        const common_flydelta_search_pipeline_runner & runner,
        common_flydelta_search_pipeline_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_search_pipeline_config_validate(config, error) ||
            directions.empty() || directions.size() > config.max_directions || !runner) {
        if (error.empty()) error = "FlyDelta search pipeline input is invalid";
        return false;
    }

    for (const auto & input : directions) {
        if (!common_flydelta_search_pipeline_direction_validate(input, config, error)) return false;
        common_flydelta_search_pipeline_direction_result direction_result;
        direction_result.direction = input.direction;
        if (!common_flydelta_build_layer_search_plan(
                input.layer_diagnostics, input.available_layers, config.layer,
                direction_result.layer_plan, error)) {
            return false;
        }
        if (direction_result.layer_plan.singleton_candidates.empty()) {
            result.directions.push_back(std::move(direction_result));
            continue;
        }

        common_flydelta_layer_search_runner layer_runner =
            [&](const common_flydelta_experiment_fixture & current_fixture,
                const common_flydelta_layer_candidate * layer,
                bool apply_overlay,
                common_flydelta_counterfactual_trial & representative,
                std::string & runner_error) {
                common_flydelta_scale_geometry representative_geometry;
                if (layer == nullptr || !apply_overlay) {
                    return runner(current_fixture, input.direction, nullptr, 0.0f, false,
                        representative, representative_geometry, runner_error);
                }

                common_flydelta_search_pipeline_layer_result nested;
                nested.candidate = *layer;
                std::vector<common_flydelta_counterfactual_trial> counterfactuals;
                common_flydelta_scale_search_runner scale_runner =
                    [&](const common_flydelta_experiment_fixture & scale_fixture,
                        float scale,
                        bool scale_apply_overlay,
                        common_flydelta_counterfactual_trial & trial,
                        common_flydelta_scale_geometry & geometry,
                        std::string & scale_error) {
                        const bool ok = runner(scale_fixture, input.direction, layer, scale,
                            scale_apply_overlay, trial, geometry, scale_error);
                        if (ok && scale_apply_overlay) counterfactuals.push_back(trial);
                        return ok;
                    };
                if (!common_flydelta_run_scale_search(
                        current_fixture, config.scale, scale_runner,
                        nested.scale_trials, nested.scale_selection, runner_error)) {
                    return false;
                }
                if (counterfactuals.empty()) {
                    runner_error = "FlyDelta scale search produced no candidate trial";
                    return false;
                }
                const size_t selected_index = nested.scale_selection.selected
                    ? nested.scale_selection.trial_index
                    : nested.scale_trials.size() - 1;
                if (selected_index >= counterfactuals.size()) {
                    runner_error = "FlyDelta scale search trial mapping is invalid";
                    return false;
                }
                representative = counterfactuals[selected_index];
                nested.representative_trial = representative;
                direction_result.layer_results.push_back(std::move(nested));
                return true;
            };

        if (!common_flydelta_run_layer_search(
                fixture, direction_result.layer_plan, layer_runner,
                direction_result.layer_trials, direction_result.layer_selection, error)) {
            return false;
        }
        for (size_t layer_index = 0;
                layer_index < direction_result.layer_results.size(); ++layer_index) {
            const auto & layer_result = direction_result.layer_results[layer_index];
            if (!layer_result.scale_selection.selected) continue;
            const float score = layer_result.scale_selection.score;
            if (better_selection(score, layer_result.candidate.layer_indices.size(),
                    layer_result.scale_selection.scale, result.selection, result)) {
                result.selection.selected = true;
                result.selection.direction_index = result.directions.size();
                result.selection.layer_result_index = layer_index;
                result.selection.scale = layer_result.scale_selection.scale;
                result.selection.score = score;
            }
        }
        result.directions.push_back(std::move(direction_result));
    }
    return true;
}
