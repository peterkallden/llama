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
    size_t current_layer_count = 0;
    if (current.direction_index < result.directions.size()) {
        const auto & selected_direction = result.directions[current.direction_index];
        if (current.intervention_region) {
            if (current.region_trial_index < selected_direction.region_trials.size()) {
                current_layer_count = selected_direction.region_trials[
                    current.region_trial_index].candidate.layer_indices.size();
            }
        } else if (current.layer_result_index < selected_direction.layer_results.size()) {
            current_layer_count = selected_direction.layer_results[
                current.layer_result_index].candidate.layer_indices.size();
        }
    }
    if (layer_count != current_layer_count) {
        return layer_count < current_layer_count;
    }
    return scale < current.scale;
}

std::string arm_suffix(size_t direction_index, size_t layer_index, size_t scale_index) {
    return "/d" + std::to_string(direction_index) + "/l" +
        std::to_string(layer_index) + "/s" + std::to_string(scale_index);
}

std::string region_arm_suffix(size_t direction_index, size_t trial_index) {
    return "/d" + std::to_string(direction_index) + "/r" +
        std::to_string(trial_index);
}

std::vector<float> region_scales(const common_flydelta_scale_search_config & config) {
    std::vector<float> scales;
    float scale = config.initial_scale;
    for (size_t index = 0; index < config.max_geometric_trials &&
            scale <= config.max_scale + 0.000001f; ++index) {
        scales.push_back(scale);
        const float next = scale * config.growth_factor;
        if (next <= scale + std::numeric_limits<float>::epsilon()) break;
        scale = next;
    }
    return scales;
}

common_flydelta_layer_candidate to_layer_candidate(
        const common_flydelta_intervention_region_candidate & candidate) {
    common_flydelta_layer_candidate layer;
    layer.layer_indices = candidate.layer_indices;
    layer.anchor_layer_index = candidate.anchor_layer_index;
    layer.total_scale = candidate.total_scale;
    layer.per_layer_scale = candidate.per_layer_scale;
    layer.source = candidate.source;
    return layer;
}

} // namespace

bool common_flydelta_search_pipeline_config_validate(
        const common_flydelta_search_pipeline_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || !valid_dimension(config.dimension) ||
            config.max_directions == 0 || config.max_directions > 32 ||
            !common_flydelta_layer_search_config_validate(config.layer, error) ||
            !common_flydelta_scale_search_config_validate(config.scale, error) ||
            config.region_max_singleton_layers == 0 ||
            config.region_max_singleton_layers > 16 ||
            config.region_max_neighborhoods > 16 || config.region_max_trials == 0 ||
            config.region_max_trials > 64 || config.region_max_stalled_scales == 0 ||
            config.region_max_stalled_scales > config.scale.max_geometric_trials) {
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

        if (config.use_intervention_region_search) {
            if (input.available_layers.size() > 16) {
                error = "FlyDelta intervention region pipeline layer count is invalid";
                return false;
            }
            common_flydelta_intervention_region_search_config region_config;
            region_config.available_layers = input.available_layers;
            region_config.scales = region_scales(config.scale);
            region_config.max_singleton_layers = std::min(
                config.region_max_singleton_layers, input.available_layers.size());
            region_config.max_neighborhoods = config.region_max_neighborhoods;
            region_config.max_trials = config.region_max_trials;
            region_config.max_stalled_scales = std::min(
                config.region_max_stalled_scales, region_config.scales.size());
            region_config.min_cosine = config.scale.min_cosine;
            region_config.max_leakage = config.scale.max_leakage;
            region_config.max_shift_norm = config.scale.max_shift_norm;

            common_flydelta_intervention_region_search_runner region_runner =
                [&](const common_flydelta_experiment_fixture & current_fixture,
                    const common_flydelta_intervention_region_candidate * region,
                    bool apply_overlay,
                    common_flydelta_counterfactual_trial & trial,
                    common_flydelta_decision_margin & margin,
                    common_flydelta_representation_diagnostics & diagnostics,
                    bool & diagnostics_available,
                    std::string & runner_error) {
                    common_flydelta_layer_candidate layer;
                    const common_flydelta_layer_candidate * layer_ptr = nullptr;
                    float scale = 0.0f;
                    if (region != nullptr && apply_overlay) {
                        layer = to_layer_candidate(*region);
                        layer_ptr = &layer;
                        scale = region->total_scale;
                    }
                    common_flydelta_scale_geometry geometry;
                    const bool ok = runner(current_fixture, input.direction, layer_ptr,
                        scale, apply_overlay, trial, geometry, runner_error);
                    margin = {};
                    diagnostics = {};
                    diagnostics_available = ok && geometry.available;
                    if (diagnostics_available) {
                        diagnostics = {
                            1,
                            region == nullptr ? 0 : region->anchor_layer_index,
                            geometry.cosine,
                            geometry.progress,
                            geometry.leakage,
                            geometry.shift_norm,
                        };
                    }
                    return ok;
                };

            if (!common_flydelta_run_intervention_region_search(
                    fixture, region_config, region_runner,
                    direction_result.region_trials, direction_result.region_selection, error)) {
                return false;
            }
            if (direction_result.region_selection.selected) {
                const auto & trial = direction_result.region_trials[
                    direction_result.region_selection.trial_index];
                const float score = direction_result.region_selection.score;
                if (better_selection(score, trial.candidate.layer_indices.size(),
                        trial.candidate.total_scale, result.selection, result)) {
                    result.selection.selected = true;
                    result.selection.intervention_region = true;
                    result.selection.direction_index = result.directions.size();
                    result.selection.region_trial_index =
                        direction_result.region_selection.trial_index;
                    result.selection.layer_result_index = 0;
                    result.selection.scale = trial.candidate.total_scale;
                    result.selection.score = score;
                }
            }
            result.directions.push_back(std::move(direction_result));
            continue;
        }

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
                result.selection.intervention_region = false;
                result.selection.direction_index = result.directions.size();
                result.selection.layer_result_index = layer_index;
                result.selection.region_trial_index = 0;
                result.selection.scale = layer_result.scale_selection.scale;
                result.selection.score = score;
            }
        }
        result.directions.push_back(std::move(direction_result));
    }
    return true;
}

bool common_flydelta_append_search_pipeline_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_search_pipeline_result & result,
        const std::string & experimental_artifact_id,
        std::string & error) {
    error.clear();
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            experimental_artifact_id.empty() || experimental_artifact_id.size() > 512) {
        if (error.empty()) error = "FlyDelta pipeline lifecycle artifact identity is invalid";
        return false;
    }

    for (size_t direction_index = 0; direction_index < result.directions.size(); ++direction_index) {
        const auto & direction = result.directions[direction_index];
        for (size_t region_index = 0; region_index < direction.region_trials.size(); ++region_index) {
            const auto & region = direction.region_trials[region_index];
            if (!common_flydelta_intervention_region_trial_validate(region, error)) return false;
            if (!region.executed) continue;

            const std::string suffix = region_arm_suffix(direction_index, region_index);
            common_flydelta_search_observation observation;
            observation.experiment_id = fixture.id;
            observation.candidate_id = experimental_artifact_id + suffix;
            observation.search_kind = "direction-layer-scale-region";
            observation.experimental_artifact_id = experimental_artifact_id;
            observation.outcome = region.outcome;
            observation.host_verified = region.verifier_known;
            observation.diagnostics_available = region.geometry_available;
            if (region.geometry_available) observation.diagnostics = region.geometry;
            observation.sequence_margin_available = region.margin.available;
            observation.sequence_margin_delta = region.margin.normalized_delta();
            observation.budget_remaining = region.safe_to_continue;

            common_flydelta_search_decision decision;
            if (!common_flydelta_decide_search_disposition(
                    observation, decision, error)) return false;

            common_flydelta_candidate_lineage lineage;
            lineage.candidate_id = observation.candidate_id;
            lineage.parent_candidate_id = experimental_artifact_id;
            lineage.mutation_kind = "region_search_arm";
            lineage.generation = 0;
            lineage.direction_id = experimental_artifact_id + "/direction-" +
                std::to_string(direction_index);
            lineage.layer_indices = region.candidate.layer_indices;
            lineage.scale = region.candidate.total_scale;
            lineage.intervention_budget = region.candidate.total_scale;

            common_flydelta_lifecycle_event_context arm_context = context;
            arm_context.event_id = context.event_id + suffix;
            arm_context.idempotency_key = context.idempotency_key + suffix;
            arm_context.source_id = context.source_id + suffix;
            if (!common_flydelta_append_search_lifecycle(
                    store, arm_context, observation, decision, &lineage, error)) {
                return false;
            }
        }
        for (size_t layer_index = 0; layer_index < direction.layer_results.size(); ++layer_index) {
            const auto & layer = direction.layer_results[layer_index];
            for (size_t scale_index = 0; scale_index < layer.scale_trials.size(); ++scale_index) {
                const auto & scale = layer.scale_trials[scale_index];
                if (!common_flydelta_scale_trial_validate(scale, error)) return false;
                if (!scale.executed) continue;

                const std::string suffix = arm_suffix(direction_index, layer_index, scale_index);
                common_flydelta_search_observation observation;
                observation.experiment_id = fixture.id;
                observation.candidate_id = experimental_artifact_id + suffix;
                observation.search_kind = "direction-layer-scale";
                observation.experimental_artifact_id = experimental_artifact_id;
                observation.outcome = scale.outcome;
                // This flag means that the host ran/classified the arm. The
                // outcome may still be UNKNOWN when the verifier had no fact.
                observation.host_verified = true;
                observation.diagnostics_available = scale.geometry_available;
                if (scale.geometry_available) {
                    observation.diagnostics = {
                        1,
                        layer.candidate.anchor_layer_index,
                        scale.geometry.cosine,
                        scale.geometry.progress,
                        scale.geometry.leakage,
                        scale.geometry.shift_norm,
                    };
                }
                observation.budget_remaining = scale.safe_to_escalate;

                common_flydelta_search_decision decision;
                if (!common_flydelta_decide_search_disposition(
                        observation, decision, error)) return false;

                common_flydelta_candidate_lineage lineage;
                lineage.candidate_id = observation.candidate_id;
                lineage.parent_candidate_id = experimental_artifact_id;
                lineage.mutation_kind = scale.refinement ? "scale_refinement" : "search_arm";
                lineage.generation = scale.refinement ? 1 : 0;
                lineage.direction_id = experimental_artifact_id + "/direction-" +
                    std::to_string(direction_index);
                lineage.layer_indices = layer.candidate.layer_indices;
                lineage.scale = scale.scale;
                lineage.intervention_budget = scale.scale;

                common_flydelta_lifecycle_event_context arm_context = context;
                arm_context.event_id = context.event_id + suffix;
                arm_context.idempotency_key = context.idempotency_key + suffix;
                arm_context.source_id = context.source_id + suffix;
                if (!common_flydelta_append_search_lifecycle(
                        store, arm_context, observation, decision, &lineage, error)) {
                    return false;
                }
            }
        }
    }
    return true;
}
