#include "agent/adaptation/flydelta/flydelta-evaluator.h"

#include <utility>

namespace {

bool is_representation_augmentation_state_ref(const std::string & value) {
    static constexpr const char * prefix =
        "flydelta://state/representation-augmentation/";
    return value.rfind(prefix, 0) == 0;
}

bool valid_reference_count(size_t count, size_t max_references) {
    return count != 0 && count <= max_references;
}

bool validate_search_pipeline_result(
        const common_flydelta_search_pipeline_result & result,
        const common_flydelta_search_pipeline_config & config,
        std::string & error) {
    if (result.directions.empty() || result.directions.size() > config.max_directions) {
        error = "FlyDelta search pipeline result direction count is invalid";
        return false;
    }
    for (const auto & direction : result.directions) {
        if (!common_flydelta_direction_candidate_validate(
                direction.direction, config.dimension, error)) return false;
        // Region/Whirlpool search deliberately does not populate the
        // classic layer-search plan. Validate the representation that the
        // bounded slice actually produced instead of rejecting a valid
        // region result as an incomplete layer plan.
        if (direction.region_trials.empty()) {
            if (!common_flydelta_layer_search_plan_validate(
                    direction.layer_plan, config.layer, error)) return false;
        }
        for (const auto & trial : direction.layer_trials) {
            if (!common_flydelta_layer_search_trial_validate(trial, error)) return false;
        }
        for (const auto & layer : direction.layer_results) {
            if (!common_flydelta_layer_candidate_validate(layer.candidate, error)) return false;
            if (layer.scale_selection.selected &&
                    layer.scale_selection.trial_index >= layer.scale_trials.size()) {
                error = "FlyDelta search pipeline selected scale trial is out of bounds";
                return false;
            }
            for (const auto & trial : layer.scale_trials) {
                if (!common_flydelta_scale_trial_validate(trial, error)) return false;
            }
        }
        for (const auto & trial : direction.region_trials) {
            if (!common_flydelta_intervention_region_trial_validate(trial, error)) return false;
        }
        if (direction.region_selection.selected &&
                direction.region_selection.trial_index >= direction.region_trials.size()) {
            error = "FlyDelta search pipeline selected region trial is out of bounds";
            return false;
        }
    }
    if (result.selection.selected &&
            (result.selection.direction_index >= result.directions.size() ||
             (result.selection.intervention_region && result.selection.region_trial_index >=
                 result.directions[result.selection.direction_index].region_trials.size()) ||
             (!result.selection.intervention_region && result.selection.layer_result_index >=
                 result.directions[result.selection.direction_index].layer_results.size()))) {
        error = "FlyDelta search pipeline selection is out of bounds";
        return false;
    }
    return true;
}

bool utility_observations_from_pipeline(
        const common_flydelta_search_pipeline_result & pipeline,
        std::vector<common_flydelta_subspace_utility_observation> & observations,
        std::string & error) {
    observations.clear();
    for (const auto & direction : pipeline.directions) {
        for (const auto & trial : direction.region_trials) {
            if (!trial.executed) continue;
            common_flydelta_subspace_utility_observation observation;
            observation.safe_to_continue = trial.safe_to_continue;
            observation.decision_margin_available = trial.margin_comparison.available;
            if (observation.decision_margin_available) {
                observation.decision_margin_delta =
                    trial.margin_comparison.normalized_delta();
            }
            observation.geometry_available = trial.geometry_available;
            observation.geometry = trial.geometry;
            if (!common_flydelta_subspace_utility_observation_validate(observation, error)) {
                return false;
            }
            observations.push_back(std::move(observation));
        }
    }
    if (pipeline.alpha_response_available) {
        common_flydelta_subspace_utility_observation observation;
        observation.alpha_response_available = true;
        observation.alpha_response_status = pipeline.alpha_response.response_status;
        observation.alpha_range_not_exhausted = pipeline.alpha_response.range_not_exhausted;
        observation.safe_to_continue =
            pipeline.alpha_response.response_status !=
                common_flydelta_alpha_response_status::safety_limited;
        observation.decision_margin_available =
            pipeline.alpha_response.best_margin_available;
        observation.decision_margin_delta =
            pipeline.alpha_response.best_margin_delta_normalized;
        if (!common_flydelta_subspace_utility_observation_validate(observation, error)) {
            return false;
        }
        observations.push_back(std::move(observation));
    }
    if (observations.empty()) {
        error = "FlyDelta search slice returned no executed utility observations";
        return false;
    }
    return true;
}

common_flydelta_next_action next_action_from_augmentation(
        const std::string & action) {
    if (action == "stop") return common_flydelta_next_action::stop;
    if (action == "retain") return common_flydelta_next_action::retain;
    if (action == "refine_bootstrap") return common_flydelta_next_action::refine_bootstrap;
    if (action == "recenter_augmented_surface") return common_flydelta_next_action::recenter_surface;
    if (action == "allow_tfo_lite") return common_flydelta_next_action::allow_tfo_lite;
    return common_flydelta_next_action::run_representation_augmentation;
}

} // namespace

bool common_flydelta_evaluate_job(
        const common_flydelta_experiment_job & job,
        const common_flydelta_evaluator_config & config,
        const common_flydelta_evaluator_callbacks & callbacks,
        common_flydelta_evaluator_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (config.max_references == 0 ||
            !common_flydelta_experiment_job_validate(
                job, config.max_references, error)) return false;
    if (static_cast<bool>(callbacks.resolve_search_orchestration_state) !=
            static_cast<bool>(callbacks.persist_search_orchestration_state)) {
        error = "FlyDelta orchestration state callbacks must be supplied together";
        return false;
    }

    switch (job.kind) {
        case common_flydelta_experiment_job_kind::donor_capture: {
            if (!callbacks.run_donor_capture) {
                error = "FlyDelta donor capture evaluator requires a host runner";
                return false;
            }
            if (!callbacks.run_donor_capture(job, result.capture_manifests, error) ||
                    result.capture_manifests.empty()) {
                if (error.empty()) error = "FlyDelta donor capture evaluator returned no manifests";
                return false;
            }
            if (result.capture_manifests.size() != job.capture_candidate_ids.size()) {
                error = "FlyDelta donor capture manifest count does not match candidates";
                return false;
            }
            for (const auto & manifest : result.capture_manifests) {
                if (!common_flydelta_capture_manifest_validate(
                        manifest, config.max_capture_bytes, error)) return false;
                if (manifest.model_profile_fingerprint != job.seed.model_profile_fingerprint ||
                        manifest.execution_context_fingerprint !=
                            job.seed.execution_context_fingerprint ||
                        manifest.behavior_key != job.seed.behavior_key) {
                    error = "FlyDelta donor capture manifest does not match job identity";
                    return false;
                }
            }
            result.processed_references = result.capture_manifests.size();
            return true;
        }
        case common_flydelta_experiment_job_kind::counterfactual: {
            if (!callbacks.run_counterfactual) {
                error = "FlyDelta counterfactual evaluator requires a host runner";
                return false;
            }
            if (!callbacks.run_counterfactual(
                    job, result.counterfactual_reports, error)) return false;
            if (result.counterfactual_reports.empty()) {
                error = "FlyDelta counterfactual evaluator returned no reports";
                return false;
            }
            for (const auto & report : result.counterfactual_reports) {
                if (!common_flydelta_counterfactual_report_validate(report, error)) return false;
                if (report.experiment_id != job.id) {
                    error = "FlyDelta evaluator report belongs to another job";
                    return false;
                }
            }
            result.processed_references = result.counterfactual_reports.size();
            return true;
        }
        case common_flydelta_experiment_job_kind::basis: {
            if (!callbacks.resolve_behavior_delta ||
                    !common_flydelta_basis_config_validate(config.basis, error)) {
                if (error.empty()) error = "FlyDelta basis evaluator requires a resolver and basis config";
                return false;
            }
            if (config.basis.execution_context_fingerprint !=
                    job.seed.execution_context_fingerprint) {
                error = "FlyDelta basis evaluator context fingerprint does not match job";
                return false;
            }
            common_flydelta_basis_builder builder(config.basis);
            for (const auto & id : job.behavior_delta_ids) {
                common_flydelta_behavior_delta delta;
                common_flydelta_intervention_credit credit;
                if (!callbacks.resolve_behavior_delta(id, delta, credit, error) ||
                        !builder.add(delta, credit, error)) return false;
                ++result.processed_references;
            }
            result.basis_directions = builder.directions();
            return true;
        }
        case common_flydelta_experiment_job_kind::direction: {
            if (!callbacks.resolve_behavior_delta ||
                    !common_flydelta_direction_search_config_validate(config.direction, error)) {
                if (error.empty()) error = "FlyDelta direction evaluator requires a resolver and direction config";
                return false;
            }
            if (config.direction.execution_context_fingerprint !=
                    job.seed.execution_context_fingerprint) {
                error = "FlyDelta direction evaluator context fingerprint does not match job";
                return false;
            }
            common_flydelta_aggregation_config aggregation_config;
            aggregation_config.identity = config.direction;
            aggregation_config.depth = config.evidence_depth;
            aggregation_config.max_retained_samples = config.aggregation_max_retained_samples;
            common_flydelta_incremental_aggregation aggregation(aggregation_config);
            std::vector<common_flydelta_contrast_sample> samples;
            samples.reserve(job.behavior_delta_ids.size());
            for (const auto & id : job.behavior_delta_ids) {
                common_flydelta_contrast_sample sample;
                if (!callbacks.resolve_behavior_delta(
                        id, sample.delta, sample.credit, error)) return false;
                if (!aggregation.ingest(sample, error)) return false;
                samples.push_back(std::move(sample));
            }
            if (!aggregation.assess_depth(result.evidence_depth, error)) return false;
            result.search_budget = common_flydelta_search_budget_for_depth(result.evidence_depth.depth);
            if (!common_flydelta_search_budget_validate(result.search_budget, error)) return false;
            if (!common_flydelta_build_direction_candidates(
                    config.direction, aggregation.snapshot().retained_samples,
                    result.direction_candidates, error)) return false;
            result.aggregation = aggregation.snapshot();
            result.processed_references = samples.size();
            return true;
        }
        case common_flydelta_experiment_job_kind::search_pipeline: {
            const bool has_representation_augmentation_state =
                is_representation_augmentation_state_ref(job.search_state_ref);
            if (has_representation_augmentation_state) {
                if (!callbacks.run_representation_augmentation_with_state ||
                        !callbacks.resolve_representation_augmentation_state ||
                        !callbacks.persist_representation_augmentation_state ||
                        !common_flydelta_representation_augmentation_config_validate(
                            config.representation_augmentation, error)) {
                    if (error.empty()) {
                        error = "FlyDelta augmentation continuation requires typed state callbacks";
                    }
                    return false;
                }
                common_flydelta_representation_augmentation_state resume_state;
                if (!callbacks.resolve_representation_augmentation_state(
                        job.search_state_ref, resume_state, error) ||
                        !common_flydelta_representation_augmentation_state_validate(
                            resume_state, config.representation_augmentation, error)) {
                    if (error.empty()) error = "FlyDelta augmentation resume state is invalid";
                    return false;
                }
                common_flydelta_search_pipeline_result pipeline_result;
                common_flydelta_representation_augmentation_state next_state;
                if (!callbacks.run_representation_augmentation_with_state(
                        job, &resume_state, pipeline_result, next_state, error) ||
                        !common_flydelta_representation_augmentation_state_validate(
                            next_state, config.representation_augmentation, error) ||
                        !validate_search_pipeline_result(
                            pipeline_result, config.pipeline, error)) return false;
                if (!callbacks.persist_representation_augmentation_state(
                        next_state, result.representation_augmentation_state_ref, error) ||
                        result.representation_augmentation_state_ref.empty() ||
                        result.representation_augmentation_state_ref.size() > 512) return false;
                next_state.state_ref = result.representation_augmentation_state_ref;
                result.has_representation_augmentation_state = true;
                result.representation_augmentation_state = std::move(next_state);
                result.search_state_ref = result.representation_augmentation_state_ref;
                result.has_next_action = true;
                result.next_action = next_action_from_augmentation(
                    result.representation_augmentation_state.next_action);
                result.next_action_reason =
                    result.representation_augmentation_state.next_action;
                common_flydelta_search_continuation continuation;
                if (!common_flydelta_select_search_continuation(
                        pipeline_result, continuation, error)) return false;
                result.search_pipeline_results.push_back(std::move(pipeline_result));
                result.search_continuations.push_back(std::move(continuation));
                result.processed_references = job.behavior_delta_ids.size();
                return true;
            }
            const bool has_post_bootstrap_state = !job.search_state_ref.empty();
            // The generic state callback is the continuation seam after
            // BootstrapZoom. It must never replace the initial Bootstrap
            // runner merely because the callback happens to be configured.
            const bool has_runner = has_post_bootstrap_state
                ? static_cast<bool>(callbacks.run_search_pipeline_with_search_state)
                : static_cast<bool>(callbacks.run_search_pipeline ||
                    callbacks.run_search_pipeline_with_state);
            if (!has_runner ||
                    !common_flydelta_search_pipeline_config_validate(config.pipeline, error)) {
                if (error.empty()) error = "FlyDelta search pipeline evaluator requires a host runner and config";
                return false;
            }
            common_flydelta_search_pipeline_result pipeline_result;
            if (has_post_bootstrap_state) {
                std::string next_state_ref;
                if (!callbacks.run_search_pipeline_with_search_state(
                        job, job.search_state_ref, pipeline_result, next_state_ref, error) ||
                        next_state_ref.size() > 512) {
                    if (error.empty()) error = "FlyDelta search state-aware runner returned an invalid state reference";
                    return false;
                }
                result.search_state_ref = std::move(next_state_ref);
            } else if (callbacks.run_search_pipeline_with_state) {
                common_flydelta_bootstrap_zoom_state resume_state;
                const common_flydelta_bootstrap_zoom_state * resume = nullptr;
                if (!job.bootstrap_zoom_state_ref.empty()) {
                    if (!callbacks.resolve_bootstrap_zoom_state ||
                            !callbacks.resolve_bootstrap_zoom_state(
                                job.bootstrap_zoom_state_ref, resume_state, error) ||
                            !common_flydelta_bootstrap_zoom_state_validate(resume_state, error)) {
                        if (error.empty()) error = "FlyDelta BootstrapZoom resume state is invalid";
                        return false;
                    }
                    resume = &resume_state;
                }
                common_flydelta_bootstrap_zoom_state next_state;
                if (!callbacks.run_search_pipeline_with_state(
                        job, resume, pipeline_result, next_state, error) ||
                        !common_flydelta_bootstrap_zoom_state_validate(next_state, error) ||
                        !validate_search_pipeline_result(pipeline_result, config.pipeline, error)) {
                    return false;
                }
                if (!callbacks.persist_bootstrap_zoom_state) {
                    error = "FlyDelta state-aware search requires a state persister";
                    return false;
                }
                result.has_bootstrap_zoom_state = true;
                result.bootstrap_zoom_state = std::move(next_state);
                if (!callbacks.persist_bootstrap_zoom_state(
                        result.bootstrap_zoom_state, result.bootstrap_zoom_state_ref, error) ||
                        result.bootstrap_zoom_state_ref.empty() ||
                        result.bootstrap_zoom_state_ref.size() > 512) return false;
                result.bootstrap_zoom_state.state_ref = result.bootstrap_zoom_state_ref;
            } else if (!callbacks.run_search_pipeline(job, pipeline_result, error) ||
                    !validate_search_pipeline_result(pipeline_result, config.pipeline, error)) {
                return false;
            }
            common_flydelta_search_continuation continuation;
            if (!common_flydelta_select_search_continuation(
                    pipeline_result, continuation, error)) return false;
            result.search_pipeline_results.push_back(std::move(pipeline_result));
            result.search_continuations.push_back(std::move(continuation));
            const bool has_resume_state = !job.search_state_ref.empty() ||
                !job.bootstrap_zoom_state_ref.empty();
            if (has_resume_state && callbacks.resolve_search_orchestration_state) {
                common_flydelta_experiment_plan current_plan;
                common_flydelta_utility_history history;
                const std::string & orchestration_ref = !job.search_state_ref.empty()
                    ? job.search_state_ref : job.bootstrap_zoom_state_ref;
                if (!callbacks.resolve_search_orchestration_state(
                        orchestration_ref, current_plan, history, error)) {
                    if (error.empty()) error =
                        "FlyDelta orchestration state resolver returned no state";
                    return false;
                }
                std::vector<common_flydelta_subspace_utility_observation> observations;
                if (!utility_observations_from_pipeline(
                        result.search_pipeline_results.back(), observations, error)) return false;
                common_flydelta_slice_orchestration_result orchestration;
                if (!common_flydelta_orchestrate_search_slice(
                        current_plan, config.utility_gate, observations, history,
                        orchestration, error)) return false;
                std::string next_orchestration_ref;
                if (!callbacks.persist_search_orchestration_state(
                        orchestration.plan, orchestration.utility.history,
                        next_orchestration_ref, error) ||
                        next_orchestration_ref.empty() || next_orchestration_ref.size() > 512) {
                    if (error.empty()) error =
                        "FlyDelta orchestration state persister returned an invalid reference";
                    return false;
                }
                result.search_state_ref = std::move(next_orchestration_ref);
                result.has_experiment_plan = true;
                result.experiment_plan = std::move(orchestration.plan);
                result.has_next_action = true;
                result.next_action = orchestration.next_action;
                result.utility_decision = orchestration.utility;
                result.next_action_reason = std::move(orchestration.reason);

                // BootstrapZoom -> AdaptiveAlpha is a state transition, not
                // a new queue lane. Keep the same opaque state lineage and
                // persist one new revision so the next bounded worker slice
                // can select the correct rank-one primitive.
                if (result.has_bootstrap_zoom_state &&
                        current_plan.bootstrap_refinement ==
                            common_flydelta_bootstrap_refinement_kind::bootstrap_zoom &&
                        orchestration.plan.bootstrap_refinement ==
                            common_flydelta_bootstrap_refinement_kind::adaptive_alpha &&
                        result.bootstrap_zoom_state.phase !=
                            common_flydelta_bootstrap_zoom_phase::adaptive_alpha) {
                    result.bootstrap_zoom_state.phase =
                        common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
                    result.bootstrap_zoom_state.refinement_kind =
                        common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
                    result.bootstrap_zoom_state.alpha_response_available = false;
                    result.bootstrap_zoom_state.alpha_response = {};
                    result.bootstrap_zoom_state.extra_model_trials = 0;
                    result.bootstrap_zoom_state.next_candidate_index = 0;
                    result.bootstrap_zoom_state.state_ref.clear();
                    if (!callbacks.persist_bootstrap_zoom_state(
                            result.bootstrap_zoom_state,
                            result.bootstrap_zoom_state_ref, error) ||
                            result.bootstrap_zoom_state_ref.empty() ||
                            result.bootstrap_zoom_state_ref.size() > 512) {
                        if (error.empty()) error =
                            "FlyDelta AdaptiveAlpha transition state persistence failed";
                        return false;
                    }
                    result.bootstrap_zoom_state.state_ref = result.bootstrap_zoom_state_ref;
                }
            }
            // A search job now emits the first ordered plan when the host
            // supplies its compatible evidence resolver. The plan starts at
            // Bootstrap even if the evidence ceiling is Deep; later phases
            // require explicit UtilityGate transitions by the host.
            // Initial search jobs establish the evidence ceiling and the
            // first Bootstrap plan. A post-Bootstrap slice already has its
            // phase in the host-owned opaque state; rebuilding a fresh
            // Bootstrap plan here would make the worker appear to restart
            // after plateau/orthogonal/Shallow/Deep work.
            if (!has_post_bootstrap_state && callbacks.resolve_behavior_delta) {
                common_flydelta_aggregation_config aggregation_config;
                aggregation_config.identity = config.direction;
                aggregation_config.depth = config.evidence_depth;
                aggregation_config.max_retained_samples = config.aggregation_max_retained_samples;
                common_flydelta_incremental_aggregation aggregation(aggregation_config);
                bool evidence_available = true;
                for (const auto & id : job.behavior_delta_ids) {
                    common_flydelta_contrast_sample sample;
                    if (!callbacks.resolve_behavior_delta(
                            id, sample.delta, sample.credit, error)) {
                        // The plan is an optional host-facing addition. A
                        // legacy search callback may expose this resolver but
                        // intentionally decline the reference; preserve the
                        // old search result and omit the plan in that case.
                        error.clear();
                        evidence_available = false;
                        break;
                    }
                    if (!aggregation.ingest(sample, error)) return false;
                }
                if (evidence_available) {
                    if (!aggregation.assess_depth(result.evidence_depth, error) ||
                            !common_flydelta_plan_search_continuation(
                                result.search_continuations.back(), result.evidence_depth,
                                result.experiment_plan, error)) return false;
                    result.has_experiment_plan = true;
                    result.search_budget = result.experiment_plan.budget;
                    // Whirlpool is the current bounded slice. The next
                    // bounded slice is explicitly Bootstrap; the evaluator
                    // does not execute it recursively.
                    result.has_next_action = true;
                    result.next_action = common_flydelta_next_action::run_bootstrap;
                    result.next_action_reason = "Whirlpool completed; schedule Bootstrap slice";
                }
            }
            result.processed_references = job.behavior_delta_ids.size();
            return true;
        }
        case common_flydelta_experiment_job_kind::delta_memory: {
            if (!callbacks.resolve_training_example ||
                    !common_flydelta_validate_memory_config(config.memory, error)) {
                if (error.empty()) error = "FlyDelta DeltaMemory evaluator requires a resolver and memory config";
                return false;
            }
            if (!valid_reference_count(job.training_example_ids.size(), config.max_references)) {
                error = "FlyDelta DeltaMemory evaluator reference count is invalid";
                return false;
            }
            std::vector<common_flydelta_training_example> examples;
            examples.reserve(job.training_example_ids.size());
            for (const auto & id : job.training_example_ids) {
                common_flydelta_training_example example;
                if (!callbacks.resolve_training_example(id, example, error)) return false;
                if (example.split != common_flydelta_training_split::train) {
                    error = "FlyDelta evaluator refuses validation or holdout training input";
                    return false;
                }
                examples.push_back(std::move(example));
            }
            common_flydelta_delta_memory memory(config.memory);
            if (!common_flydelta_train_delta_memory(
                    memory, examples, job.learning_rate, job.decay,
                    result.processed_references, error)) return false;
            result.delta_memory_weights = memory.weights();
            return true;
        }
    }
    error = "FlyDelta evaluator job kind is unsupported";
    return false;
}
