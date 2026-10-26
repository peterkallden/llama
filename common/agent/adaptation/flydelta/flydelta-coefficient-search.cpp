#include "agent/adaptation/flydelta/experiment/flydelta-coefficient-execution-internal.h"
#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"

std::string common_flydelta_coefficient_basis_identity(
        const common_flydelta_low_rank_basis & basis) {
    return common_flydelta_coefficient_detail::guidance_basis_identity(basis);
}

void common_flydelta_coefficient_trial_set_observation(
        common_flydelta_coefficient_trial & target,
        const common_flydelta_counterfactual_trial & baseline,
        const common_flydelta_counterfactual_trial & candidate) {
    target.arm_id = candidate.arm_id;
    target.intervention_ref = candidate.intervention_ref;
    target.verifier_reason = candidate.verifier_reason;
    target.observed_decision_summary = candidate.observed_decision_summary;
    target.expected_decision_summary = candidate.expected_decision_summary;
    target.evidence_ref = candidate.evidence_ref;
    target.outcome = common_flydelta_classify_counterfactual(baseline, candidate);
    target.semantic_progress_observation = candidate.semantic_progress_observation;
    target.semantic_progress = {};
    if (baseline.executed && candidate.executed && baseline.host_evaluated && candidate.host_evaluated) {
        target.semantic_progress = common_flydelta_compare_semantic_progress(
            baseline.semantic_progress_observation, candidate.semantic_progress_observation,
            baseline.verifier_known && baseline.passed,
            candidate.verifier_known && candidate.passed);
    }
    target.quality_delta = candidate.quality - baseline.quality;
    target.executed = candidate.executed;
    target.host_evaluated = baseline.host_evaluated && candidate.host_evaluated;
    target.verifier_known = baseline.verifier_known && candidate.verifier_known;
    target.overlay_applied = candidate.overlay_applied;
    if (!candidate.overlay_applied) {
        target.applied_intervention_norm = 0;
        target.applied_residual_fraction = 0;
    }
}

bool common_flydelta_coefficient_trial_to_region(
        const common_flydelta_coefficient_trial & source,
        const common_flydelta_low_rank_basis & basis,
        common_flydelta_intervention_region_trial & target,
        std::string & error) {
    target = {};
    target.arm_id = source.arm_id;
    target.intervention_ref = source.intervention_ref;
    target.coefficients = source.coefficients;
    target.verifier_reason = source.verifier_reason;
    target.observed_decision_summary = source.observed_decision_summary;
    target.expected_decision_summary = source.expected_decision_summary;
    target.evidence_ref = source.evidence_ref;
    target.semantic_progress_observation = source.semantic_progress_observation;
    target.semantic_progress = source.semantic_progress;
    target.candidate.layer_indices = {static_cast<uint32_t>(basis.layer_index)};
    target.candidate.anchor_layer_index = static_cast<uint32_t>(basis.layer_index);
    target.candidate.total_scale = source.executed_strength;
    target.candidate.per_layer_scale = source.executed_strength;
    target.requested_total_scale = source.requested_strength;
    target.executed_total_scale = source.executed_strength;
    target.dose_action = source.dose_action;
    target.relative_dose = source.relative_dose;
    target.dose_evaluated = source.dose_evaluated;
    target.dose_safety_limited = source.dose_safety_limited;
    target.dose_reason = source.dose_reason;
    target.outcome = source.outcome;
    target.quality_delta = source.quality_delta;
    target.margin = source.margin;
    target.margin_comparison = source.margin_comparison;
    target.executed = source.executed;
    target.host_evaluated = source.host_evaluated;
    target.verifier_known = source.verifier_known;
    target.geometry_available = source.geometry_available;
    target.geometry = source.geometry;
    target.search_score = source.search_fitness;
    target.promising = source.outcome == common_flydelta_counterfactual_outcome::helped || source.search_fitness > 0;
    target.safe_to_continue = source.outcome != common_flydelta_counterfactual_outcome::harmed &&
        (!source.geometry_available || (source.geometry.shift_norm <= 1 && source.geometry.leakage <= 1));
    return common_flydelta_intervention_region_trial_validate(target, error);
}

bool common_flydelta_append_coefficient_search_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const std::vector<common_flydelta_coefficient_trial> & trials,
        const std::string & experimental_artifact_id,
        std::string & error) {
    error.clear();
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_low_rank_basis_validate(basis, 16, error) ||
            !common_flydelta_coefficient_search_config_validate(
                config, basis.vectors.size(), error) ||
            experimental_artifact_id.empty() || experimental_artifact_id.size() > 512) {
        if (error.empty()) error = "FlyDelta coefficient lifecycle input is invalid";
        return false;
    }
    const std::string kind = std::string("coefficient-") +
        common_flydelta_coefficient_search_strategy_name(config.strategy);
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        // Diagnostics remain in experiment state/trace, not the journal of
        // host-evaluated lifecycle observations.
        if (!trial.executed || !trial.host_evaluated) continue;
        common_flydelta_search_observation observation;
        observation.experiment_id = fixture.id;
        observation.candidate_id = experimental_artifact_id + "/coefficient/" +
            std::to_string(index);
        observation.search_kind = kind;
        observation.experimental_artifact_id = experimental_artifact_id;
        observation.fixture_baseline_ref = fixture.id;
        observation.surface_parent_best_ref = experimental_artifact_id;
        observation.outcome = trial.outcome;
        observation.host_evaluated = trial.host_evaluated;
        observation.verifier_known = trial.verifier_known;
        observation.diagnostics_available = trial.geometry_available;
        if (trial.geometry_available) {
            observation.diagnostics = trial.geometry;
        }
        observation.coefficients = trial.coefficients;
        observation.search_fitness = trial.search_fitness;
        observation.sequence_margin_available = trial.margin_comparison.available;
        observation.sequence_margin = trial.margin_comparison;
        observation.budget_remaining = index + 1 < trials.size();
        if (!fixture.scope_fingerprint.empty() &&
                !fixture.baseline_deployment_fingerprint.empty() &&
                !fixture.semantic_domain.empty() && !fixture.guidance_fixture_group_id.empty() &&
                !fixture.outcome_guidance_verifier_revision.empty() && fixture.model_n_layers > 1 &&
                !trial.arm_id.empty()) {
            std::string logical_arm_id = trial.guidance_arm_id;
            if (logical_arm_id.empty()) {
                logical_arm_id = common_flydelta_coefficient_detail::guidance_candidate(
                    trial.requested_coefficients.empty() ? trial.coefficients : trial.requested_coefficients,
                    basis, config).arm_id;
            }
            observation.guidance_observation_id = fixture.task_fingerprint + ":" + logical_arm_id;
            observation.guidance_fixture_group_id = fixture.guidance_fixture_group_id;
            observation.guidance_scope_fingerprint = fixture.guidance_scope_fingerprint.empty()
                ? fixture.scope_fingerprint : fixture.guidance_scope_fingerprint;
            observation.guidance_model_profile_fingerprint = fixture.model_profile_fingerprint;
            observation.guidance_tokenizer_fingerprint = fixture.tokenizer_fingerprint;
            observation.guidance_template_fingerprint = fixture.template_fingerprint;
            observation.guidance_baseline_deployment_fingerprint =
                fixture.baseline_deployment_fingerprint;
            observation.guidance_verifier_revision = fixture.outcome_guidance_verifier_revision;
            observation.guidance_ruleset_revision = fixture.ruleset_revision;
            observation.guidance_action_contract_fingerprint = fixture.action_contract_fingerprint;
            observation.guidance_domain = fixture.semantic_domain;
            observation.guidance_strategy = common_flydelta_coefficient_search_strategy_name(
                config.strategy);
            observation.guidance_estimator = "coefficient-margin-geometry-v1";
            observation.guidance_basis_id = trial.guidance_basis_id.empty()
                ? common_flydelta_coefficient_detail::guidance_basis_identity(basis)
                : trial.guidance_basis_id;
            observation.guidance_rank = basis.vectors.size();
            observation.guidance_relative_layer = static_cast<double>(basis.layer_index) /
                static_cast<double>(fixture.model_n_layers - 1);
            observation.guidance_relative_dose = config.max_l2_norm > 0
                ? std::min(1.0, static_cast<double>(common_flydelta_coefficient_detail::norm(
                    trial.coefficients) / config.max_l2_norm)) : 0.0;
            size_t nonzero = 0;
            for (const float coefficient : trial.coefficients) if (std::abs(coefficient) > 1e-7f) ++nonzero;
            observation.guidance_mixed_coefficients = nonzero > 1;
            observation.guidance_margin_sign = trial.guidance_margin_sign;
            observation.guidance_comparison_completed = trial.host_evaluated;
            observation.guidance_overlay_applied = trial.overlay_applied;
            observation.guidance_outcome_known =
                trial.outcome != common_flydelta_counterfactual_outcome::unknown;
        }

        common_flydelta_search_decision decision;
        if (!common_flydelta_decide_search_disposition(
                observation, decision, error)) return false;

        common_flydelta_candidate_lineage lineage;
        lineage.candidate_id = observation.candidate_id;
        lineage.parent_candidate_id = experimental_artifact_id;
        if (trial.parent_trial_index < trials.size()) {
            lineage.parent_candidate_id = experimental_artifact_id + "/coefficient/" +
                std::to_string(trial.parent_trial_index);
        }
        lineage.mutation_kind = trial.mutation_kind.empty()
            ? "coefficient_arm" : trial.mutation_kind;
        lineage.generation = static_cast<uint32_t>(trial.iteration);
        lineage.direction_id = experimental_artifact_id + "/basis";
        lineage.layer_indices = {static_cast<uint32_t>(basis.layer_index)};
        lineage.scale = common_flydelta_coefficient_detail::norm(trial.coefficients);
        lineage.intervention_budget = config.max_l2_norm;

        const std::string suffix = "/coefficient/" + std::to_string(index);
        common_flydelta_lifecycle_event_context arm_context = context;
        arm_context.event_id = context.event_id + suffix;
        arm_context.idempotency_key = context.idempotency_key + suffix;
        arm_context.source_id = context.source_id + suffix;
        if (!common_flydelta_append_search_lifecycle(
                store, arm_context, observation, decision, &lineage, error)) return false;
    }
    return true;
}
