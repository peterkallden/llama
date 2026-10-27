#include "agent/adaptation/flydelta/experiment/flydelta-coefficient-execution-internal.h"
#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"

#include <cmath>

bool common_flydelta_reserve_search_evaluations(
        common_flydelta_search_execution_budget & budget, size_t count, std::string & error) {
    error.clear();
    if (budget.limit == 0 || budget.used > budget.limit || count > budget.limit - budget.used) {
        error = "FlyDelta search model evaluation budget exhausted";
        return false;
    }
    budget.used += count;
    return true;
}

std::string common_flydelta_coefficient_basis_identity(
        const common_flydelta_low_rank_basis & basis) {
    return common_flydelta_coefficient_detail::guidance_basis_identity(basis);
}

bool common_flydelta_reconstruct_intervention(
        const common_flydelta_low_rank_basis & basis,
        const std::string & expected_basis_identity,
        const std::vector<float> & applied_coefficients,
        std::vector<float> & intervention,
        std::string & error) {
    intervention.clear();
    error.clear();
    if (!common_flydelta_low_rank_basis_validate(basis, 16, error) ||
            expected_basis_identity.empty() ||
            expected_basis_identity != common_flydelta_coefficient_basis_identity(basis) ||
            applied_coefficients.size() != basis.vectors.size()) {
        if (error.empty()) error = "FlyDelta intervention basis or coefficient identity differs";
        return false;
    }
    intervention.assign(basis.dimension, 0.0f);
    for (size_t row = 0; row < applied_coefficients.size(); ++row) {
        if (!std::isfinite(applied_coefficients[row])) {
            intervention.clear(); error = "FlyDelta intervention coefficient is non-finite"; return false;
        }
        for (size_t column = 0; column < basis.dimension; ++column) {
            intervention[column] += applied_coefficients[row] * basis.vectors[row][column];
            if (!std::isfinite(intervention[column])) {
                intervention.clear(); error = "FlyDelta intervention reconstruction overflow"; return false;
            }
        }
    }
    return true;
}

void common_flydelta_coefficient_trial_set_observation(
        common_flydelta_coefficient_trial & target,
        const common_flydelta_counterfactual_trial & baseline,
        const common_flydelta_counterfactual_trial & candidate) {
    target.candidate_observation = candidate;
    target.baseline_observation = baseline;
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

bool common_flydelta_coefficient_trial_progress_evidence(
        const common_flydelta_coefficient_trial & trial,
        common_flydelta_progress_evidence & evidence, std::string & error) {
    evidence = {}; error.clear();
    if (!trial.executed || !trial.host_evaluated || !trial.overlay_applied ||
            !trial.baseline_observation || !trial.candidate_observation) return false;
    const auto & before = *trial.baseline_observation;
    const auto & after = *trial.candidate_observation;
    if (!before.executed || !after.executed || before.overlay_applied || !after.overlay_applied ||
            !before.host_evaluated || !after.host_evaluated) return false;
    evidence.baseline_ref = before.arm_id;
    evidence.candidate_ref = after.arm_id;
    evidence.baseline_evidence_ref = before.evidence_ref;
    evidence.candidate_evidence_ref = after.evidence_ref;
    evidence.baseline = before.semantic_progress_observation.objective;
    evidence.candidate = after.semantic_progress_observation.objective;
    evidence.parent_ref = trial.progress_parent_ref;
    evidence.parent = trial.progress_parent;
    // Legacy text projections remain diagnostic; they cannot fabricate goals.
    if (!evidence.baseline.assessment_known && !evidence.candidate.assessment_known) return false;
    return common_flydelta_progress_evidence_validate(evidence, error);
}

bool common_flydelta_coefficient_trial_to_region(
        const common_flydelta_coefficient_trial & source,
        const common_flydelta_low_rank_basis & basis,
        common_flydelta_intervention_region_trial & target,
        std::string & error) {
    target = {};
    target.candidate_observation = source.candidate_observation;
    target.baseline_observation = source.baseline_observation;
    target.arm_id = source.arm_id;
    target.intervention_ref = source.intervention_ref;
    target.coefficients = source.coefficients;
    target.requested_coefficients = source.requested_coefficients;
    target.intervention_norms_available = source.intervention_norms_available;
    target.requested_intervention_norm = source.requested_intervention_norm;
    target.applied_intervention_norm = source.applied_intervention_norm;
    target.requested_residual_fraction = source.requested_residual_fraction;
    target.applied_residual_fraction = source.applied_residual_fraction;
    target.residual_calibration_ref = source.residual_calibration_ref;

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
            const auto observation_identity = fixture.task_fingerprint + ":" + context.event_id + ":" + logical_arm_id;
            observation.guidance_observation_id = "sha256:" + hash_sha256_hex(
                observation_identity.data(), observation_identity.size());
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
            const auto physical = common_flydelta_coefficient_detail::guidance_candidate(trial.coefficients, basis, config);
            observation.guidance_task_fingerprint = fixture.task_fingerprint;
            observation.guidance_lineage_ref = experimental_artifact_id;
            observation.guidance_direction_id = physical.direction_identity;
            observation.guidance_dose_identity = physical.dose_identity;
            common_flydelta_progress_evidence progress;
            if (common_flydelta_coefficient_trial_progress_evidence(trial, progress, error)) {
                observation.guidance_progress = std::move(progress);
            } else if (!error.empty()) return false;
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
