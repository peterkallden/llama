#include "agent/adaptation/flydelta/experiment/flydelta-coefficient-execution-internal.h"
#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"

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
    target.quality_delta = candidate.quality - baseline.quality;
    target.executed = candidate.executed;
    target.host_evaluated = baseline.host_evaluated && candidate.host_evaluated;
    target.verifier_known = baseline.verifier_known && candidate.verifier_known;
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
