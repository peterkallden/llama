#include "agent/adaptation/flydelta/flydelta-evidence.h"

#include <algorithm>
#include <cmath>

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool supported_seed_source(common_adaptation_evidence_source source) {
    switch (source) {
        case common_adaptation_evidence_source::tool_repair:
        case common_adaptation_evidence_source::reflection_alternative:
        case common_adaptation_evidence_source::planning_revision:
        case common_adaptation_evidence_source::research_alternative:
        case common_adaptation_evidence_source::dataset_resource:
        case common_adaptation_evidence_source::workflow_code:
        case common_adaptation_evidence_source::procedure_blueprint:
        case common_adaptation_evidence_source::user_correction:
        case common_adaptation_evidence_source::user_taught_concept:
            return true;
    }
    return false;
}

bool verified_model_behavior(const common_learning_transaction & transaction) {
    const auto & observation = transaction.observation;
    return observation.collection_allowed && observation.cause == common_learning_cause::model_behavior &&
        (observation.verification == common_learning_verification::host_verified ||
         observation.verification == common_learning_verification::user_confirmed) &&
        common_learning_observation_qualifies(observation);
}

bool has_signal(const common_learning_observation & observation, common_learning_signal_type type) {
    return std::any_of(observation.signals.begin(), observation.signals.end(),
        [type](const auto & signal) { return signal.type == type; });
}

} // namespace

bool common_flydelta_behavior_transition_validate(
        const common_flydelta_behavior_transition & transition,
        std::string & error) {
    error.clear();
    if (transition.schema_version != 1 || !nonempty_bounded(transition.id) ||
            !nonempty_bounded(transition.behavior_key) || transition.scope.namespace_id.empty() ||
            !nonempty_bounded(transition.task_fingerprint) ||
            !nonempty_bounded(transition.baseline_transaction_id) ||
            !nonempty_bounded(transition.candidate_transaction_id) ||
            transition.baseline_transaction_id == transition.candidate_transaction_id ||
            !nonempty_bounded(transition.baseline_execution_ref) ||
            !nonempty_bounded(transition.candidate_execution_ref) ||
            !nonempty_bounded(transition.host_verifier_ref)) {
        error = "FlyDelta behavior transition identity is incomplete";
        return false;
    }
    return true;
}

bool common_flydelta_tool_repair_transition_from_transactions(
        const common_learning_transaction & failed,
        const common_learning_transaction & repaired,
        const std::string & task_fingerprint,
        const std::string & behavior_key,
        const std::string & failed_execution_ref,
        const std::string & repaired_execution_ref,
        const std::string & host_verifier_ref,
        common_flydelta_behavior_transition & transition,
        std::string & error) {
    error.clear();
    if (!common_learning_transaction_validate(failed, 64, error) ||
            !common_learning_transaction_validate(repaired, 64, error)) {
        return false;
    }
    if (!verified_model_behavior(failed) || !verified_model_behavior(repaired) ||
            !has_signal(failed.observation, common_learning_signal_type::tool_failure) ||
            !has_signal(repaired.observation, common_learning_signal_type::successful_recovery) ||
            !nonempty_bounded(task_fingerprint) || !nonempty_bounded(behavior_key) ||
            failed.observation.scope.namespace_id != repaired.observation.scope.namespace_id ||
            failed.observation.scope.project_id != repaired.observation.scope.project_id ||
            failed.observation.scope.session_id != repaired.observation.scope.session_id ||
            failed.observation.cause != repaired.observation.cause) {
        error = "FlyDelta tool-repair transition requires aligned host-verified failure and recovery";
        return false;
    }
    transition = {};
    transition.id = "learning://flydelta/tool-repair/" + failed.id + "/" + repaired.id;
    transition.source = common_adaptation_evidence_source::tool_repair;
    transition.behavior_key = behavior_key;
    transition.scope = failed.observation.scope;
    transition.task_fingerprint = task_fingerprint;
    transition.baseline_transaction_id = failed.id;
    transition.candidate_transaction_id = repaired.id;
    transition.baseline_execution_ref = failed_execution_ref;
    transition.candidate_execution_ref = repaired_execution_ref;
    transition.host_verifier_ref = host_verifier_ref;
    return common_flydelta_behavior_transition_validate(transition, error);
}

bool common_flydelta_behavior_transition_from_evidence(
        const common_adaptation_evidence & evidence,
        const std::string & baseline_transaction_id,
        const std::string & candidate_transaction_id,
        common_flydelta_behavior_transition & transition,
        std::string & error) {
    error.clear();
    if (!common_adaptation_evidence_validate(evidence, 64, error) ||
            !evidence.host_verified || !nonempty_bounded(baseline_transaction_id) ||
            !nonempty_bounded(candidate_transaction_id) ||
            baseline_transaction_id == candidate_transaction_id ||
            std::find(evidence.transaction_ids.begin(), evidence.transaction_ids.end(),
                baseline_transaction_id) == evidence.transaction_ids.end() ||
            std::find(evidence.transaction_ids.begin(), evidence.transaction_ids.end(),
                candidate_transaction_id) == evidence.transaction_ids.end()) {
        if (error.empty()) error = "FlyDelta behavior transition requires aligned host evidence";
        return false;
    }
    transition = {};
    transition.id = evidence.id + "/flydelta/behavior";
    transition.source = evidence.source;
    transition.behavior_key = evidence.behavior_key;
    transition.scope = evidence.scope;
    transition.task_fingerprint = evidence.task_fingerprint;
    transition.baseline_transaction_id = baseline_transaction_id;
    transition.candidate_transaction_id = candidate_transaction_id;
    transition.baseline_execution_ref = evidence.baseline_ref;
    transition.candidate_execution_ref = evidence.candidate_ref;
    transition.host_verifier_ref = evidence.verifier_ref;
    return common_flydelta_behavior_transition_validate(transition, error);
}

bool common_flydelta_contrast_set_validate(
        const common_flydelta_contrast_set & contrast_set,
        size_t max_transitions,
        std::string & error) {
    error.clear();
    if (contrast_set.schema_version != 1 || !nonempty_bounded(contrast_set.id) ||
            !nonempty_bounded(contrast_set.behavior_key) || contrast_set.scope.namespace_id.empty() ||
            contrast_set.transition_ids.empty() ||
            contrast_set.transition_ids.size() > max_transitions ||
            contrast_set.transition_ids.size() != contrast_set.positive_transaction_ids.size() ||
            contrast_set.transition_ids.size() != contrast_set.negative_transaction_ids.size()) {
        error = "FlyDelta contrast set bounds or identity are invalid";
        return false;
    }
    for (size_t i = 0; i < contrast_set.transition_ids.size(); ++i) {
        if (!nonempty_bounded(contrast_set.transition_ids[i]) ||
                !nonempty_bounded(contrast_set.positive_transaction_ids[i]) ||
                !nonempty_bounded(contrast_set.negative_transaction_ids[i]) ||
                contrast_set.positive_transaction_ids[i] == contrast_set.negative_transaction_ids[i]) {
            error = "FlyDelta contrast set contains an invalid aligned pair";
            return false;
        }
    }
    return true;
}

bool common_flydelta_contrast_set_from_transitions(
        const std::string & id,
        const std::string & behavior_key,
        const std::vector<common_flydelta_behavior_transition> & transitions,
        size_t max_transitions,
        common_flydelta_contrast_set & contrast_set,
        std::string & error) {
    error.clear();
    if (transitions.empty() || transitions.size() > max_transitions ||
            !nonempty_bounded(id) || !nonempty_bounded(behavior_key)) {
        error = "FlyDelta contrast set input bound is invalid";
        return false;
    }
    contrast_set = {};
    contrast_set.id = id;
    contrast_set.behavior_key = behavior_key;
    contrast_set.scope = transitions.front().scope;
    for (const auto & transition : transitions) {
        if (!common_flydelta_behavior_transition_validate(transition, error) ||
                transition.scope.namespace_id != contrast_set.scope.namespace_id ||
                transition.scope.project_id != contrast_set.scope.project_id ||
                transition.scope.session_id != contrast_set.scope.session_id) {
            error = "FlyDelta contrast set mixes invalid or incompatible transitions";
            return false;
        }
        contrast_set.transition_ids.push_back(transition.id);
        contrast_set.positive_transaction_ids.push_back(transition.candidate_transaction_id);
        contrast_set.negative_transaction_ids.push_back(transition.baseline_transaction_id);
    }
    return common_flydelta_contrast_set_validate(contrast_set, max_transitions, error);
}

bool common_flydelta_intervention_credit_validate(
        const common_flydelta_intervention_credit & credit,
        std::string & error) {
    error.clear();
    if (credit.schema_version != 1 || !nonempty_bounded(credit.experiment_id) ||
            !nonempty_bounded(credit.candidate_id) || !nonempty_bounded(credit.fixture_id) ||
            !std::isfinite(credit.quality_delta) || credit.quality_delta < -1.0f ||
            credit.quality_delta > 1.0f ||
            (credit.eligible_for_learning &&
                credit.outcome != common_flydelta_counterfactual_outcome::helped)) {
        error = "FlyDelta intervention credit is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_intervention_credit_from_report(
        const common_flydelta_counterfactual_report & report,
        common_flydelta_intervention_credit & credit,
        std::string & error) {
    error.clear();
    if (!common_flydelta_counterfactual_report_validate(report, error)) return false;
    credit = {};
    credit.experiment_id = report.experiment_id;
    credit.candidate_id = report.candidate_id;
    credit.fixture_id = report.fixture_id;
    credit.outcome = report.outcome;
    credit.quality_delta = report.quality_delta;
    // Learning credit is deliberately stricter than search retention:
    // NEUTRAL and UNKNOWN remain useful observations, but only HELPED can
    // make a relation eligible for evidence-depth or promotion.
    credit.eligible_for_learning =
        report.outcome == common_flydelta_counterfactual_outcome::helped;
    return common_flydelta_intervention_credit_validate(credit, error);
}

bool common_flydelta_experiment_seed_validate(
        const common_flydelta_experiment_seed & seed,
        std::string & error) {
    error.clear();
    common_adaptation_evidence evidence;
    evidence.id = seed.evidence_ref;
    evidence.teaching_key = seed.teaching_key;
    evidence.source = seed.source;
    evidence.scope = seed.scope;
    evidence.behavior_key = seed.behavior_key;
    evidence.task_fingerprint = seed.task_fingerprint;
    evidence.baseline_ref = seed.baseline_ref;
    evidence.candidate_ref = seed.candidate_ref;
    evidence.verifier_ref = seed.verifier_ref;
    evidence.transaction_ids = seed.transaction_ids;
    evidence.host_verified = true;
    if (seed.schema_version != 1 || !nonempty_bounded(seed.id) ||
            !nonempty_bounded(seed.behavior_key) || !supported_seed_source(seed.source) ||
            !nonempty_bounded(seed.model_profile_fingerprint) ||
            !nonempty_bounded(seed.tokenizer_fingerprint) ||
            !nonempty_bounded(seed.template_fingerprint) ||
            !nonempty_bounded(seed.execution_context_fingerprint) ||
            !nonempty_bounded(seed.evidence_ref) ||
            !common_adaptation_evidence_validate(evidence, 64, error)) {
        if (error.empty()) error = "FlyDelta experiment seed identity is incomplete";
        return false;
    }
    return true;
}

bool common_flydelta_experiment_seed_from_evidence(
        const common_adaptation_evidence & evidence,
        const std::string & behavior_key,
        const std::string & model_profile_fingerprint,
        const std::string & tokenizer_fingerprint,
        const std::string & template_fingerprint,
        const std::string & execution_context_fingerprint,
        common_flydelta_training_split split,
        common_flydelta_experiment_seed & seed,
        std::string & error) {
    error.clear();
    if (!common_adaptation_evidence_validate(evidence, 64, error) ||
            !evidence.host_verified || !supported_seed_source(evidence.source) ||
            !nonempty_bounded(behavior_key) || !nonempty_bounded(model_profile_fingerprint) ||
            !nonempty_bounded(tokenizer_fingerprint) || !nonempty_bounded(template_fingerprint) ||
            !nonempty_bounded(execution_context_fingerprint)) {
        if (error.empty()) error = "FlyDelta experiment seed requires host-verified supported evidence";
        return false;
    }
    seed = {};
    seed.id = evidence.id + "/flydelta";
    seed.teaching_key = evidence.teaching_key;
    seed.behavior_key = behavior_key;
    seed.source = evidence.source;
    seed.scope = evidence.scope;
    seed.split = split;
    seed.task_fingerprint = evidence.task_fingerprint;
    seed.model_profile_fingerprint = model_profile_fingerprint;
    seed.tokenizer_fingerprint = tokenizer_fingerprint;
    seed.template_fingerprint = template_fingerprint;
    seed.execution_context_fingerprint = execution_context_fingerprint;
    seed.baseline_ref = evidence.baseline_ref;
    seed.candidate_ref = evidence.candidate_ref;
    seed.verifier_ref = evidence.verifier_ref;
    seed.evidence_ref = evidence.id;
    seed.transaction_ids = evidence.transaction_ids;
    return common_flydelta_experiment_seed_validate(seed, error);
}

bool common_flydelta_experiment_fixture_from_seed(
        const common_flydelta_experiment_seed & seed,
        common_flydelta_experiment_fixture & fixture,
        std::string & error) {
    error.clear();
    if (!common_flydelta_experiment_seed_validate(seed, error)) return false;
    fixture = {};
    fixture.id = seed.id + "/fixture";
    fixture.task_fingerprint = seed.task_fingerprint;
    fixture.model_profile_fingerprint = seed.model_profile_fingerprint;
    fixture.tokenizer_fingerprint = seed.tokenizer_fingerprint;
    fixture.template_fingerprint = seed.template_fingerprint;
    fixture.execution_context_fingerprint = seed.execution_context_fingerprint;
    fixture.verifier_revision = seed.verifier_ref;
    return common_flydelta_experiment_fixture_validate(fixture, error);
}
