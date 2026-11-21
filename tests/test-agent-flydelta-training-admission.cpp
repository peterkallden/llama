#include "agent/adaptation/flydelta/flydelta-training-admission.h"

#include <cassert>

static common_flydelta_training_case_source source_for(size_t index) {
    common_flydelta_training_case_source source;
    const auto suffix = std::to_string(index);
    auto & transaction = source.transaction;
    transaction.id = "learning://transaction/" + suffix;
    transaction.created_at = "2026-10-10T00:00:00Z";
    transaction.observation.id = transaction.id;
    transaction.observation.scope.namespace_id = "local";
    transaction.observation.scope.session_id = "session-" + suffix;
    transaction.observation.scope.turn_id = "turn-" + suffix;
    transaction.observation.source_turn_id = "turn-" + suffix;
    transaction.observation.source_plan_id = "plan-" + suffix;
    transaction.observation.signals = {common_learning_signal{
        common_learning_signal_type::successful_recovery, "plan-" + suffix,
        "step-1", "data.aggregate", "evidence-" + suffix,
        "host verified recovery", "data", "native"}};
    transaction.observation.evidence_ids = {"evidence-" + suffix};
    transaction.observation.cause = common_learning_cause::model_behavior;
    transaction.observation.verification = common_learning_verification::host_verified;
    transaction.observation.idempotency_key = "idempotency-" + suffix;
    transaction.observation.content_hash = "identity:fnv1a64:abc" + suffix;
    transaction.observation.collection_allowed = true;

    auto & evidence = source.evidence;
    evidence.id = "evidence://flydelta/" + suffix;
    evidence.source = common_adaptation_evidence_source::tool_repair;
    evidence.scope = transaction.observation.scope;
    evidence.teaching_key = "data-aggregate-v1";
    evidence.behavior_key = "data.aggregate/group-and-sum";
    evidence.task_fingerprint = "task:" + suffix;
    evidence.baseline_ref = "execution:baseline-" + suffix;
    evidence.candidate_ref = "execution:conditioned-" + suffix;
    evidence.verifier_ref = "oracle:data-v1";
    evidence.transaction_ids = {transaction.id};
    evidence.cause = common_learning_cause::model_behavior;
    evidence.host_verified = true;

    auto & relation = source.relation;
    std::string error;
    assert(common_flydelta_teaching_relation_from_evidence(evidence, relation, error));
    relation.teaching_key = evidence.teaching_key;
    relation.model_profile_fingerprint = "model:qwen-v1";
    relation.tokenizer_fingerprint = "tokenizer:qwen-v1";
    relation.template_fingerprint = "template:chat-v1";
    relation.oracle_revision = "oracle-revision:data-v1";
    relation.contrast_ref = "contrast:" + suffix;

    auto & learning_case = source.learning_case;
    learning_case.id = "learning://case/" + suffix;
    learning_case.observation_id = transaction.id;
    learning_case.source = evidence.source;
    learning_case.scope = transaction.observation.scope;
    learning_case.evidence_ids = transaction.observation.evidence_ids;
    learning_case.schema_fingerprint = "tool-schema:data.aggregate:v1";
    learning_case.input = "Aggregate the registered data by region.";
    learning_case.rejected_action = "Use data.describe without grouping.";
    learning_case.preferred_action = "Call data.aggregate grouped by region.";
    learning_case.redaction_policy_id = "redaction:v1";
    learning_case.redaction_method = "fixture-redaction";
    learning_case.redaction_status = common_learning_redaction_status::policy_checked;
    learning_case.status = common_learning_case_status::approved;
    learning_case.content_hash = "identity:fnv1a64:case" + suffix;
    learning_case.learning_domain = "tool_use";
    learning_case.tool_family = "data";
    learning_case.provider_kind = "native";
    return source;
}

int main() {
    std::vector<common_flydelta_training_case_source> sources = {
        source_for(1), source_for(2), source_for(3),
    };
    common_flydelta_training_admission_policy policy;
    std::vector<common_training_candidate> candidates;
    std::string error;
    assert(common_flydelta_admit_training_cases(sources, policy, candidates, error));
    assert(candidates.size() == 3);
    for (const auto & candidate : candidates) {
        assert(common_training_candidate_qualifies(candidate, {}, error));
        assert(candidate.rejected_target == "Use data.describe without grouping.");
        assert(candidate.teaching_key == "data-aggregate-v1");
        assert(candidate.teaching_relation_id.find("/teaching") != std::string::npos);
        assert(candidate.evidence_ref.rfind("evidence://flydelta/", 0) == 0);
        assert(candidate.model_profile_fingerprint == "model:qwen-v1");
        assert(candidate.contrast_ref.rfind("contrast:", 0) == 0);
        assert(candidate.verifier_revision == "oracle-revision:data-v1");
    }

    auto incompatible = sources;
    incompatible.back().relation.oracle_revision = "oracle-revision:changed";
    assert(!common_flydelta_admit_training_cases(incompatible, policy, candidates, error));

    auto duplicate_task = sources;
    duplicate_task.back().relation.task_fingerprint = duplicate_task.front().relation.task_fingerprint;
    duplicate_task.back().evidence.task_fingerprint = duplicate_task.back().relation.task_fingerprint;
    assert(!common_flydelta_admit_training_cases(duplicate_task, policy, candidates, error));

    auto unverified = sources;
    unverified.front().relation.host_approved = false;
    assert(!common_flydelta_admit_training_cases(unverified, policy, candidates, error));

    auto conflicting_case_provenance = sources;
    conflicting_case_provenance.front().learning_case.verifier_revision = "oracle-revision:other";
    assert(!common_flydelta_admit_training_cases(
        conflicting_case_provenance, policy, candidates, error));
    return 0;
}
