#include "agent/adaptation/training-candidate.h"

#include <cassert>

int main() {
    common_learning_observation model_observation;
    model_observation.cause = common_learning_cause::model_behavior;
    model_observation.verification = common_learning_verification::host_verified;
    assert(common_learning_destination_for(model_observation) == common_learning_destination::training_candidate);

    common_learning_observation host_observation = model_observation;
    host_observation.cause = common_learning_cause::host_contract;
    assert(common_learning_destination_for(host_observation) == common_learning_destination::retain);

    common_learning_observation project_observation = model_observation;
    project_observation.cause = common_learning_cause::project_knowledge;
    assert(common_learning_destination_for(project_observation) == common_learning_destination::memory);

    common_training_candidate candidate;
    candidate.id = "learning://candidate/1";
    candidate.transaction_ids = {"learning://transaction/1", "learning://transaction/2", "learning://transaction/3"};
    candidate.cause = common_learning_cause::model_behavior;
    candidate.hypothesis = "The model emits an invalid binding for a stable contract.";
    candidate.approved_prompt = "Use the stable fixture contract.";
    candidate.approved_target = "Emit a valid tool binding.";
    candidate.observed_occurrences = 3;
    candidate.verified_recoveries = 2;
    candidate.confidence = 0.9f;
    std::string error;
    assert(!common_training_candidate_qualifies(candidate, {}, error));
    candidate.status = common_training_candidate_status::approved;
    assert(common_training_candidate_qualifies(candidate, {}, error));

    candidate.contradictions = 1;
    assert(!common_training_candidate_qualifies(candidate, {}, error));
    candidate.contradictions = 0;
    candidate.cause = common_learning_cause::host_contract;
    assert(!common_training_candidate_qualifies(candidate, {}, error));

    common_learning_transaction transaction;
    transaction.id = "learning://transaction/approved-1";
    transaction.created_at = "2026-08-29T00:00:00Z";
    transaction.observation.id = transaction.id;
    transaction.observation.scope.namespace_id = "local";
    transaction.observation.scope.session_id = "session-1";
    transaction.observation.scope.turn_id = "turn-1";
    transaction.observation.source_turn_id = "turn-1";
    transaction.observation.source_plan_id = "plan-1";
    transaction.observation.signals = {common_learning_signal{
        common_learning_signal_type::tool_failure, "plan-1", "step-1", "diagnostics.compile", "evidence-1",
        "invalid binding", "diagnostics", "openapi"}};
    transaction.observation.evidence_ids = {"evidence-1"};
    transaction.observation.cause = common_learning_cause::model_behavior;
    transaction.observation.verification = common_learning_verification::host_verified;
    transaction.observation.idempotency_key = "idempotency-1";
    transaction.observation.content_hash = "identity:fnv1a64:abc";
    transaction.observation.collection_allowed = true;

    common_learning_case learning_case;
    learning_case.id = "case-approved-1";
    learning_case.observation_id = transaction.id;
    learning_case.scope = transaction.observation.scope;
    learning_case.evidence_ids = {"evidence-1"};
    learning_case.schema_fingerprint = "tool-schema:diagnostics.compile:v1";
    learning_case.input = "Compile the project.";
    learning_case.rejected_action = "emit an invalid tool binding";
    learning_case.preferred_action = "emit a valid diagnostics binding";
    learning_case.redaction_policy_id = "stub:caller-asserted-v1";
    learning_case.redaction_method = "caller_asserted";
    learning_case.redaction_status = common_learning_redaction_status::caller_asserted;
    learning_case.status = common_learning_case_status::approved;
    learning_case.content_hash = "identity:fnv1a64:case";
    learning_case.learning_domain = "tool_use";
    learning_case.tool_family = "diagnostics";
    learning_case.provider_kind = "openapi";
    common_training_candidate promoted;
    assert(common_training_candidate_from_approved_case(
        learning_case, transaction,
        {"stable diagnostics binding behavior", 3, 2, 0, 0.95f}, promoted, error));
    assert(promoted.status == common_training_candidate_status::approved);
    assert(promoted.tool_family == "diagnostics" && promoted.provider_kind == "openapi");
    assert(common_training_candidate_qualifies(promoted, {}, error));

    learning_case.status = common_learning_case_status::redacted;
    assert(!common_training_candidate_from_approved_case(
        learning_case, transaction,
        {"not explicitly approved", 3, 2, 0, 0.95f}, promoted, error));
    return 0;
}
