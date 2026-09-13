#include "agent/adaptation/adaptation-evidence.h"

#include <cassert>

static common_adaptation_evidence make_evidence() {
    common_adaptation_evidence value;
    value.id = "adaptation://evidence/1";
    value.source = common_adaptation_evidence_source::reflection_alternative;
    value.scope.namespace_id = "local";
    value.scope.session_id = "session-1";
    value.scope.project_id = "project-1";
    value.scope.turn_id = "turn-1";
    value.task_fingerprint = "sha256:task";
    value.baseline_ref = "execution:baseline";
    value.candidate_ref = "execution:candidate";
    value.verifier_ref = "verifier:tests-v1";
    value.transaction_ids = {"learning://transaction/1", "learning://transaction/2"};
    value.cause = common_learning_cause::model_behavior;
    value.host_verified = true;
    return value;
}

int main() {
    auto value = make_evidence();
    std::string error;
    assert(common_adaptation_evidence_validate(value, 4, error));
    const auto encoded = common_adaptation_evidence_to_json(value);
    common_adaptation_evidence decoded;
    assert(common_adaptation_evidence_from_json(encoded, decoded, error));
    assert(decoded.source == common_adaptation_evidence_source::reflection_alternative);
    assert(decoded.cause == common_learning_cause::model_behavior);
    assert(decoded.transaction_ids.size() == 2);

    value.host_verified = true;
    value.verifier_ref.clear();
    assert(!common_adaptation_evidence_validate(value, 4, error));
    value = make_evidence();
    value.baseline_ref = value.candidate_ref;
    assert(!common_adaptation_evidence_validate(value, 4, error));
    return 0;
}
