#include "agent/adaptation/flydelta/flydelta-basis.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_hidden_state_capture capture(float first, float second) {
    common_flydelta_hidden_state_capture value;
    value.captured = true;
    value.model_profile_fingerprint = "sha256:model";
    value.capture_layout_revision = "l_out:v1";
    value.layer_indices = {2, 4};
    value.n_embd = 2;
    value.token_index = 3;
    value.values = {first, 0.0f, second, 0.0f};
    return value;
}

int main() {
    std::string error;
    common_flydelta_capture_manifest manifest;
    manifest.id = "flydelta://capture/repair-1";
    manifest.observation_id = "transaction-1";
    manifest.behavior_key = "tool_use/diagnostics/missing-argument";
    manifest.model_profile_fingerprint = "sha256:model";
    manifest.template_fingerprint = "sha256:template";
    manifest.execution_context_fingerprint = "sha256:execution-context";
    manifest.positive_execution_ref = "execution:repaired";
    manifest.negative_execution_ref = "execution:failed";
    manifest.capture_layout_revision = "l_out:v1";
    manifest.evidence_hash = "sha256:evidence";
    manifest.redaction_attested = true;
    manifest.captured_bytes = 2 * 2 * 2 * sizeof(float);

    std::vector<common_flydelta_behavior_delta> deltas;
    CHECK(common_flydelta_behavior_deltas_from_captures(
        manifest, capture(1.0f, 3.0f), capture(2.5f, 2.0f),
        "verifier:repair", 1024, 1024, deltas, error));
    CHECK(deltas.size() == 2);
    CHECK(deltas[0].layer_index == 2 && deltas[0].values[0] == 1.5f);
    CHECK(deltas[1].layer_index == 4 && deltas[1].values[0] == -1.0f);
    CHECK(deltas[0].capture_manifest_id == manifest.id);

    common_adaptation_evidence evidence;
    evidence.id = "evidence://repair/repair-delta-1";
    evidence.source = common_adaptation_evidence_source::tool_repair;
    evidence.behavior_key = manifest.behavior_key;
    evidence.scope.namespace_id = "local";
    evidence.scope.project_id = "project";
    evidence.scope.session_id = "session";
    evidence.task_fingerprint = "sha256:task";
    evidence.baseline_ref = manifest.negative_execution_ref;
    evidence.candidate_ref = manifest.positive_execution_ref;
    evidence.verifier_ref = "verifier:repair";
    evidence.transaction_ids = {"failed-tx", "repaired-tx"};
    evidence.host_verified = true;
    common_flydelta_behavior_transition transition;
    transition.id = "learning://flydelta/repair-transition";
    transition.source = evidence.source;
    transition.behavior_key = evidence.behavior_key;
    transition.scope = evidence.scope;
    transition.task_fingerprint = evidence.task_fingerprint;
    transition.baseline_transaction_id = "failed-tx";
    transition.candidate_transaction_id = "repaired-tx";
    transition.baseline_execution_ref = evidence.baseline_ref;
    transition.candidate_execution_ref = evidence.candidate_ref;
    transition.host_verifier_ref = evidence.verifier_ref;
    manifest.observation_id = transition.candidate_transaction_id;
    std::vector<common_flydelta_behavior_delta> verified_deltas;
    CHECK(common_flydelta_behavior_deltas_from_verified_transition(
        transition, evidence, manifest, capture(1.0f, 3.0f), capture(2.5f, 2.0f),
        1024, 1024, verified_deltas, error));
    CHECK(verified_deltas.size() == 2);

    auto misaligned = capture(2.5f, 2.0f);
    misaligned.layer_indices = {2, 5};
    CHECK(!common_flydelta_behavior_deltas_from_captures(
        manifest, capture(1.0f, 3.0f), misaligned,
        "verifier:repair", 1024, 1024, deltas, error));

    auto zero = capture(1.0f, 3.0f);
    CHECK(!common_flydelta_behavior_deltas_from_captures(
        manifest, capture(1.0f, 3.0f), zero,
        "verifier:repair", 1024, 1024, deltas, error));
    return 0;
}
