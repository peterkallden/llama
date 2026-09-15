#include "agent/adaptation/flydelta/flydelta-capture.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_flydelta_capture_candidate candidate;
    candidate.id = "flydelta://capture-candidate/tool_repair/transaction-1";
    candidate.transaction_id = "transaction-1";
    candidate.source = common_adaptation_evidence_source::tool_repair;
    candidate.evidence_refs = {"evidence-1"};
    candidate.model_profile_fingerprint = "sha256:model";
    candidate.capture_layout_revision = "l_out:v1";
    candidate.candidate_ready = true;

    common_adaptation_evidence evidence;
    evidence.id = "evidence-1";
    evidence.source = candidate.source;
    evidence.behavior_key = "tool_use/diagnostics/missing-argument";
    evidence.scope.namespace_id = "local";
    evidence.scope.session_id = "session";
    evidence.task_fingerprint = "sha256:task";
    evidence.baseline_ref = "execution:failed";
    evidence.candidate_ref = "execution:repaired";
    evidence.verifier_ref = "verifier:v1";
    evidence.transaction_ids = {candidate.transaction_id};
    evidence.host_verified = true;

    common_flydelta_capture_manifest manifest;
    CHECK(common_flydelta_capture_manifest_from_candidate(
        candidate, evidence, "sha256:template", "sha256:execution-context",
        "sha256:evidence", 128, true,
        manifest, error));
    CHECK(manifest.observation_id == candidate.transaction_id);
    CHECK(manifest.source == candidate.source);
    CHECK(manifest.behavior_key == evidence.behavior_key);
    CHECK(manifest.positive_execution_ref == evidence.candidate_ref);
    CHECK(manifest.negative_execution_ref == evidence.baseline_ref);
    CHECK(common_flydelta_capture_manifest_validate(manifest, 1024, error));

    candidate.behavior_key = "tool_use/diagnostics/other";
    CHECK(!common_flydelta_capture_manifest_from_candidate(
        candidate, evidence, "sha256:template", "sha256:execution-context",
        "sha256:evidence", 128, true,
        manifest, error));
    candidate.behavior_key = evidence.behavior_key;

    candidate.source = common_adaptation_evidence_source::reflection_alternative;
    evidence.source = candidate.source;
    candidate.behavior_key = "reflection/alternative";
    evidence.behavior_key = candidate.behavior_key;
    CHECK(common_flydelta_capture_manifest_from_candidate(
        candidate, evidence, "sha256:template", "sha256:execution-context",
        "sha256:evidence", 128, true,
        manifest, error));
    CHECK(manifest.source == common_adaptation_evidence_source::reflection_alternative);

    evidence.source = common_adaptation_evidence_source::tool_repair;
    CHECK(!common_flydelta_capture_manifest_from_candidate(
        candidate, evidence, "sha256:template", "sha256:execution-context",
        "sha256:evidence", 128, true,
        manifest, error));
    evidence.source = candidate.source;
    CHECK(!common_flydelta_capture_manifest_from_candidate(
        candidate, evidence, "sha256:template", "sha256:execution-context",
        "sha256:evidence", 128, false,
        manifest, error));
    return 0;
}
