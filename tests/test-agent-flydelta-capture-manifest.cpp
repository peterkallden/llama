#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-evidence.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_adaptation_evidence transition_evidence;
    transition_evidence.id = "evidence://repair/transition-1";
    transition_evidence.source = common_adaptation_evidence_source::tool_repair;
    transition_evidence.behavior_key = "tool_use/diagnostics/missing-argument";
    transition_evidence.scope.namespace_id = "local";
    transition_evidence.scope.project_id = "project";
    transition_evidence.scope.session_id = "session";
    transition_evidence.task_fingerprint = "sha256:task";
    transition_evidence.baseline_ref = "execution:failed";
    transition_evidence.candidate_ref = "execution:repaired";
    transition_evidence.verifier_ref = "verifier:v1";
    transition_evidence.transaction_ids = {"transaction-failed", "transaction-repaired"};
    transition_evidence.host_verified = true;
    common_flydelta_behavior_transition transition;
    transition.id = "learning://flydelta/transition-1";
    transition.source = transition_evidence.source;
    transition.behavior_key = transition_evidence.behavior_key;
    transition.scope = transition_evidence.scope;
    transition.task_fingerprint = transition_evidence.task_fingerprint;
    transition.baseline_transaction_id = "transaction-failed";
    transition.candidate_transaction_id = "transaction-repaired";
    transition.baseline_execution_ref = transition_evidence.baseline_ref;
    transition.candidate_execution_ref = transition_evidence.candidate_ref;
    transition.host_verifier_ref = transition_evidence.verifier_ref;
    common_flydelta_capture_candidate generated;
    CHECK(common_flydelta_capture_candidate_from_transition(
        transition, transition_evidence, "sha256:model", "l_out:v1", generated, error));
    CHECK(generated.transaction_id == transition.candidate_transaction_id);

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

    // The adapter is intentionally transport/source neutral. Exercise the
    // same verified transition -> candidate -> manifest path for sources
    // which do not originate in tool repair.
    const common_adaptation_evidence_source generic_sources[] = {
        common_adaptation_evidence_source::planning_revision,
        common_adaptation_evidence_source::research_alternative,
        common_adaptation_evidence_source::dataset_resource,
        common_adaptation_evidence_source::user_correction,
    };
    for (const auto source : generic_sources) {
        common_adaptation_evidence generic = evidence;
        generic.id = std::string("evidence://generic/") +
            common_adaptation_evidence_source_name(source);
        generic.source = source;
        generic.behavior_key = std::string("verified/") +
            common_adaptation_evidence_source_name(source);
        generic.transaction_ids = {"generic-baseline", "generic-candidate"};
        common_flydelta_behavior_transition generic_transition;
        CHECK(common_flydelta_behavior_transition_from_evidence(
            generic, generic.transaction_ids[0], generic.transaction_ids[1],
            generic_transition, error));
        common_flydelta_capture_candidate generic_candidate;
        CHECK(common_flydelta_capture_candidate_from_transition(
            generic_transition, generic, "sha256:model", "l_out:v1",
            generic_candidate, error));
        common_flydelta_capture_manifest generic_manifest;
        CHECK(common_flydelta_capture_manifest_from_candidate(
            generic_candidate, generic, "sha256:template", "sha256:execution-context",
            "sha256:generic-evidence", 128, true, generic_manifest, error));
        CHECK(generic_manifest.source == source &&
            generic_manifest.behavior_key == generic.behavior_key);
    }
    return 0;
}
