#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "agent/adaptation/flydelta/flydelta-sideband-registry.h"

#include <chrono>
#include <string>

#define CHECK(condition) do { if (!(condition)) return 1; } while (false)

static common_flydelta_sideband_manifest manifest() {
    common_flydelta_sideband_manifest value;
    value.id = "flydelta://sideband/tool-repair-v1";
    value.artifact_path = "sidebands/tool-repair-v1.json";
    value.artifact_hash = "sha256:artifact-v1";
    value.compatibility.base_model_fingerprint = "sha256:base";
    value.compatibility.tokenizer_fingerprint = "sha256:tokenizer";
    value.compatibility.template_fingerprint = "sha256:template";
    value.compatibility.architecture = "qwen2";
    value.compatibility.inference_layout_revision = "layout:cvec-v1";
    value.model_n_embd = 4;
    value.model_n_layers = 3;
    value.il_end = 2;
    return value;
}

static common_agent_model_profile profile() {
    common_agent_model_profile value;
    value.id = "agent-default";
    value.base_model_id = "qwen";
    value.base_model_fingerprint = "sha256:base";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.chat_template_fingerprint = "sha256:template";
    value.context_size_tokens = 4096;
    value.sidebands.push_back({"flydelta://sideband/tool-repair-v1", 0.5});
    return value;
}

int main() {
    std::string error;
    auto sideband = manifest();
    common_flydelta_sideband_manifest parsed;
    CHECK(common_flydelta_sideband_manifest_from_json(
        common_flydelta_sideband_manifest_to_json(sideband), parsed, error));
    CHECK(parsed.id == sideband.id && parsed.namespace_id == "local" &&
        parsed.project_id == "default");
    common_flydelta_sideband_registry registry;
    CHECK(registry.admit(sideband, error));
    CHECK(registry.admit(sideband, error));
    auto conflicting_sideband = sideband;
    conflicting_sideband.artifact_hash = "sha256:other-artifact";
    CHECK(!registry.admit(conflicting_sideband, error));
    CHECK(error.find("different metadata") != std::string::npos);
    CHECK(!registry.activate(sideband.id, error));
    CHECK(registry.stage_canary(sideband.id, "eval:tool-repair-v1", error));
    CHECK(registry.activate(sideband.id, error));

    common_flydelta_compatibility expected = sideband.compatibility;
    common_flydelta_sideband_manifest resolved;
    double scale = 0.0;
    CHECK(registry.resolve(profile(), sideband.id, expected, 4, 3, resolved, scale, error));
    CHECK(resolved.id == sideband.id && scale == 0.5 &&
            common_flydelta_sideband_status_name(resolved.status) == std::string("active"));

    expected.base_model_fingerprint = "sha256:other";
    CHECK(!registry.resolve(profile(), sideband.id, expected, 4, 3, resolved, scale, error));
    CHECK(error.find("base model") != std::string::npos);

    auto expired = manifest();
    expired.id = "flydelta://sideband/expired-v1";
    expired.expires_at_epoch_ms = 1;
    CHECK(registry.admit(expired, error));
    CHECK(!registry.stage_canary(expired.id, "eval:expired", error));
    CHECK(error.find("expired") != std::string::npos);

    CHECK(registry.revoke(sideband.id, "manual safety rollback", error));
    CHECK(!registry.resolve(profile(), sideband.id, expected, 4, 3, resolved, scale, error));
    CHECK(error.find("not active") != std::string::npos);

    auto experimental = manifest();
    experimental.id = "flydelta://sideband/experiment-v1";
    experimental.status = common_flydelta_sideband_status::experimental;
    auto invalid_experimental = experimental;
    invalid_experimental.evaluation_passed = true;
    CHECK(!common_flydelta_sideband_manifest_validate(invalid_experimental, error));
    CHECK(common_flydelta_sideband_manifest_from_json(
        common_flydelta_sideband_manifest_to_json(experimental), parsed, error));
    CHECK(parsed.status == common_flydelta_sideband_status::experimental);
    CHECK(registry.admit_experimental(experimental, error));
    auto experimental_profile = profile();
    experimental_profile.sidebands.front().sideband_id = experimental.id;
    CHECK(!registry.resolve(experimental_profile, experimental.id, experimental.compatibility, 4, 3, resolved, scale, error));
    CHECK(error.find("not active") != std::string::npos);
    CHECK(!registry.stage_canary(experimental.id, "eval:experiment", error));
    CHECK(error.find("candidate") != std::string::npos);
    CHECK(registry.promote_experimental(experimental.id, "eval:experiment-v1", error));
    CHECK(!registry.list().at(experimental.id).evaluation_passed);
    CHECK(registry.stage_canary(experimental.id, "eval:canary-v1", error));
    CHECK(registry.activate(experimental.id, error));

    common_flydelta_capture_candidate_collector collector(
        "profile:qwen", "layout:cvec-v1", 3);
    common_adaptation_evidence_source_match not_ready;
    not_ready.source = common_adaptation_evidence_source::reflection_alternative;
    common_learning_transaction transaction;
    transaction.id = "learning://transaction/reflection";
    CHECK(collector.observe(not_ready, transaction, error));
    CHECK(collector.candidates().empty());

    common_adaptation_evidence_source_match ready;
    ready.source = common_adaptation_evidence_source::tool_repair;
    ready.candidate_ready = true;
    ready.evidence_refs = {"evidence:failed", "evidence:repaired"};
    CHECK(collector.observe(ready, transaction, error));
    CHECK(collector.candidates().size() == 1);
    CHECK(collector.candidates().front().model_profile_fingerprint == "profile:qwen");

    common_flydelta_capture_candidate_collector full_collector(
        "profile:qwen", "layout:cvec-v1", 1);
    CHECK(full_collector.observe(ready, transaction, error));
    CHECK(!full_collector.observe(
        common_adaptation_evidence_source_match{
            common_adaptation_evidence_source::reflection_alternative,
            {"evidence:other"}, true}, transaction, error));
    CHECK(full_collector.observe(ready, transaction, error));
    CHECK(full_collector.candidates().size() == 1);

    common_adaptation_evidence_relation reflection_relation;
    reflection_relation.source = common_adaptation_evidence_source::reflection_alternative;
    reflection_relation.behavior_key = "reflection/alternative-selection";
    reflection_relation.host_verified = true;
    common_adaptation_evidence reflection_evidence;
    reflection_evidence.id = "evidence:reflection-relation";
    reflection_evidence.source = reflection_relation.source;
    reflection_evidence.behavior_key = reflection_relation.behavior_key;
    reflection_evidence.scope.namespace_id = "local";
    reflection_evidence.scope.session_id = "session";
    reflection_evidence.task_fingerprint = "sha256:task";
    reflection_evidence.baseline_ref = "execution:reflection-failed";
    reflection_evidence.candidate_ref = "execution:reflection-repaired";
    reflection_evidence.verifier_ref = "verifier:reflection-v1";
    reflection_evidence.host_verified = true;

    common_adaptation_evidence_relation second_relation = reflection_relation;
    second_relation.behavior_key = "reflection/other-alternative";
    common_adaptation_evidence second_evidence = reflection_evidence;
    second_evidence.id = "evidence:reflection-2";
    second_evidence.behavior_key = second_relation.behavior_key;
    common_flydelta_capture_candidate_collector identity_collector(
        "profile:qwen", "layout:cvec-v1", 2);
    CHECK(identity_collector.observe_verified_relation(
        reflection_relation, reflection_evidence, transaction, error));
    CHECK(identity_collector.observe_verified_relation(
        second_relation, second_evidence, transaction, error));
    CHECK(identity_collector.candidates().size() == 2);
    CHECK(identity_collector.candidates()[0].id != identity_collector.candidates()[1].id);

    CHECK(collector.observe_verified_relation(
        reflection_relation, reflection_evidence, transaction, error));
    CHECK(collector.candidates().size() == 2);
    CHECK(collector.candidates().back().source ==
        common_adaptation_evidence_source::reflection_alternative);
    reflection_relation.host_verified = false;
    CHECK(collector.observe_verified_relation(
        reflection_relation, reflection_evidence, transaction, error));
    CHECK(collector.candidates().size() == 2);

    CHECK(collector.observe(ready, transaction, error));
    CHECK(collector.candidates().size() == 2);
    return 0;
}
