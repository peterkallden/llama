#include "agent/adaptation/adapter-registry.h"

#include <cassert>

static common_learning_adapter_manifest manifest(const std::string & id) {
    common_learning_adapter_manifest value;
    value.id = id;
    value.base_architecture = "qwen2";
    value.serving_model_fingerprint = "sha256:base-serving";
    value.training_model_fingerprint = "sha256:base-training";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.chat_template_fingerprint = "sha256:template";
    value.corpus_revision_id = "learning://corpus/1";
    value.corpus_bundle_hash = "identity:fnv1a64:0123456789abcdef";
    value.trainer_version = "fake-trainer-1";
    value.artifact_path = "adapters/" + id + ".gguf";
    value.artifact_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    value.evaluation_revision = "adapter-eval-1";
    value.evaluation_status = "passed";
    return value;
}

static common_learning_evaluation_report evaluation_for(
        const common_learning_adapter_manifest & adapter) {
    common_learning_evaluation_report value;
    value.revision_id = adapter.evaluation_revision;
    value.candidate_adapter_id = adapter.id;
    value.baseline_profile_id = "agent-baseline";
    value.candidate_profile_id = "agent-candidate";
    value.corpus_revision_id = adapter.corpus_revision_id;
    value.corpus_bundle_hash = adapter.corpus_bundle_hash;
    value.base_training_fingerprint = adapter.training_model_fingerprint;
    value.test_suite_revision = "agent-eval-suite-1";
    value.intended_behavior_passed = true;
    value.retention_passed = true;
    value.agent_regression_passed = true;
    value.evaluated_turns = 10;
    value.runtime_interventions = 1;
    value.status = "passed";
    value.evaluated_at = "2026-08-29T00:00:00Z";
    return value;
}

static common_agent_model_profile profile_for(const std::string & adapter_id) {
    common_agent_model_profile value;
    value.id = "agent-runtime";
    value.base_model_id = "qwen-small";
    value.base_model_fingerprint = "sha256:base-serving";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.chat_template_fingerprint = "sha256:template";
    value.context_size_tokens = 4096;
    value.adapters.push_back({adapter_id, 1.0});
    return value;
}

int main() {
    std::string error;
    common_learning_adapter_registry registry;
    auto first = manifest("adapter-v1");
    assert(registry.admit(first, error));
    assert(!registry.admit(first, error));
    assert(registry.active_id().empty());
    assert(!registry.activate(first.id, error));
    assert(error.find("canary") != std::string::npos);
    assert(registry.stage_canary(first.id, evaluation_for(first), error));
    assert(registry.activate(first.id, error));
    assert(registry.active_id() == first.id);
    std::vector<common_learning_adapter_manifest> overlays;
    assert(registry.resolve_active_overlays(profile_for(first.id), overlays, error));
    assert(overlays.size() == 1 && overlays.front().id == first.id);

    auto second = manifest("adapter-v2");
    assert(registry.admit(second, error));
    assert(registry.stage_canary(second.id, evaluation_for(second), error));
    assert(registry.activate(second.id, error));
    assert(registry.active_id() == second.id);
    assert(registry.list().front().status == common_learning_adapter_status::retired);
    auto mismatched = profile_for(second.id);
    mismatched.base_model_fingerprint = "sha256:other-serving";
    assert(!registry.resolve_active_overlays(mismatched, overlays, error));
    assert(error.find("identity") != std::string::npos);
    assert(registry.retire(second.id, error));
    assert(registry.active_id().empty());
    assert(!registry.resolve_active_overlays(common_agent_model_profile{}, overlays, error));

    auto invalid = manifest("bad");
    invalid.evaluation_status = "failed";
    assert(!registry.admit(invalid, error));

    common_learning_training_job job;
    job.id = "learning://job/1";
    job.corpus_revision_id = "learning://corpus/1";
    job.corpus_bundle_hash = "identity:fnv1a64:0123456789abcdef";
    job.base_training_fingerprint = "base:qwen-small:training-v1";
    job.trainer_version = "fake-trainer-1";
    job.code_revision = "trainer-revision-1";
    common_learning_training_result result;
    result.job_id = job.id;
    result.adapter_id = "adapter-v3";
    result.artifact_path = "adapters/adapter-v3.gguf";
    result.artifact_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    result.corpus_bundle_hash = job.corpus_bundle_hash;
    result.base_training_fingerprint = job.base_training_fingerprint;
    result.trainer_kind = job.trainer_kind;
    result.trainer_version = job.trainer_version;
    result.evaluation_revision = "adapter-eval-3";
    result.evaluation_status = "passed";
    common_learning_adapter_manifest generated;
    assert(common_learning_make_adapter_manifest(job, result, "qwen2", "serve:qwen-small",
        "tokenizer:v1", "chat:v1", generated, error));
    assert(generated.corpus_bundle_hash == job.corpus_bundle_hash);
    assert(generated.status == common_learning_adapter_status::candidate);
    assert(registry.admit(generated, error));
    auto generated_evaluation = evaluation_for(generated);
    assert(registry.stage_canary(generated.id, generated_evaluation, error));
    return 0;
}
