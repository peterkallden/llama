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

int main() {
    std::string error;
    common_learning_adapter_registry registry;
    auto first = manifest("adapter-v1");
    assert(registry.admit(first, error));
    assert(!registry.admit(first, error));
    assert(registry.active_id().empty());
    assert(registry.activate(first.id, error));
    assert(registry.active_id() == first.id);

    auto second = manifest("adapter-v2");
    assert(registry.admit(second, error));
    assert(registry.activate(second.id, error));
    assert(registry.active_id() == second.id);
    assert(registry.list().front().status == common_learning_adapter_status::retired);
    assert(registry.retire(second.id, error));
    assert(registry.active_id().empty());

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
    return 0;
}
