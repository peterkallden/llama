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
    return 0;
}
