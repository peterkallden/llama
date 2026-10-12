#include "agent-learning-adapter-import.h"

#include "hash/hash.h"

#include <cassert>
#include <filesystem>
#include <fstream>

static common_learning_adapter_manifest manifest(const std::string & hash) {
    common_learning_adapter_manifest value;
    value.id = "adapter-import-test";
    value.base_architecture = "qwen2";
    value.serving_model_fingerprint = "serve:qwen-small";
    value.training_model_fingerprint = "train:qwen-small";
    value.tokenizer_fingerprint = "tokenizer:v1";
    value.chat_template_fingerprint = "chat:v1";
    value.corpus_revision_id = "learning://corpus/1";
    value.corpus_bundle_hash = "identity:fnv1a64:0123456789abcdef";
    value.trainer_version = "fake-trainer-1";
    value.artifact_path = "adapters/adapter-import-test.gguf";
    value.artifact_sha256 = hash;
    value.evaluation_revision = "eval:1";
    value.evaluation_status = "passed";
    return value;
}

int main() {
    const auto root = std::filesystem::temp_directory_path() / "llama-agent-adapter-import-test";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    std::filesystem::create_directories(root / "adapters");
    const std::string bytes = "adapter-fixture";
    std::ofstream(root / "adapters" / "adapter-import-test.gguf", std::ios::binary) << bytes;
    const auto hash = "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
    auto value = manifest(hash);
    std::string error;
    std::filesystem::path resolved;
    assert(common_learning_verify_adapter_artifact(value, root, 1024, resolved, error));
    assert(resolved.filename() == "adapter-import-test.gguf");

    value.artifact_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    assert(!common_learning_verify_adapter_artifact(value, root, 1024, resolved, error));
    assert(error.find("SHA-256") != std::string::npos);

    value = manifest(hash);
    assert(!common_learning_verify_adapter_artifact(value, root, 4, resolved, error));
    assert(error.find("byte bound") != std::string::npos);

    std::filesystem::remove_all(root, ignored);
    return 0;
}
