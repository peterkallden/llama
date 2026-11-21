#include "agent/adaptation/flydelta/flydelta-artifact-store.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#define CHECK(condition) do { if (!(condition)) { std::cerr << "check failed at line " << __LINE__ << ": " << error << '\n'; return __LINE__; } } while (false)

static common_flydelta_artifact make_artifact() {
    common_flydelta_artifact value;
    value.id = "flydelta:test:store-v1";
    value.encoder = {42, 4, 8, 2, 2};
    value.memory = {8, 1, 0.5f};
    value.weights.assign(8, 0.0f);
    value.weights[0] = 0.25f;
    value.compatibility.base_model_fingerprint = "sha256:model";
    value.compatibility.tokenizer_fingerprint = "sha256:tokenizer";
    value.compatibility.template_fingerprint = "sha256:template";
    value.compatibility.architecture = "qwen2";
    value.compatibility.inference_layout_revision = "l_out:v1";
    return value;
}

int main() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / ("flydelta-artifact-store-" + std::to_string(suffix));
    std::string error;
    common_flydelta_artifact_store store(root);
    const auto source = make_artifact();
    CHECK(store.write("candidate-v1.flyd", source, error));
    CHECK(store.write("candidate-v1.flyd", source, error));

    common_flydelta_artifact loaded;
    CHECK(store.read("candidate-v1.flyd", loaded, error));
    CHECK(loaded.id == source.id && loaded.content_hash == common_flydelta_artifact_hash(loaded));

    auto name_only = source;
    name_only.id = "flydelta:test:store-name-only-v0";
    name_only.compatibility.base_model_id = "Qwen2.5-Coder-1.5B-Instruct.gguf";
    name_only.compatibility.base_model_fingerprint.clear();
    CHECK(store.write("candidate-name-only-v0.flyd", name_only, error));
    CHECK(store.read("candidate-name-only-v0.flyd", loaded, error));
    CHECK(loaded.compatibility.base_model_id ==
        "Qwen2.5-Coder-1.5B-Instruct.gguf");

    CHECK(!store.read("../outside.flyd", loaded, error));
    CHECK(!store.read("candidate-v1.tmp.flyd.bak", loaded, error));
    CHECK(!store.write("candidate-v1.json", source, error));

    auto changed = source;
    changed.generation = 2;
    CHECK(!store.write("candidate-v1.flyd", changed, error));

    std::ofstream tamper(root / "tampered.flyd", std::ios::binary);
    tamper << common_flydelta_artifact_to_json(source);
    tamper.close();
    std::ofstream rewrite(root / "tampered.flyd", std::ios::binary | std::ios::trunc);
    rewrite << "{}";
    rewrite.close();
    CHECK(!store.read("tampered.flyd", loaded, error));
    std::error_code cleanup;
    std::filesystem::remove_all(root, cleanup);
    return 0;
}
