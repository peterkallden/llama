#include "agent/adaptation/flydelta/flydelta-artifact-lifecycle.h"

#include <chrono>
#include <filesystem>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_flydelta_encoder_config encoder = {42, 2, 4, 1, 2};
    common_flydelta_memory_config memory = {4, 2, 1.0f};
    common_flydelta_compatibility compatibility;
    compatibility.base_model_fingerprint = "sha256:model";
    compatibility.tokenizer_fingerprint = "sha256:tokenizer";
    compatibility.template_fingerprint = "sha256:template";
    compatibility.architecture = "qwen2";
    compatibility.inference_layout_revision = "layout:v1";
    std::vector<common_flydelta_artifact_direction> basis = {
        {2, {1.0f, 0.0f}}, {2, {0.0f, 1.0f}},
    };
    common_flydelta_artifact artifact;
    CHECK(common_flydelta_build_experimental_artifact(
        "flydelta://sideband/experiment-lifecycle", 3, encoder, memory,
        compatibility, std::vector<float>(8, 0.0f), 2, 4, 1, 3, basis,
        artifact, error));
    CHECK(artifact.schema_version == 2 && !artifact.content_hash.empty());
    common_flydelta_sideband_manifest manifest;
    CHECK(common_flydelta_experimental_manifest_from_artifact(
        artifact, "experiments/lifecycle.flyd", "local", "project", 0,
        manifest, error));
    CHECK(manifest.status == common_flydelta_sideband_status::experimental);
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() /
        ("flydelta-artifact-lifecycle-" + std::to_string(suffix));
    common_flydelta_artifact_store store(root);
    common_flydelta_sideband_registry registry;
    CHECK(common_flydelta_persist_experimental_artifact(
        store, "experiments/lifecycle.flyd", registry, "local", "project", 0,
        artifact, manifest, error));
    CHECK(common_flydelta_persist_experimental_artifact(
        store, "experiments/lifecycle.flyd", registry, "local", "project", 0,
        artifact, manifest, error));
    CHECK(registry.list().at(artifact.id).status == common_flydelta_sideband_status::experimental);
    common_flydelta_artifact loaded;
    CHECK(store.read("experiments/lifecycle.flyd", loaded, error));
    CHECK(loaded.content_hash == artifact.content_hash);
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    return 0;
}
