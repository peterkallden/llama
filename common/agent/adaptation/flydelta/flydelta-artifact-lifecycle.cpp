#include "agent/adaptation/flydelta/flydelta-artifact-lifecycle.h"

namespace {

bool bounded(const std::string & value) {
    return !value.empty() && value.size() <= 512;
}

} // namespace

bool common_flydelta_build_experimental_artifact(
        const std::string & id,
        uint64_t generation,
        const common_flydelta_encoder_config & encoder,
        const common_flydelta_memory_config & memory,
        const common_flydelta_compatibility & compatibility,
        const std::vector<float> & weights,
        size_t model_n_embd,
        size_t model_n_layers,
        int32_t il_start,
        int32_t il_end,
        const std::vector<common_flydelta_artifact_direction> & steering_basis,
        common_flydelta_artifact & artifact,
        std::string & error) {
    error.clear();
    artifact = {};
    artifact.schema_version = 2;
    artifact.id = id;
    artifact.generation = generation;
    artifact.encoder = encoder;
    artifact.memory = memory;
    artifact.compatibility = compatibility;
    artifact.weights = weights;
    artifact.model_n_embd = model_n_embd;
    artifact.model_n_layers = model_n_layers;
    artifact.il_start = il_start;
    artifact.il_end = il_end;
    artifact.steering_basis = steering_basis;
    if (!common_flydelta_artifact_validate(
            artifact, 1U << 20, 4U * 1024U * 1024U, error)) return false;
    artifact.content_hash = common_flydelta_artifact_hash(artifact);
    return true;
}

bool common_flydelta_experimental_manifest_from_artifact(
        const common_flydelta_artifact & artifact,
        const std::filesystem::path & relative_path,
        const std::string & namespace_id,
        const std::string & project_id,
        uint64_t expires_at_epoch_ms,
        common_flydelta_sideband_manifest & manifest,
        std::string & error) {
    error.clear();
    if (!common_flydelta_artifact_validate(
            artifact, 1U << 20, 4U * 1024U * 1024U, error) ||
            artifact.content_hash != common_flydelta_artifact_hash(artifact) ||
            relative_path.empty() || relative_path.is_absolute() ||
            relative_path.extension() != ".flyd" || !bounded(namespace_id) || !bounded(project_id)) {
        if (error.empty()) error = "FlyDelta experimental manifest input is invalid";
        return false;
    }
    manifest = {};
    manifest.id = artifact.id;
    manifest.status = common_flydelta_sideband_status::experimental;
    manifest.artifact_path = relative_path.generic_string();
    manifest.artifact_hash = artifact.content_hash;
    manifest.namespace_id = namespace_id;
    manifest.project_id = project_id;
    manifest.expires_at_epoch_ms = expires_at_epoch_ms;
    manifest.compatibility = artifact.compatibility;
    manifest.model_n_embd = artifact.model_n_embd;
    manifest.model_n_layers = artifact.model_n_layers;
    manifest.il_start = artifact.il_start;
    manifest.il_end = artifact.il_end;
    return common_flydelta_sideband_manifest_validate(manifest, error);
}

bool common_flydelta_persist_experimental_artifact(
        const common_flydelta_artifact_store & store,
        const std::filesystem::path & relative_path,
        common_flydelta_sideband_registry & registry,
        const std::string & namespace_id,
        const std::string & project_id,
        uint64_t expires_at_epoch_ms,
        common_flydelta_artifact & artifact,
        common_flydelta_sideband_manifest & manifest,
        std::string & error) {
    error.clear();
    if (!common_flydelta_experimental_manifest_from_artifact(
            artifact, relative_path, namespace_id, project_id,
            expires_at_epoch_ms, manifest, error) ||
            !store.write(relative_path, artifact, error)) return false;
    return registry.admit_experimental(manifest, error);
}
