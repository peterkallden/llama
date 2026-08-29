#include "agent/adaptation/adapter-registry.h"

#include <algorithm>
#include <cctype>
#include <filesystem>

static bool nonempty(const std::string & value) { return !value.empty() && value.size() <= 512; }
static bool sha256_shape(const std::string & value) {
    if (value.rfind("sha256:", 0) != 0 || value.size() != 71) return false;
    for (size_t i = 7; i < value.size(); ++i) if (!std::isxdigit(static_cast<unsigned char>(value[i]))) return false;
    return true;
}

static bool safe_artifact_path(const std::string & value) {
    const std::filesystem::path path(value);
    return !path.empty() && !path.is_absolute() && path.lexically_normal() == path &&
        value.find("..") == std::string::npos;
}

bool common_learning_make_adapter_manifest(
        const common_learning_training_job & job,
        const common_learning_training_result & result,
        const std::string & base_architecture,
        const std::string & serving_model_fingerprint,
        const std::string & tokenizer_fingerprint,
        const std::string & chat_template_fingerprint,
        common_learning_adapter_manifest & manifest,
        std::string & error) {
    error.clear();
    if (!common_learning_validate_training_result(job, result, error)) return false;
    if (!nonempty(base_architecture) || !nonempty(serving_model_fingerprint) ||
            !nonempty(tokenizer_fingerprint) || !nonempty(chat_template_fingerprint)) {
        error = "adapter manifest requires serving identity";
        return false;
    }
    manifest = {};
    manifest.id = result.adapter_id;
    manifest.base_architecture = base_architecture;
    manifest.serving_model_fingerprint = serving_model_fingerprint;
    manifest.training_model_fingerprint = job.base_training_fingerprint;
    manifest.tokenizer_fingerprint = tokenizer_fingerprint;
    manifest.chat_template_fingerprint = chat_template_fingerprint;
    manifest.corpus_revision_id = job.corpus_revision_id;
    manifest.corpus_bundle_hash = job.corpus_bundle_hash;
    manifest.trainer_version = job.trainer_version;
    manifest.artifact_path = result.artifact_path;
    manifest.artifact_sha256 = result.artifact_sha256;
    manifest.evaluation_revision = result.evaluation_revision;
    manifest.evaluation_status = result.evaluation_status;
    return common_learning_validate_adapter_manifest(manifest, error);
}

const char * common_learning_adapter_status_name(common_learning_adapter_status status) {
    switch (status) {
        case common_learning_adapter_status::candidate: return "candidate";
        case common_learning_adapter_status::active: return "active";
        case common_learning_adapter_status::retired: return "retired";
        case common_learning_adapter_status::rejected: return "rejected";
    }
    return "rejected";
}

bool common_learning_validate_adapter_manifest(const common_learning_adapter_manifest & manifest, std::string & error) {
    error.clear();
    if (manifest.schema_version != 1) { error = "unsupported adapter manifest schema"; return false; }
    if (!nonempty(manifest.id) || !nonempty(manifest.base_architecture) ||
            !nonempty(manifest.serving_model_fingerprint) || !nonempty(manifest.training_model_fingerprint) ||
            !nonempty(manifest.tokenizer_fingerprint) || !nonempty(manifest.chat_template_fingerprint) ||
            !nonempty(manifest.corpus_revision_id) || !nonempty(manifest.corpus_bundle_hash) ||
            !nonempty(manifest.trainer_version) || !nonempty(manifest.artifact_path) ||
            !sha256_shape(manifest.artifact_sha256) || !nonempty(manifest.evaluation_revision) ||
            manifest.evaluation_status != "passed") {
        error = "adapter manifest is incomplete or unevaluated";
        return false;
    }
    if (!safe_artifact_path(manifest.artifact_path)) {
        error = "adapter artifact path is unsafe";
        return false;
    }
    if (manifest.status == common_learning_adapter_status::active) {
        error = "new adapter must enter registry as candidate";
        return false;
    }
    return true;
}

bool common_learning_adapter_registry::admit(const common_learning_adapter_manifest & manifest, std::string & error) {
    if (!common_learning_validate_adapter_manifest(manifest, error)) return false;
    if (std::any_of(manifests.begin(), manifests.end(), [&](const auto & item) { return item.id == manifest.id; })) {
        error = "adapter id is already registered";
        return false;
    }
    manifests.push_back(manifest);
    return true;
}

bool common_learning_adapter_registry::activate(const std::string & id, std::string & error) {
    error.clear();
    auto selected = std::find_if(manifests.begin(), manifests.end(), [&](const auto & item) { return item.id == id; });
    if (selected == manifests.end()) { error = "adapter is not registered"; return false; }
    if (selected->status != common_learning_adapter_status::candidate) { error = "only a candidate adapter can be activated"; return false; }
    for (auto & item : manifests) if (item.status == common_learning_adapter_status::active) item.status = common_learning_adapter_status::retired;
    selected->status = common_learning_adapter_status::active;
    return true;
}

bool common_learning_adapter_registry::retire(const std::string & id, std::string & error) {
    error.clear();
    auto selected = std::find_if(manifests.begin(), manifests.end(), [&](const auto & item) { return item.id == id; });
    if (selected == manifests.end()) { error = "adapter is not registered"; return false; }
    if (selected->status != common_learning_adapter_status::active) { error = "only an active adapter can be retired"; return false; }
    selected->status = common_learning_adapter_status::retired;
    return true;
}

bool common_learning_adapter_registry::reject(const std::string & id, std::string & error) {
    error.clear();
    auto selected = std::find_if(manifests.begin(), manifests.end(), [&](const auto & item) { return item.id == id; });
    if (selected == manifests.end()) { error = "adapter is not registered"; return false; }
    if (selected->status != common_learning_adapter_status::candidate) { error = "only a candidate adapter can be rejected"; return false; }
    selected->status = common_learning_adapter_status::rejected;
    return true;
}

std::vector<common_learning_adapter_manifest> common_learning_adapter_registry::list() const { return manifests; }

std::string common_learning_adapter_registry::active_id() const {
    const auto active = std::find_if(manifests.begin(), manifests.end(), [](const auto & item) { return item.status == common_learning_adapter_status::active; });
    return active == manifests.end() ? std::string{} : active->id;
}
