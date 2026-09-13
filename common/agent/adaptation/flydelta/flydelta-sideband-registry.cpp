#include "agent/adaptation/flydelta/flydelta-sideband-registry.h"

#include <algorithm>
#include <cmath>

namespace {

bool bounded(const std::string & value, size_t max = 512) {
    return !value.empty() && value.size() <= max;
}
bool hash_like(const std::string & value) {
    return value.size() >= 8 && value.size() <= 256 && value.find(':') != std::string::npos;
}

bool same(const std::string & actual, const std::string & expected, const char * name,
        std::string & error) {
    if (actual != expected) {
        error = std::string("FlyDelta sideband ") + name + " is incompatible";
        return false;
    }
    return true;
}

} // namespace

const char * common_flydelta_sideband_status_name(common_flydelta_sideband_status status) {
    switch (status) {
        case common_flydelta_sideband_status::candidate: return "candidate";
        case common_flydelta_sideband_status::canary: return "canary";
        case common_flydelta_sideband_status::active: return "active";
        case common_flydelta_sideband_status::retired: return "retired";
        case common_flydelta_sideband_status::rejected: return "rejected";
    }
    return "rejected";
}

bool common_flydelta_sideband_manifest_validate(
        const common_flydelta_sideband_manifest & manifest,
        std::string & error) {
    error.clear();
    if (manifest.schema_version != 1 || !bounded(manifest.id) ||
            !bounded(manifest.artifact_path) || !hash_like(manifest.artifact_hash) ||
            !bounded(manifest.compatibility.base_model_fingerprint) ||
            !bounded(manifest.compatibility.tokenizer_fingerprint) ||
            !bounded(manifest.compatibility.template_fingerprint) ||
            !bounded(manifest.compatibility.architecture) ||
            !bounded(manifest.compatibility.inference_layout_revision) ||
            manifest.model_n_embd == 0 || manifest.model_n_layers < 2 ||
            manifest.il_start < 1 || manifest.il_end < manifest.il_start ||
            static_cast<size_t>(manifest.il_end) >= manifest.model_n_layers) {
        error = "FlyDelta sideband manifest identity or layout is invalid";
        return false;
    }
    if (manifest.status == common_flydelta_sideband_status::active &&
            (!manifest.evaluation_passed || !bounded(manifest.evaluation_revision))) {
        error = "active FlyDelta sideband requires a passed evaluation";
        return false;
    }
    return true;
}

bool common_flydelta_sideband_registry::admit(
        const common_flydelta_sideband_manifest & manifest, std::string & error) {
    if (!common_flydelta_sideband_manifest_validate(manifest, error)) return false;
    if (manifests.find(manifest.id) != manifests.end()) {
        error = "FlyDelta sideband is already registered: " + manifest.id;
        return false;
    }
    manifests.emplace(manifest.id, manifest);
    return true;
}

bool common_flydelta_sideband_registry::stage_canary(
        const std::string & id, const std::string & evaluation_revision, std::string & error) {
    const auto it = manifests.find(id);
    if (it == manifests.end()) { error = "FlyDelta sideband is unavailable: " + id; return false; }
    if (!bounded(evaluation_revision)) { error = "FlyDelta sideband evaluation revision is invalid"; return false; }
    it->second.evaluation_revision = evaluation_revision;
    it->second.evaluation_passed = true;
    it->second.status = common_flydelta_sideband_status::canary;
    error.clear();
    return true;
}

bool common_flydelta_sideband_registry::activate(const std::string & id, std::string & error) {
    const auto it = manifests.find(id);
    if (it == manifests.end()) { error = "FlyDelta sideband is unavailable: " + id; return false; }
    if (it->second.status != common_flydelta_sideband_status::canary ||
            !it->second.evaluation_passed) {
        error = "FlyDelta sideband must pass canary evaluation before activation";
        return false;
    }
    it->second.status = common_flydelta_sideband_status::active;
    error.clear();
    return true;
}

bool common_flydelta_sideband_registry::retire(const std::string & id, std::string & error) {
    const auto it = manifests.find(id);
    if (it == manifests.end()) { error = "FlyDelta sideband is unavailable: " + id; return false; }
    it->second.status = common_flydelta_sideband_status::retired;
    error.clear();
    return true;
}

bool common_flydelta_sideband_registry::resolve(
        const common_agent_model_profile & profile,
        const std::string & sideband_id,
        const common_flydelta_compatibility & expected,
        size_t model_n_embd,
        size_t model_n_layers,
        common_flydelta_sideband_manifest & manifest,
        double & profile_scale,
        std::string & error) const {
    error.clear();
    if (!common_agent_validate_model_profile(profile, error)) return false;
    const auto configured = std::find_if(profile.sidebands.begin(), profile.sidebands.end(),
        [&](const auto & sideband) { return sideband.sideband_id == sideband_id; });
    if (configured == profile.sidebands.end()) {
        error = "FlyDelta sideband is not configured in model profile: " + sideband_id;
        return false;
    }
    const auto it = manifests.find(sideband_id);
    if (it == manifests.end()) { error = "FlyDelta sideband is not registered: " + sideband_id; return false; }
    if (it->second.status != common_flydelta_sideband_status::active) {
        error = "FlyDelta sideband is not active: " + sideband_id;
        return false;
    }
    if (it->second.model_n_embd != model_n_embd || it->second.model_n_layers != model_n_layers ||
            it->second.il_start < 1 || it->second.il_end >= static_cast<int32_t>(model_n_layers)) {
        error = "FlyDelta sideband model dimensions or layout are incompatible";
        return false;
    }
    if (!same(it->second.compatibility.base_model_fingerprint,
            expected.base_model_fingerprint, "base model", error) ||
            !same(it->second.compatibility.tokenizer_fingerprint,
            expected.tokenizer_fingerprint, "tokenizer", error) ||
            !same(it->second.compatibility.template_fingerprint,
            expected.template_fingerprint, "template", error) ||
            !same(it->second.compatibility.architecture, expected.architecture,
            "architecture", error) ||
            !same(it->second.compatibility.inference_layout_revision,
            expected.inference_layout_revision, "inference layout", error)) return false;
    manifest = it->second;
    profile_scale = configured->scale;
    return true;
}
