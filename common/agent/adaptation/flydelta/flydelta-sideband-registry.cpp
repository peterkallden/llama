#include "agent/adaptation/flydelta/flydelta-sideband-registry.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

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

bool model_identity_matches(const common_flydelta_compatibility & actual,
        const common_flydelta_compatibility & expected, std::string & error) {
    if (!actual.base_model_id.empty() || !expected.base_model_id.empty()) {
        if (actual.base_model_id != expected.base_model_id) {
            error = "FlyDelta sideband base model id is incompatible";
            return false;
        }
    } else if (actual.base_model_fingerprint != expected.base_model_fingerprint) {
        error = "FlyDelta sideband base model fingerprint is incompatible";
        return false;
    }
    if (!expected.base_model_fingerprint.empty() &&
            actual.base_model_fingerprint != expected.base_model_fingerprint) {
        error = "FlyDelta sideband base model fingerprint is incompatible";
        return false;
    }
    return true;
}

bool same_manifest(const common_flydelta_sideband_manifest & actual,
        const common_flydelta_sideband_manifest & expected) {
    // Compare the canonical manifest projection so retries are idempotent
    // without maintaining a second hand-written field-by-field comparison.
    return common_flydelta_sideband_manifest_to_json(actual) ==
        common_flydelta_sideband_manifest_to_json(expected);
}

} // namespace

const char * common_flydelta_sideband_status_name(common_flydelta_sideband_status status) {
    switch (status) {
        case common_flydelta_sideband_status::experimental: return "experimental";
        case common_flydelta_sideband_status::candidate: return "candidate";
        case common_flydelta_sideband_status::canary: return "canary";
        case common_flydelta_sideband_status::active: return "active";
        case common_flydelta_sideband_status::retired: return "retired";
        case common_flydelta_sideband_status::rejected: return "rejected";
        case common_flydelta_sideband_status::revoked: return "revoked";
    }
    return "rejected";
}

bool common_flydelta_sideband_manifest_validate(
        const common_flydelta_sideband_manifest & manifest,
        std::string & error) {
    error.clear();
    if (manifest.schema_version != 1 || !bounded(manifest.id) ||
            !bounded(manifest.artifact_path) || !hash_like(manifest.artifact_hash) ||
            !bounded(manifest.namespace_id) || !bounded(manifest.project_id) ||
            (manifest.compatibility.base_model_id.empty() &&
             manifest.compatibility.base_model_fingerprint.empty()) ||
            (!manifest.compatibility.base_model_id.empty() &&
             !bounded(manifest.compatibility.base_model_id)) ||
            (!manifest.compatibility.base_model_fingerprint.empty() &&
             !bounded(manifest.compatibility.base_model_fingerprint)) ||
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
    const bool has_behavior = !manifest.applicability.behavior_key.empty();
    const bool has_scope = !manifest.applicability.scope_fingerprint.empty();
    const bool has_verifier = !manifest.applicability.verifier_revision.empty();
    if ((has_behavior || has_scope || has_verifier) && !(has_behavior && has_scope && has_verifier)) {
        error = "FlyDelta sideband applicability identity is incomplete";
        return false;
    }
    if (manifest.status == common_flydelta_sideband_status::revoked &&
            !bounded(manifest.revocation_reason)) {
        error = "revoked FlyDelta sideband requires a reason";
        return false;
    }
    if (manifest.status == common_flydelta_sideband_status::experimental &&
            manifest.evaluation_passed) {
        error = "experimental FlyDelta sideband cannot have a passed activation evaluation";
        return false;
    }
    if (manifest.status == common_flydelta_sideband_status::active &&
            (!manifest.evaluation_passed || !bounded(manifest.evaluation_revision))) {
        error = "active FlyDelta sideband requires a passed evaluation";
        return false;
    }
    return true;
}

std::string common_flydelta_sideband_manifest_to_json(
        const common_flydelta_sideband_manifest & manifest) {
    return json{
        {"schema_version", manifest.schema_version},
        {"kind", "flydelta-sideband"},
        {"id", manifest.id},
        {"status", common_flydelta_sideband_status_name(manifest.status)},
        {"artifact_path", manifest.artifact_path},
        {"artifact_hash", manifest.artifact_hash},
        {"scope", {
            {"namespace_id", manifest.namespace_id},
            {"project_id", manifest.project_id},
        }},
        {"expires_at_epoch_ms", manifest.expires_at_epoch_ms},
        {"revocation_reason", manifest.revocation_reason},
        {"compatibility", {
            {"base_model_id", manifest.compatibility.base_model_id},
            {"base_model_fingerprint", manifest.compatibility.base_model_fingerprint},
            {"tokenizer_fingerprint", manifest.compatibility.tokenizer_fingerprint},
            {"template_fingerprint", manifest.compatibility.template_fingerprint},
            {"architecture", manifest.compatibility.architecture},
            {"inference_layout_revision", manifest.compatibility.inference_layout_revision},
        }},
        {"applicability", {
            {"behavior_key", manifest.applicability.behavior_key},
            {"scope_fingerprint", manifest.applicability.scope_fingerprint},
            {"verifier_revision", manifest.applicability.verifier_revision},
        }},
        {"model_n_embd", manifest.model_n_embd},
        {"model_n_layers", manifest.model_n_layers},
        {"il_start", manifest.il_start},
        {"il_end", manifest.il_end},
        {"evaluation_revision", manifest.evaluation_revision},
        {"evaluation_passed", manifest.evaluation_passed},
    }.dump();
}

bool common_flydelta_sideband_manifest_from_json(
        const std::string & text,
        common_flydelta_sideband_manifest & manifest,
        std::string & error) {
    error.clear();
    try {
        const auto value = json::parse(text);
        if (!value.is_object() || value.value("kind", "") != "flydelta-sideband") {
            error = "invalid FlyDelta sideband manifest JSON";
            return false;
        }
        manifest = {};
        manifest.schema_version = value.value("schema_version", 0);
        manifest.id = value.value("id", "");
        manifest.artifact_path = value.value("artifact_path", "");
        manifest.artifact_hash = value.value("artifact_hash", "");
        const auto scope = value.value("scope", json::object());
        manifest.namespace_id = scope.value("namespace_id", "");
        manifest.project_id = scope.value("project_id", "");
        manifest.expires_at_epoch_ms = value.value("expires_at_epoch_ms", 0ULL);
        manifest.revocation_reason = value.value("revocation_reason", "");
        const auto compatibility = value.value("compatibility", json::object());
        manifest.compatibility.base_model_id = compatibility.value("base_model_id", "");
        manifest.compatibility.base_model_fingerprint = compatibility.value("base_model_fingerprint", "");
        manifest.compatibility.tokenizer_fingerprint = compatibility.value("tokenizer_fingerprint", "");
        manifest.compatibility.template_fingerprint = compatibility.value("template_fingerprint", "");
        manifest.compatibility.architecture = compatibility.value("architecture", "");
        manifest.compatibility.inference_layout_revision = compatibility.value("inference_layout_revision", "");
        const auto applicability = value.value("applicability", json::object());
        manifest.applicability.behavior_key = applicability.value("behavior_key", "");
        manifest.applicability.scope_fingerprint = applicability.value("scope_fingerprint", "");
        manifest.applicability.verifier_revision = applicability.value("verifier_revision", "");
        manifest.model_n_embd = value.value("model_n_embd", 0U);
        manifest.model_n_layers = value.value("model_n_layers", 0U);
        manifest.il_start = value.value("il_start", 0);
        manifest.il_end = value.value("il_end", 0);
        manifest.evaluation_revision = value.value("evaluation_revision", "");
        manifest.evaluation_passed = value.value("evaluation_passed", false);
        const auto status = value.value("status", "candidate");
        if (status == "experimental") manifest.status = common_flydelta_sideband_status::experimental;
        else if (status == "candidate") manifest.status = common_flydelta_sideband_status::candidate;
        else if (status == "canary") manifest.status = common_flydelta_sideband_status::canary;
        else if (status == "active") manifest.status = common_flydelta_sideband_status::active;
        else if (status == "retired") manifest.status = common_flydelta_sideband_status::retired;
        else if (status == "rejected") manifest.status = common_flydelta_sideband_status::rejected;
        else if (status == "revoked") manifest.status = common_flydelta_sideband_status::revoked;
        else { error = "unknown FlyDelta sideband status"; return false; }
    } catch (const std::exception & exception) {
        error = std::string("invalid FlyDelta sideband manifest fields: ") + exception.what();
        return false;
    }
    return common_flydelta_sideband_manifest_validate(manifest, error);
}

static uint64_t current_epoch_ms() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

bool common_flydelta_sideband_registry::admit(
        const common_flydelta_sideband_manifest & manifest, std::string & error) {
    if (!common_flydelta_sideband_manifest_validate(manifest, error)) return false;
    if (manifest.status != common_flydelta_sideband_status::candidate &&
            manifest.status != common_flydelta_sideband_status::experimental) {
        error = "new FlyDelta sideband must enter registry as candidate or experimental";
        return false;
    }
    const auto existing = manifests.find(manifest.id);
    if (existing != manifests.end()) {
        if (same_manifest(existing->second, manifest)) {
            // A worker retry may reach registry admission after the immutable
            // artifact was already published. Identical metadata is a
            // successful retry; a changed manifest remains fail-closed.
            error.clear();
            return true;
        }
        error = "FlyDelta sideband is already registered with different metadata: " + manifest.id;
        return false;
    }
    manifests.emplace(manifest.id, manifest);
    return true;
}

bool common_flydelta_sideband_registry::admit_experimental(
        const common_flydelta_sideband_manifest & manifest, std::string & error) {
    if (manifest.status != common_flydelta_sideband_status::experimental) {
        error = "experimental FlyDelta sideband must be admitted with experimental status";
        return false;
    }
    return admit(manifest, error);
}

bool common_flydelta_sideband_registry::promote_experimental(
        const std::string & id, const std::string & evaluation_revision, std::string & error,
        bool explicit_host_approval) {
    if (!explicit_host_approval) {
        error = "experimental FlyDelta promotion requires explicit host approval";
        return false;
    }
    const auto it = manifests.find(id);
    if (it == manifests.end()) { error = "FlyDelta sideband is unavailable: " + id; return false; }
    if (it->second.status != common_flydelta_sideband_status::experimental) {
        error = "only an experimental FlyDelta sideband can enter candidate lifecycle";
        return false;
    }
    if (!bounded(evaluation_revision)) {
        error = "FlyDelta experimental evaluation revision is invalid";
        return false;
    }
    it->second.status = common_flydelta_sideband_status::candidate;
    it->second.evaluation_revision = evaluation_revision;
    // This is the experimental review revision, not the canary evaluation
    // required by the activation gate. stage_canary owns that gate.
    it->second.evaluation_passed = false;
    error.clear();
    return true;
}

bool common_flydelta_sideband_registry::stage_canary(
        const std::string & id, const std::string & evaluation_revision, std::string & error) {
    const auto it = manifests.find(id);
    if (it == manifests.end()) { error = "FlyDelta sideband is unavailable: " + id; return false; }
    if (it->second.status != common_flydelta_sideband_status::candidate) {
        error = "only a candidate FlyDelta sideband can enter canary";
        return false;
    }
    if (it->second.expires_at_epoch_ms != 0 && it->second.expires_at_epoch_ms <= current_epoch_ms()) {
        error = "FlyDelta sideband has expired";
        return false;
    }
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
    if (it->second.status != common_flydelta_sideband_status::active) {
        error = "only an active FlyDelta sideband can be retired";
        return false;
    }
    it->second.status = common_flydelta_sideband_status::retired;
    error.clear();
    return true;
}

bool common_flydelta_sideband_registry::revoke(
        const std::string & id, const std::string & reason, std::string & error) {
    const auto it = manifests.find(id);
    if (it == manifests.end()) { error = "FlyDelta sideband is unavailable: " + id; return false; }
    if (!bounded(reason)) { error = "FlyDelta sideband revocation reason is invalid"; return false; }
    if (it->second.status == common_flydelta_sideband_status::revoked ||
            it->second.status == common_flydelta_sideband_status::rejected) {
        error = "FlyDelta sideband cannot be revoked from its current state";
        return false;
    }
    it->second.status = common_flydelta_sideband_status::revoked;
    it->second.revocation_reason = reason;
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
    return resolve(profile, sideband_id, expected, {}, model_n_embd, model_n_layers,
        manifest, profile_scale, error);
}

bool common_flydelta_sideband_registry::resolve(
        const common_agent_model_profile & profile,
        const std::string & sideband_id,
        const common_flydelta_compatibility & expected,
        const common_flydelta_applicability & expected_applicability,
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
    if (it->second.expires_at_epoch_ms != 0 && it->second.expires_at_epoch_ms <= current_epoch_ms()) {
        error = "FlyDelta sideband has expired: " + sideband_id;
        return false;
    }
    if (it->second.model_n_embd != model_n_embd || it->second.model_n_layers != model_n_layers ||
            it->second.il_start < 1 || it->second.il_end >= static_cast<int32_t>(model_n_layers)) {
        error = "FlyDelta sideband model dimensions or layout are incompatible";
        return false;
    }
    if (!model_identity_matches(it->second.compatibility, expected, error) ||
            !same(it->second.compatibility.tokenizer_fingerprint,
            expected.tokenizer_fingerprint, "tokenizer", error) ||
            !same(it->second.compatibility.template_fingerprint,
            expected.template_fingerprint, "template", error) ||
            !same(it->second.compatibility.architecture, expected.architecture,
            "architecture", error) ||
            !same(it->second.compatibility.inference_layout_revision,
            expected.inference_layout_revision, "inference layout", error)) return false;
    const bool manifest_has_applicability = !it->second.applicability.behavior_key.empty() ||
        !it->second.applicability.scope_fingerprint.empty() ||
        !it->second.applicability.verifier_revision.empty();
    const bool expected_has_applicability = !expected_applicability.behavior_key.empty() ||
        !expected_applicability.scope_fingerprint.empty() ||
        !expected_applicability.verifier_revision.empty();
    if (manifest_has_applicability != expected_has_applicability ||
            (manifest_has_applicability &&
             (it->second.applicability.behavior_key != expected_applicability.behavior_key ||
              it->second.applicability.scope_fingerprint != expected_applicability.scope_fingerprint ||
              it->second.applicability.verifier_revision != expected_applicability.verifier_revision))) {
        error = "FlyDelta sideband applicability is incompatible or missing";
        return false;
    }
    manifest = it->second;
    profile_scale = configured->scale;
    return true;
}
