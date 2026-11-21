#include "../../daemon/agent-daemon-flydelta-internal.h"

namespace agent_daemon_flydelta_internal {

json daemon_flydelta_capture_json(
        const common_flydelta_hidden_state_capture & capture) {
    return {
        {"kind", "flydelta_hidden_state_capture"},
        {"schema_version", capture.schema_version},
        {"captured", capture.captured},
        {"model_profile_fingerprint", capture.model_profile_fingerprint},
        {"capture_layout_revision", capture.capture_layout_revision},
        {"layer_indices", capture.layer_indices},
        {"n_embd", capture.n_embd},
        {"position", common_flydelta_capture_position_name(capture.position)},
        {"token_index", capture.token_index},
        {"values", capture.values},
        {"failure_reason", capture.failure_reason},
    };
}

bool daemon_flydelta_capture_from_json(
        const json & value,
        common_flydelta_hidden_state_capture & capture,
        std::string & error) {
    error.clear();
    try {
        capture = {};
        capture.schema_version = value.value("schema_version", 0);
        capture.captured = value.value("captured", false);
        capture.model_profile_fingerprint = value.value("model_profile_fingerprint", "");
        capture.capture_layout_revision = value.value("capture_layout_revision", "");
        capture.layer_indices = value.value("layer_indices", std::vector<uint32_t>{});
        capture.n_embd = value.value("n_embd", 0U);
        const auto position = value.value("position", "prompt_row");
        capture.position = position == "generation_boundary"
            ? common_flydelta_capture_position::generation_boundary
            : common_flydelta_capture_position::prompt_row;
        capture.token_index = value.value("token_index", -1);
        capture.values = value.value("values", std::vector<float>{});
        capture.failure_reason = value.value("failure_reason", "");
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta capture resource is malformed: ") + exception.what();
        return false;
    }
    return common_flydelta_hidden_state_capture_validate(
        capture, 16U * 1024U * 1024U, error);
}

bool daemon_flydelta_load_persisted_capture(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const std::string & arm_id,
        std::shared_ptr<const common_flydelta_hidden_state_capture> & capture) {
    if (!provider || provider->resources == nullptr || arm_id.empty()) return false;
    const std::string name = "flydelta-capture-" +
        hash_sha256_hex(arm_id.data(), arm_id.size()).substr(0, 24) + ".json";
    std::vector<agent_resource_descriptor> descriptors;
    std::string error;
    if (!provider->resources->list(provider->authority, descriptors, error)) return false;
    for (const auto & descriptor : descriptors) {
        if (descriptor.name != name || descriptor.mime_type != "application/json") continue;
        std::string text;
        if (!provider->resources->read_text(
                descriptor.uri, provider->authority, 16U * 1024U * 1024U, text, error)) {
            return false;
        }
        json value;
        try {
            value = json::parse(text);
        } catch (...) {
            return false;
        }
        common_flydelta_hidden_state_capture parsed;
        if (!daemon_flydelta_capture_from_json(value, parsed, error)) return false;
        capture = std::make_shared<const common_flydelta_hidden_state_capture>(
            std::move(parsed));
        std::lock_guard<std::mutex> lock(provider->capture_mutex);
        provider->captures[arm_id] = capture;
        return true;
    }
    return false;
}

} // namespace agent_daemon_flydelta_internal
