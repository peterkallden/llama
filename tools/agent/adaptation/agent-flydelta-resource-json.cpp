#include "agent-daemon-flydelta-internal.h"

namespace agent_daemon_flydelta_internal {

bool daemon_flydelta_read_json(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        json & parsed,
        std::string & error) {
    return daemon_flydelta_read_json_bounded(
        provider, reference, 1024U * 1024U, parsed, error);
}

bool daemon_flydelta_read_json_bounded(
        const daemon_flydelta_resource_provider & provider,
        const std::string & reference,
        size_t max_bytes,
        json & parsed,
        std::string & error) {
    error.clear();
    if (provider.resources == nullptr || reference.empty() || reference.size() > 512) {
        error = "FlyDelta resource provider received an invalid reference";
        return false;
    }
    std::string text;
    if (!provider.resources->read_text(reference, provider.authority, max_bytes, text, error)) {
        error = "FlyDelta resource read failed reference=" + reference +
            " scope=" + provider.authority.namespace_id + "/" +
            provider.authority.project_id + "/" + provider.authority.session_id +
            ": " + error;
        return false;
    }
    try {
        parsed = json::parse(text);
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta referenced resource is not valid JSON: ") + exception.what();
        return false;
    }
    if (!parsed.is_object()) {
        error = "FlyDelta referenced resource must be a JSON object";
        return false;
    }
    return true;
}

} // namespace agent_daemon_flydelta_internal
