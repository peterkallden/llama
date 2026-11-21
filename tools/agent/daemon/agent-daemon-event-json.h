#pragma once

#include "agent-daemon-events.h"
#include <nlohmann/json.hpp>

// HTTP command results and streamed JSONL/SSE deliveries use one projection.
inline nlohmann::ordered_json common_agent_daemon_event_to_json(const common_agent_daemon_event & event) {
    using json = nlohmann::ordered_json;
    json out = {{"type", event.type}};
    if (event.sequence) out["sequence"] = event.sequence;
    if (event.event_type != common_agent_daemon_event_type::unknown) {
        out["event_type"] = common_agent_daemon_event_type_name(event.event_type);
        out["event_category"] = common_agent_daemon_event_category_name(event.category);
    }
    const auto field = [&](const char * key, const std::string & value) { if (!value.empty()) out[key] = value; };
    field("request_id", event.request_id); field("turn_id", event.turn_id);
    field("namespace_id", event.namespace_id); field("project_id", event.project_id);
    field("session_id", event.session_id); field("operation_id", event.operation_id);
    field("detail", event.detail);
    if (!event.resource_uri.empty()) {
        out["resource_uri"] = event.resource_uri;
        out["resource_name"] = event.resource_name;
        out["resource_mime_type"] = event.resource_mime_type;
    }
    if (event.app_context_available) {
        out["app_context_available"] = true;
        field("provider_id", event.provider_id); field("server_tool_name", event.server_tool_name);
        const auto input = json::parse(event.tool_input_json, nullptr, false);
        const auto result = json::parse(event.tool_result_json, nullptr, false);
        if (input.is_object()) out["tool_input"] = input;
        if (result.is_object()) out["tool_result"] = result;
    }
    if (!event.visualization_json.empty()) {
        const auto visualization = json::parse(event.visualization_json, nullptr, false);
        if (visualization.is_object()) out["visualization"] = visualization;
    }
    return out;
}
