#include "agent-jsonl.h"

using json = nlohmann::ordered_json;

namespace {

void add_optional_timeouts(const common_agent_jsonl_turn_request & request, json & message) {
    if (request.turn_timeout_ms.has_value()) message["turn_timeout_ms"] = *request.turn_timeout_ms;
    if (request.inference_step_timeout_ms.has_value()) message["inference_step_timeout_ms"] = *request.inference_step_timeout_ms;
    if (request.tool_timeout_ms.has_value()) message["tool_timeout_ms"] = *request.tool_timeout_ms;
    if (request.mcp_connect_timeout_ms.has_value()) message["mcp_connect_timeout_ms"] = *request.mcp_connect_timeout_ms;
    if (request.mcp_request_timeout_ms.has_value()) message["mcp_request_timeout_ms"] = *request.mcp_request_timeout_ms;
    if (request.mcp_shutdown_timeout_ms.has_value()) message["mcp_shutdown_timeout_ms"] = *request.mcp_shutdown_timeout_ms;
}

} // namespace

bool common_agent_jsonl_read_message(FILE * stream, json & out, std::string & error) {
    out = json();
    error.clear();

    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), stream) != nullptr) {
        std::string line(buffer);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line.empty()) continue;

        const auto parsed = json::parse(line, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            error = "agent JSONL stream emitted a non-JSON object line";
            return false;
        }
        out = parsed;
        return true;
    }

    error = "agent JSONL stream closed before returning a message";
    return false;
}

bool common_agent_jsonl_write_message(FILE * stream, const json & message, std::string & error) {
    error.clear();
    const std::string line = message.dump() + "\n";
    if (std::fwrite(line.data(), 1, line.size(), stream) != line.size()) {
        error = "failed to write agent JSONL message";
        return false;
    }
    if (std::fflush(stream) != 0) {
        error = "failed to flush agent JSONL message";
        return false;
    }
    return true;
}

json common_agent_jsonl_make_turn_request(const common_agent_jsonl_turn_request & request) {
    json message = {
        {"command", "run_turn"},
        {"prompt", request.prompt},
        {"session_id", request.session_id},
        {"namespace_id", request.namespace_id},
        {"project_id", request.project_id},
        {"turn_id", request.turn_id},
        {"memory_scope", request.memory_scope},
        {"plan_scope", request.plan_scope},
        {"n_predict", request.n_predict},
        {"mode", request.mode},
    };
    if (!request.resource_refs.empty()) message["resource_refs"] = request.resource_refs;
    if (request.include_summary) message["include_summary"] = true;
    add_optional_timeouts(request, message);
    return message;
}

json common_agent_jsonl_make_turn_result(const common_agent_jsonl_turn_result & result) {
    json message = {
        {"message_type", "response"},
        {"request_id", result.request_id},
        {"ok", result.ok},
        {"cancelled", result.cancelled},
        {"response", result.response},
        {"plan_id", result.plan_id},
        {"error", result.error},
        {"failure_class", result.failure_class},
        {"event_count", result.event_count},
    };
    return message;
}

json common_agent_jsonl_make_event_message(const common_agent_jsonl_event_entry & event) {
    json payload = {
        {"type", event.type},
        {"event_type", event.event_type},
        {"sequence", event.sequence},
    };
    if (!event.request_id.empty()) payload["request_id"] = event.request_id;
    if (!event.turn_id.empty()) payload["turn_id"] = event.turn_id;
    if (!event.namespace_id.empty()) payload["namespace_id"] = event.namespace_id;
    if (!event.project_id.empty()) payload["project_id"] = event.project_id;
    if (!event.session_id.empty()) payload["session_id"] = event.session_id;
    if (!event.operation_id.empty()) payload["operation_id"] = event.operation_id;
    if (!event.detail.empty()) payload["detail"] = event.detail;
    if (!event.memory_id.empty()) payload["memory_id"] = event.memory_id;
    if (!event.plan_id.empty()) payload["plan_id"] = event.plan_id;
    if (!event.step_id.empty()) payload["step_id"] = event.step_id;
    if (!event.observation_id.empty()) payload["observation_id"] = event.observation_id;
    if (!event.tool_name.empty()) payload["tool_name"] = event.tool_name;
    if (!event.resource_uri.empty()) payload["resource_uri"] = event.resource_uri;
    return {
        {"message_type", "event"},
        {"event", std::move(payload)},
    };
}
