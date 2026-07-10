#include "agent-daemon-adapter.h"

#include "agent-cli-selection.h"

using json = nlohmann::ordered_json;

namespace {

constexpr int agent_daemon_protocol_version = 1;

std::string default_plan_scope_for_memory_scope(common_memory_scope memory_scope) {
    switch (memory_scope) {
        case common_memory_scope::turn:    return "turn";
        case common_memory_scope::session: return "session";
        case common_memory_scope::project: return "project";
        case common_memory_scope::global:  return "global";
    }
    return "session";
}

const char * daemon_plan_scope_name(common_plan_scope scope) {
    switch (scope) {
        case common_plan_scope::turn:    return "turn";
        case common_plan_scope::session: return "session";
        case common_plan_scope::project: return "project";
        case common_plan_scope::global:  return "global";
    }
    return "turn";
}

bool parse_agent_daemon_command_name(
        const json & parsed,
        common_agent_daemon_command & command,
        std::string & error) {
    const std::string command_name = parsed.value("command", "");
    if (command_name == "shutdown") {
        command.type = common_agent_daemon_command_type::shutdown;
        error.clear();
        return true;
    }
    if (command_name == "status") {
        command.type = common_agent_daemon_command_type::get_status;
        error.clear();
        return true;
    }
    if (command_name == "cancel_turn") {
        command.type = common_agent_daemon_command_type::cancel_turn;
        command.target_request_id = parsed.value("target_request_id", "");
        command.target_turn_id = parsed.value("target_turn_id", "");
        if (command.target_request_id.empty() && command.target_turn_id.empty()) {
            error = "cancel_turn requires target_request_id or target_turn_id";
            return false;
        }
        error.clear();
        return true;
    }
    if (command_name == "reset_session" || command_name == "close_session") {
        command.type = command_name == "reset_session"
            ? common_agent_daemon_command_type::reset_session
            : common_agent_daemon_command_type::close_session;
        command.session = common_agent_runtime_session_key{
            parsed.value("namespace_id", "default-namespace"),
            parsed.value("session_id", "default-session"),
        };
        error.clear();
        return true;
    }
    if (!command_name.empty() && command_name != "run_turn") {
        error = "unsupported command: " + command_name;
        return false;
    }
    command.type = common_agent_daemon_command_type::run_turn;
    error.clear();
    return true;
}

bool parse_agent_daemon_turn_request(
        const json & parsed,
        const daemon_options & options,
        common_agent_runtime_host_mode default_mode,
        common_agent_runtime_session_host_turn_request & request,
        std::string & error) {
    request = {};
    request.prompt = parsed.value("prompt", "");
    request.session_id = parsed.value("session_id", "default-session");
    request.namespace_id = parsed.value("namespace_id", "default-namespace");
    request.project_id = parsed.value("project_id", "");
    request.turn_id = parsed.value("turn_id", "");
    request.n_predict = parsed.value("n_predict", 0);
    request.mode = default_mode;
    request.memory_scope = common_memory_scope::session;
    request.plan_scope = common_plan_scope::turn;

    const std::string mode_value = parsed.value("mode", options.default_mode);
    if (!parse_mode(mode_value, request.mode)) {
        error = "unsupported mode: " + mode_value;
        return false;
    }

    const std::string memory_scope_value = parsed.value("memory_scope", "session");
    if (!common_memory_scope_parse(memory_scope_value, request.memory_scope)) {
        error = "unsupported memory_scope: " + memory_scope_value;
        return false;
    }

    const std::string plan_scope_value = parsed.value(
        "plan_scope",
        default_plan_scope_for_memory_scope(request.memory_scope));
    if (!parse_plan_scope(plan_scope_value, request.plan_scope)) {
        error = "unsupported plan_scope: " + plan_scope_value;
        return false;
    }

    error.clear();
    return true;
}

json serialize_agent_daemon_event(
        const common_agent_daemon_event & event) {
    json event_json = {
        {"type", event.type},
    };
    if (!event.request_id.empty()) {
        event_json["request_id"] = event.request_id;
    }
    if (!event.turn_id.empty()) {
        event_json["turn_id"] = event.turn_id;
    }
    if (!event.detail.empty()) {
        event_json["detail"] = event.detail;
    }
    return event_json;
}

json serialize_agent_daemon_session_status(
        const common_agent_runtime_session_descriptor & session) {
    return {
        {"namespace_id", session.key.namespace_id},
        {"session_id", session.key.session_id},
        {"project_id", session.project_id},
        {"memory_scope", common_memory_scope_name(session.memory_scope)},
        {"plan_scope", daemon_plan_scope_name(session.plan_scope)},
    };
}

json serialize_agent_daemon_trace_entry(
        const common_runtime_trace_entry & entry) {
    json entry_json = {
        {"stage", common_runtime_trace_stage_name(entry.stage)},
        {"kind", common_runtime_trace_kind_name(entry.kind)},
    };
    if (!entry.detail.empty()) {
        entry_json["detail"] = entry.detail;
    }
    if (!entry.plan_id.empty()) {
        entry_json["plan_id"] = entry.plan_id;
    }
    if (!entry.step_id.empty()) {
        entry_json["step_id"] = entry.step_id;
    }
    if (!entry.tool_name.empty()) {
        entry_json["tool_name"] = entry.tool_name;
    }
    if (!entry.observation_id.empty()) {
        entry_json["observation_id"] = entry.observation_id;
    }
    if (!entry.related_id.empty()) {
        entry_json["related_id"] = entry.related_id;
    }
    return entry_json;
}

json make_agent_daemon_base_response(
        const common_agent_daemon_command_result & result) {
    json response = {
        {"ok", result.ok},
        {"daemon_event_count", result.daemon_event_count},
    };
    if (!result.request_id.empty()) {
        response["request_id"] = result.request_id;
    }
    if (!result.event.empty()) {
        response["event"] = result.event;
    }

    json daemon_events = json::array();
    for (const auto & event : result.events) {
        daemon_events.push_back(serialize_agent_daemon_event(event));
    }
    response["events"] = std::move(daemon_events);
    return response;
}

json make_agent_daemon_status_response(
        const common_agent_daemon_command_result & result) {
    json response = make_agent_daemon_base_response(result);
    response["state"] = result.state;
    response["live"] = result.live;
    response["ready"] = result.ready;
    response["worker_running"] = result.worker_running;
    response["accepting_commands"] = result.accepting_commands;
    response["shutdown_requested"] = result.shutdown_requested;
    response["sessions"] = result.session_count;
    response["queued_commands"] = result.queued_command_count;
    response["max_queue_size"] = result.max_queue_size;
    response["queue_capacity_remaining"] = result.queue_capacity_remaining;
    if (!result.active_request_id.empty()) {
        response["active_request_id"] = result.active_request_id;
    }
    if (!result.active_turn_id.empty()) {
        response["active_turn_id"] = result.active_turn_id;
    }
    json session_array = json::array();
    for (const auto & session : result.sessions) {
        session_array.push_back(serialize_agent_daemon_session_status(session));
    }
    response["session_keys"] = std::move(session_array);
    if (!result.error.empty()) {
        response["error"] = result.error;
    }
    return response;
}

json make_agent_daemon_lifecycle_response(
        const common_agent_daemon_command_result & result) {
    json response = make_agent_daemon_base_response(result);
    if (!result.target_request_id.empty()) {
        response["target_request_id"] = result.target_request_id;
    }
    if (!result.target_turn_id.empty()) {
        response["target_turn_id"] = result.target_turn_id;
    }
    if (!result.active_request_id.empty()) {
        response["active_request_id"] = result.active_request_id;
    }
    if (!result.active_turn_id.empty()) {
        response["active_turn_id"] = result.active_turn_id;
    }
    if (!result.error.empty()) {
        response["error"] = result.error;
    }
    return response;
}

json make_agent_daemon_turn_response(
        const common_agent_daemon_command_result & result) {
    json response = make_agent_daemon_base_response(result);
    const auto & turn = result.turn_result;
    response["cancelled"] = turn.cancelled;
    response["runtime_reused"] = turn.runtime_reused;
    response["limit_reached"] = turn.limit_reached;
    response["reflected"] = turn.reflected;
    response["revised"] = turn.revised;
    response["response"] = turn.response;
    response["total_decoded_tokens"] = turn.total_decoded_tokens;
    response["event_count"] = turn.event_count;
    response["trace_count"] = turn.trace_count;
    response["memory_learning_related_count"] = turn.memory_learning_related_count;
    response["memory_learning_summary"] = turn.memory_learning_summary;
    json trace_entries = json::array();
    for (const auto & entry : turn.trace) {
        trace_entries.push_back(serialize_agent_daemon_trace_entry(entry));
    }
    response["trace"] = std::move(trace_entries);
    if (!turn.plan_id.empty()) {
        response["plan_id"] = turn.plan_id;
    }
    if (!result.error.empty()) {
        response["error"] = result.error;
    }
    return response;
}

} // namespace

bool parse_agent_daemon_command(
        const json & parsed,
        const daemon_options & options,
        common_agent_runtime_host_mode default_mode,
        common_agent_daemon_command & command,
        std::string & error) {
    command = {};
    command.request_id = parsed.value("request_id", "");

    if (!parse_agent_daemon_command_name(parsed, command, error)) {
        return false;
    }
    if (command.type == common_agent_daemon_command_type::run_turn) {
        command.turn.emplace();
        if (!parse_agent_daemon_turn_request(parsed, options, default_mode, *command.turn, error)) {
            return false;
        }
    }
    error.clear();
    return true;
}

json make_agent_daemon_ready_response(const daemon_options & options) {
    return {
        {"ok", true},
        {"event", "ready"},
        {"default_mode", options.default_mode},
        {"protocol_version", agent_daemon_protocol_version},
        {"capabilities", json::array({
            "chat",
            "mini",
            "planning",
            "reflection",
            "memory_learning",
            "scoped_sessions",
        })},
    };
}

json make_agent_daemon_error_response(const std::string & error) {
    return {
        {"ok", false},
        {"error", error},
    };
}

json make_agent_daemon_command_response(const common_agent_daemon_command_result & result) {
    if (result.event == "status") {
        return make_agent_daemon_status_response(result);
    }

    if (!result.event.empty()) {
        return make_agent_daemon_lifecycle_response(result);
    }
    return make_agent_daemon_turn_response(result);
}
