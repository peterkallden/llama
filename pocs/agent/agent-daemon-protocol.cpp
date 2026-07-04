#include "agent-daemon-adapter.h"

#include "agent-cli-selection.h"

using json = nlohmann::ordered_json;

namespace {

std::string default_plan_scope_for_memory_scope(common_memory_scope memory_scope) {
    switch (memory_scope) {
        case common_memory_scope::turn:    return "turn";
        case common_memory_scope::session: return "session";
        case common_memory_scope::project: return "project";
        case common_memory_scope::global:  return "global";
    }
    return "session";
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
            parsed.value("project_id", ""),
        };
        error.clear();
        return true;
    }
    if (!command_name.empty() && command_name != "run_turn") {
        error = "unsupported command: " + command_name;
        return false;
    }

    command.type = common_agent_daemon_command_type::run_turn;
    command.turn.emplace();
    auto & request = *command.turn;
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

json make_agent_daemon_ready_response(const daemon_options & options) {
    return {
        {"ok", true},
        {"event", "ready"},
        {"default_mode", options.default_mode},
        {"protocol_version", 1},
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
    json response = {
        {"ok", result.ok},
    };
    if (!result.request_id.empty()) {
        response["request_id"] = result.request_id;
    }
    if (!result.event.empty()) {
        response["event"] = result.event;
    }

    if (result.event == "status") {
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
            session_array.push_back({
                {"namespace_id", session.namespace_id},
                {"session_id", session.session_id},
                {"project_id", session.project_id},
            });
        }
        response["session_keys"] = std::move(session_array);
        if (!result.error.empty()) {
            response["error"] = result.error;
        }
        return response;
    }

    if (!result.event.empty()) {
        if (!result.target_request_id.empty()) {
            response["target_request_id"] = result.target_request_id;
        }
        if (!result.target_turn_id.empty()) {
            response["target_turn_id"] = result.target_turn_id;
        }
        if (!result.error.empty()) {
            response["error"] = result.error;
        }
        return response;
    }

    const auto & turn = result.turn_result;
    response["cancelled"] = turn.cancelled;
    response["runtime_reused"] = turn.runtime_reused;
    response["limit_reached"] = turn.limit_reached;
    response["reflected"] = turn.reflected;
    response["revised"] = turn.revised;
    response["response"] = turn.response;
    response["total_decoded_tokens"] = turn.total_decoded_tokens;
    response["event_count"] = turn.event_count;
    response["memory_learning_related_count"] = turn.memory_learning_related_count;
    response["memory_learning_summary"] = turn.memory_learning_summary;
    if (!turn.plan_id.empty()) {
        response["plan_id"] = turn.plan_id;
    }
    if (!result.error.empty()) {
        response["error"] = result.error;
    }

    return response;
}
