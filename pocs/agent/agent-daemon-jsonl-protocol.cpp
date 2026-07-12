#include "agent-daemon-jsonl-protocol.h"

using json = nlohmann::ordered_json;

namespace {

const char * session_command_name(agent_daemon_jsonl_session_command command) {
    switch (command) {
        case agent_daemon_jsonl_session_command::reset: return "reset_session";
        case agent_daemon_jsonl_session_command::close: return "close_session";
    }
    return "reset_session";
}

bool parse_string_array_field(
        const json & value,
        std::vector<std::string> & output) {
    if (!value.is_array()) {
        return false;
    }
    output.clear();
    for (const auto & item : value) {
        if (!item.is_string()) {
            return false;
        }
        output.push_back(item.get<std::string>());
    }
    return true;
}

} // namespace

bool read_agent_daemon_jsonl_message(
        FILE * stream,
        json & out,
        std::string & error) {
    out = json();
    error.clear();

    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), stream) != nullptr) {
        std::string line(buffer);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        const auto parsed = json::parse(line, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            error = "daemon emitted a non-JSON protocol line: " + line;
            return false;
        }

        out = parsed;
        return true;
    }

    error = "daemon closed before returning a protocol response";
    return false;
}

bool write_agent_daemon_jsonl_message(
        FILE * stream,
        const json & message,
        std::string & error) {
    error.clear();
    const std::string line = message.dump() + "\n";
    if (std::fwrite(line.data(), 1, line.size(), stream) != line.size()) {
        error = "failed to write daemon request";
        return false;
    }
    if (std::fflush(stream) != 0) {
        error = "failed to flush daemon request";
        return false;
    }
    return true;
}

json make_agent_daemon_jsonl_turn_request(
        const agent_daemon_jsonl_turn_request & request) {
    return {
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
}

json make_agent_daemon_jsonl_status_request(
        const agent_daemon_jsonl_status_request &) {
    return {
        {"command", "status"},
    };
}

json make_agent_daemon_jsonl_shutdown_request(
        const agent_daemon_jsonl_shutdown_request &) {
    return {
        {"command", "shutdown"},
    };
}

json make_agent_daemon_jsonl_session_request(
        const agent_daemon_jsonl_session_request & request) {
    return {
        {"command", session_command_name(request.command)},
        {"session_id", request.session_id},
        {"namespace_id", request.namespace_id},
    };
}

json make_agent_daemon_jsonl_reset_session_request(
        const std::string & session_id,
        const std::string & namespace_id) {
    return make_agent_daemon_jsonl_session_request({
        agent_daemon_jsonl_session_command::reset,
        session_id,
        namespace_id,
    });
}

json make_agent_daemon_jsonl_close_session_request(
        const std::string & session_id,
        const std::string & namespace_id) {
    return make_agent_daemon_jsonl_session_request({
        agent_daemon_jsonl_session_command::close,
        session_id,
        namespace_id,
    });
}

json make_agent_daemon_jsonl_cancel_request(
        const agent_daemon_jsonl_cancel_request & request) {
    return {
        {"command", "cancel_turn"},
        {"target_request_id", request.target_request_id},
        {"target_turn_id", request.target_turn_id},
    };
}

bool parse_agent_daemon_jsonl_ready_response(
        const json & message,
        agent_daemon_jsonl_ready_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object() ||
            !message.value("ok", false) ||
            message.value("event", std::string()) != "ready") {
        error = "unexpected daemon ready response";
        return false;
    }
    if (!message.contains("default_mode") || !message["default_mode"].is_string()) {
        error = "daemon ready response is missing default_mode";
        return false;
    }
    if (!message.contains("protocol_version") || !message["protocol_version"].is_number_integer()) {
        error = "daemon ready response is missing protocol_version";
        return false;
    }
    if (!message.contains("capabilities") ||
            !parse_string_array_field(message["capabilities"], response.capabilities)) {
        error = "daemon ready response is missing capabilities";
        return false;
    }

    response.default_mode = message["default_mode"].get<std::string>();
    response.protocol_version = message["protocol_version"].get<int>();
    error.clear();
    return true;
}

bool parse_agent_daemon_jsonl_turn_response(
        const json & message,
        agent_daemon_jsonl_turn_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object()) {
        error = "unexpected daemon turn response";
        return false;
    }

    response.ok = message.value("ok", false);
    response.response = message.value("response", std::string());
    response.error = message.value("error", std::string());
    response.runtime_reused = message.value("runtime_reused", false);
    response.event_count = message.value("event_count", 0);

    if (response.ok) {
        error.clear();
        return true;
    }

    error = response.error.empty() ? "daemon turn failed" : response.error;
    return false;
}

bool parse_agent_daemon_jsonl_status_response(
        const json & message,
        agent_daemon_jsonl_status_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object()) {
        error = "unexpected daemon status response";
        return false;
    }

    response.ok = message.value("ok", false);
    response.event = message.value("event", std::string());
    response.state = message.value("state", std::string());
    response.live = message.value("live", false);
    response.ready = message.value("ready", false);
    response.worker_running = message.value("worker_running", false);
    response.accepting_commands = message.value("accepting_commands", false);
    response.shutdown_requested = message.value("shutdown_requested", false);
    response.sessions = message.value("sessions", 0);
    response.queued_commands = message.value("queued_commands", 0);
    response.max_queue_size = message.value("max_queue_size", 0);
    response.queue_capacity_remaining = message.value("queue_capacity_remaining", 0);
    response.active_request_id = message.value("active_request_id", std::string());
    response.active_turn_id = message.value("active_turn_id", std::string());
    response.payload = message;
    response.error = message.value("error", std::string());

    if (message.contains("session_keys") && message["session_keys"].is_array()) {
        for (const auto & item : message["session_keys"]) {
            if (!item.is_object()) {
                continue;
            }
            response.session_keys.push_back({
                item.value("namespace_id", std::string()),
                item.value("session_id", std::string()),
                item.value("project_id", std::string()),
                item.value("memory_scope", std::string()),
                item.value("plan_scope", std::string()),
                item.value("policy_pack_id", std::string()),
            });
        }
    }

    if (response.ok) {
        error.clear();
        return true;
    }

    error = response.error.empty() ? "daemon status failed" : response.error;
    return false;
}

bool parse_agent_daemon_jsonl_event_response(
        const json & message,
        agent_daemon_jsonl_event_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object()) {
        error = "unexpected daemon event response";
        return false;
    }

    response.ok = message.value("ok", false);
    response.event = message.value("event", std::string());
    response.error = message.value("error", std::string());

    if (response.ok && !response.event.empty()) {
        error.clear();
        return true;
    }

    error = response.error.empty() ? "daemon event failed" : response.error;
    return false;
}

bool parse_agent_daemon_jsonl_event_response(
        const json & message,
        const std::string & expected_event,
        std::string & error) {
    agent_daemon_jsonl_event_response response;
    if (!parse_agent_daemon_jsonl_event_response(message, response, error) ||
            response.event != expected_event) {
        error = "unexpected daemon " + expected_event + " response";
        return false;
    }
    error.clear();
    return true;
}
