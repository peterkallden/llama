#include "agent-daemon-jsonl-protocol.h"

using json = nlohmann::ordered_json;

namespace {

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

json make_agent_daemon_jsonl_shutdown_request() {
    return {
        {"command", "shutdown"},
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

bool parse_agent_daemon_jsonl_event_response(
        const json & message,
        const std::string & expected_event,
        std::string & error) {
    if (!message.is_object() ||
            !message.value("ok", false) ||
            message.value("event", std::string()) != expected_event) {
        error = "unexpected daemon " + expected_event + " response";
        return false;
    }
    error.clear();
    return true;
}
