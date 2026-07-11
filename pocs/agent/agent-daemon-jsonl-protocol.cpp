#include "agent-daemon-jsonl-protocol.h"

using json = nlohmann::ordered_json;

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
