#include "agent-daemon-jsonl-protocol.h"

#include <cstdio>

using json = nlohmann::ordered_json;

int main() {
    std::string error;

    const auto turn_request = make_agent_daemon_jsonl_turn_request({
        "hello",
        "session-a",
        "namespace-a",
        "project-a",
        "turn-a",
        "project",
        "project",
        42,
        "mini",
    });
    if (!turn_request.is_object() ||
            turn_request.value("prompt", "") != "hello" ||
            turn_request.value("mode", "") != "mini" ||
            turn_request.value("n_predict", 0) != 42) {
        std::fprintf(stderr, "turn request contract mismatch\n");
        return 1;
    }

    const auto shutdown_request = make_agent_daemon_jsonl_shutdown_request();
    if (!shutdown_request.is_object() ||
            shutdown_request.value("command", "") != "shutdown") {
        std::fprintf(stderr, "shutdown request contract mismatch\n");
        return 1;
    }

    FILE * stream = std::tmpfile();
    if (stream == nullptr) {
        std::fprintf(stderr, "tmpfile failed\n");
        return 1;
    }

    if (!write_agent_daemon_jsonl_message(stream, turn_request, error)) {
        std::fprintf(stderr, "write failed: %s\n", error.c_str());
        std::fclose(stream);
        return 1;
    }
    std::rewind(stream);

    json parsed;
    if (!read_agent_daemon_jsonl_message(stream, parsed, error)) {
        std::fprintf(stderr, "read failed: %s\n", error.c_str());
        std::fclose(stream);
        return 1;
    }
    std::fclose(stream);

    if (parsed != turn_request) {
        std::fprintf(stderr, "roundtrip mismatch\n");
        return 1;
    }

    agent_daemon_jsonl_ready_response ready;
    if (!parse_agent_daemon_jsonl_ready_response(
            {
                {"ok", true},
                {"event", "ready"},
                {"default_mode", "chat"},
                {"protocol_version", 1},
                {"capabilities", json::array({"chat", "mini"})},
            },
            ready,
            error) ||
            ready.default_mode != "chat" ||
            ready.protocol_version != 1 ||
            ready.capabilities.size() != 2) {
        std::fprintf(stderr, "ready response contract mismatch: %s\n", error.c_str());
        return 1;
    }

    if (!parse_agent_daemon_jsonl_event_response(
            {
                {"ok", true},
                {"event", "shutdown"},
            },
            "shutdown",
            error)) {
        std::fprintf(stderr, "shutdown response contract mismatch: %s\n", error.c_str());
        return 1;
    }

    std::printf("daemon_jsonl_mode=%s\n", parsed.value("mode", "").c_str());
    std::printf("daemon_jsonl_shutdown=%s\n", shutdown_request.value("command", "").c_str());
    return 0;
}
