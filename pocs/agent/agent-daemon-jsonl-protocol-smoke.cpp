#include "agent-daemon-client-status.h"
#include "agent-daemon-jsonl-protocol.h"

#include <cstdio>

using json = nlohmann::ordered_json;

int main() {
    std::string error;

    agent_daemon_jsonl_turn_request turn_request_spec;
    turn_request_spec.prompt = "hello";
    turn_request_spec.session_id = "session-a";
    turn_request_spec.namespace_id = "namespace-a";
    turn_request_spec.project_id = "project-a";
    turn_request_spec.turn_id = "turn-a";
    turn_request_spec.memory_scope = "project";
    turn_request_spec.plan_scope = "project";
    turn_request_spec.n_predict = 42;
    turn_request_spec.mode = "mini";
    turn_request_spec.turn_timeout_ms = 12000;
    turn_request_spec.inference_step_timeout_ms = 345;
    turn_request_spec.tool_timeout_ms = 678;
    turn_request_spec.mcp_connect_timeout_ms = 901;
    turn_request_spec.mcp_request_timeout_ms = 2345;
    turn_request_spec.mcp_shutdown_timeout_ms = 456;
    const auto turn_request = make_agent_daemon_jsonl_turn_request(turn_request_spec);
    if (!turn_request.is_object() ||
            turn_request.value("command", "") != "run_turn" ||
            turn_request.value("prompt", "") != "hello" ||
            turn_request.value("mode", "") != "mini" ||
            turn_request.value("n_predict", 0) != 42 ||
            turn_request.value("turn_timeout_ms", 0) != 12000 ||
            turn_request.value("inference_step_timeout_ms", 0) != 345 ||
            turn_request.value("tool_timeout_ms", 0) != 678 ||
            turn_request.value("mcp_connect_timeout_ms", 0) != 901 ||
            turn_request.value("mcp_request_timeout_ms", 0) != 2345 ||
            turn_request.value("mcp_shutdown_timeout_ms", 0) != 456) {
        std::fprintf(stderr, "turn request contract mismatch\n");
        return 1;
    }

    const auto status_request = make_agent_daemon_jsonl_status_request({});
    if (!status_request.is_object() ||
            status_request.value("command", "") != "status") {
        std::fprintf(stderr, "status request contract mismatch\n");
        return 1;
    }

    const auto cancel_request = make_agent_daemon_jsonl_cancel_request({
        "req-2",
        "turn-2",
    });
    if (!cancel_request.is_object() ||
            cancel_request.value("command", "") != "cancel_turn" ||
            cancel_request.value("target_request_id", "") != "req-2" ||
            cancel_request.value("target_turn_id", "") != "turn-2") {
        std::fprintf(stderr, "cancel request contract mismatch\n");
        return 1;
    }

    const auto reset_request = make_agent_daemon_jsonl_session_request({
        agent_daemon_jsonl_session_command::reset,
        "session-a",
        "namespace-a",
    });
    if (!reset_request.is_object() ||
            reset_request.value("command", "") != "reset_session" ||
            reset_request.value("session_id", "") != "session-a" ||
            reset_request.value("namespace_id", "") != "namespace-a") {
        std::fprintf(stderr, "session request contract mismatch\n");
        return 1;
    }
    const auto close_request = make_agent_daemon_jsonl_close_session_request(
        "session-a",
        "namespace-a");
    if (!close_request.is_object() ||
            close_request.value("command", "") != "close_session") {
        std::fprintf(stderr, "close session request contract mismatch\n");
        return 1;
    }

    const auto shutdown_request = make_agent_daemon_jsonl_shutdown_request({});
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

    agent_daemon_jsonl_lifecycle_response lifecycle_response;
    if (!parse_agent_daemon_jsonl_lifecycle_response(
            {
                {"ok", true},
                {"event", "shutdown"},
                {"state", "draining"},
                {"live", true},
                {"ready", false},
                {"worker_running", true},
                {"accepting_commands", false},
                {"shutdown_requested", true},
                {"sessions", 0},
                {"queued_commands", 0},
                {"max_queue_size", 8},
                {"queue_capacity_remaining", 8},
            },
            lifecycle_response,
            error) ||
            lifecycle_response.event != "shutdown" ||
            lifecycle_response.status.state != "draining" ||
            !lifecycle_response.status.shutdown_requested) {
        std::fprintf(stderr, "lifecycle response contract mismatch: %s\n", error.c_str());
        return 1;
    }

    agent_daemon_jsonl_status_response status_response;
    if (!parse_agent_daemon_jsonl_status_response(
            {
                {"ok", true},
                {"event", "status"},
                {"state", "ready"},
                {"live", true},
                {"ready", true},
                {"worker_running", true},
                {"accepting_commands", true},
                {"shutdown_requested", false},
                {"sessions", 1},
                {"queued_commands", 0},
                {"max_queue_size", 8},
                {"queue_capacity_remaining", 8},
                {"active_request_id", "request-a"},
                {"active_turn_id", "turn-active"},
                {"active_turn_phase", "awaiting_inference"},
                {"active_turn_disposition", "continue_immediately"},
                {"session_keys", json::array({
                    {
                        {"namespace_id", "namespace-a"},
                        {"session_id", "session-a"},
                        {"project_id", "project-a"},
                        {"memory_scope", "project"},
                        {"plan_scope", "project"},
                        {"policy_pack_id", "pack-a"},
                        {"queued_turn_count", 0},
                        {"active_request_id", "request-a"},
                        {"active_turn_id", "turn-active"},
                        {"active_turn_phase", "awaiting_inference"},
                        {"active_turn_disposition", "continue_immediately"},
                        {"last_turn_id", "turn-a"},
                        {"last_turn_phase", "completed"},
                        {"last_turn_disposition", "completed"},
                    }
                })},
            },
            status_response,
            error) ||
            status_response.state != "ready" ||
            status_response.sessions != 1 ||
            status_response.active_request_id != "request-a" ||
            status_response.active_turn_id != "turn-active" ||
            status_response.active_turn_phase != "awaiting_inference" ||
            status_response.active_turn_disposition != "continue_immediately" ||
            status_response.session_keys.size() != 1 ||
            status_response.session_keys[0].policy_pack_id != "pack-a" ||
            status_response.session_keys[0].active_request_id != "request-a" ||
            status_response.session_keys[0].active_turn_id != "turn-active" ||
            status_response.session_keys[0].active_turn_phase != "awaiting_inference" ||
            status_response.session_keys[0].active_turn_disposition != "continue_immediately" ||
            status_response.session_keys[0].last_turn_id != "turn-a" ||
            status_response.session_keys[0].last_turn_phase != "completed" ||
            status_response.session_keys[0].last_turn_disposition != "completed") {
        std::fprintf(stderr, "status response contract mismatch: %s\n", error.c_str());
        return 1;
    }
    const auto summary = make_agent_daemon_client_status_summary(status_response);
    const std::string rendered_status = render_agent_daemon_client_status_summary(summary);
    if (rendered_status.find("state=ready") == std::string::npos ||
            rendered_status.find("active_turn=turn-active/awaiting_inference:continue_immediately") == std::string::npos ||
            rendered_status.find(
                "session_bindings=namespace-a/session-a@project-a#pack-a[active=turn-active/awaiting_inference:continue_immediately]") == std::string::npos) {
        std::fprintf(stderr, "status render mismatch: %s\n", rendered_status.c_str());
        return 1;
    }

    agent_daemon_jsonl_turn_response turn_response;
    parse_agent_daemon_jsonl_turn_response(
        {
            {"ok", false},
            {"cancelled", true},
            {"failure_class", "timeout"},
            {"response_generation_status", "cancelled"},
            {"response_stop_reason", "cancelled"},
            {"error", "turn deadline exceeded"},
            {"runtime_reused", true},
            {"event_count", 2},
        },
        turn_response,
        error);
    if (error != "turn deadline exceeded" ||
            !turn_response.cancelled ||
            turn_response.failure_class != "timeout" ||
            turn_response.response_generation_status != "cancelled" ||
            turn_response.response_stop_reason != "cancelled") {
        std::fprintf(stderr, "turn response contract mismatch: %s\n", error.c_str());
        return 1;
    }

    std::printf("daemon_jsonl_mode=%s\n", parsed.value("mode", "").c_str());
    std::printf("daemon_jsonl_status=%s\n", status_request.value("command", "").c_str());
    std::printf("daemon_jsonl_cancel=%s\n", cancel_request.value("command", "").c_str());
    std::printf("daemon_jsonl_close=%s\n", close_request.value("command", "").c_str());
    std::printf("daemon_jsonl_status_summary=%s\n", rendered_status.c_str());
    std::printf("daemon_jsonl_shutdown=%s\n", shutdown_request.value("command", "").c_str());
    return 0;
}
