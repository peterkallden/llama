#include "agent-daemon-adapter.h"

#include <cstdio>

using json = nlohmann::ordered_json;

int main() {
    daemon_options options;
    options.default_mode = "chat";

    std::string error;
    common_agent_daemon_command status_command;
    if (!parse_agent_daemon_command(
            json{{"request_id", "status-1"}, {"command", "status"}},
            options,
            common_agent_runtime_host_mode::chat,
            status_command,
            error)) {
        std::fprintf(stderr, "failed to parse status command: %s\n", error.c_str());
        return 1;
    }
    if (status_command.type != common_agent_daemon_command_type::get_status) {
        std::fprintf(stderr, "status command parsed to wrong type\n");
        return 1;
    }

    common_agent_daemon_command turn_command;
    if (!parse_agent_daemon_command(
            json{
                {"request_id", "turn-1"},
                {"command", "run_turn"},
                {"prompt", "hello"},
                {"session_id", "session-a"},
                {"namespace_id", "namespace-a"},
                {"project_id", "project-a"},
                {"turn_id", "turn-a"},
                {"mode", "mini"},
                {"memory_scope", "project"},
                {"plan_scope", "project"},
                {"n_predict", 42},
            },
            options,
            common_agent_runtime_host_mode::chat,
            turn_command,
            error)) {
        std::fprintf(stderr, "failed to parse run_turn command: %s\n", error.c_str());
        return 1;
    }
    if (turn_command.type != common_agent_daemon_command_type::run_turn ||
            !turn_command.turn.has_value() ||
            turn_command.turn->request.mode != common_agent_runtime_host_mode::mini ||
            turn_command.turn->request.memory_scope != common_memory_scope::project ||
            turn_command.turn->request.plan_scope != common_plan_scope::project ||
            turn_command.turn->request.n_predict != 42) {
        std::fprintf(stderr, "run_turn command did not preserve expected fields\n");
        return 1;
    }

    const json ready = make_agent_daemon_ready_response(options);
    if (!ready.value("ok", false) ||
            ready.value("event", "") != "ready" ||
            ready.value("protocol_version", 0) != 1) {
        std::fprintf(stderr, "ready response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result status_result;
    status_result.ok = true;
    status_result.request_id = "status-1";
    status_result.response_kind = common_agent_daemon_response_kind::status;
    status_result.event = "status";
    status_result.daemon_event_count = 1;
    status_result.events.push_back({"status.reported", "status-1", "", "ready"});
    status_result.status.state = common_agent_daemon_state::ready;
    status_result.status.live = true;
    status_result.status.ready = true;
    status_result.status.worker_running = true;
    status_result.status.accepting_commands = true;
    status_result.status.shutdown_requested = false;
    status_result.status.session_count = 1;
    status_result.status.queued_command_count = 0;
    status_result.status.max_queue_size = 8;
    status_result.status.queue_capacity_remaining = 8;
    status_result.status.sessions.push_back({
        {"namespace-a", "session-a"},
        "project-a",
        common_memory_scope::project,
        common_plan_scope::project,
    });

    const json status_response = make_agent_daemon_command_response(status_result);
    if (!status_response.value("ok", false) ||
            status_response.value("event", "") != "status" ||
            status_response.value("sessions", 0) != 1 ||
            !status_response.contains("session_keys") ||
            !status_response["session_keys"].is_array() ||
            status_response["session_keys"].size() != 1) {
        std::fprintf(stderr, "status response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result turn_result;
    turn_result.ok = true;
    turn_result.request_id = "turn-1";
    turn_result.response_kind = common_agent_daemon_response_kind::turn;
    turn_result.turn_result.response = "DONE";
    turn_result.turn_result.runtime_reused = true;
    turn_result.turn_result.trace.push_back({
        common_runtime_trace_stage::tool,
        common_runtime_trace_kind::succeeded,
        "calculator ok",
        "plan-a",
        "step-a",
        "calculator",
        "",
        "tool-call-1",
    });
    turn_result.turn_result.trace_count = turn_result.turn_result.trace.size();
    turn_result.turn_result.event_count = 3;
    turn_result.turn_result.memory_learning_summary = "none";

    const json turn_response = make_agent_daemon_command_response(turn_result);
    if (!turn_response.value("ok", false) ||
            turn_response.value("response", "") != "DONE" ||
            !turn_response.value("runtime_reused", false) ||
            !turn_response.contains("trace") ||
            !turn_response["trace"].is_array() ||
            turn_response["trace"].size() != 1 ||
            turn_response["trace"][0].value("tool_name", "") != "calculator") {
        std::fprintf(stderr, "turn response mismatch\n");
        return 1;
    }

    std::printf("daemon_protocol_ready=%s\n", ready.dump().c_str());
    std::printf("daemon_protocol_status=%s\n", status_response.dump().c_str());
    std::printf("daemon_protocol_turn=%s\n", turn_response.dump().c_str());
    return 0;
}
