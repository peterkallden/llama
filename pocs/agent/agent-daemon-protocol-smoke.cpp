#include "agent-daemon-adapter.h"

#include <chrono>
#include <cstdio>

using json = nlohmann::ordered_json;

int main() {
    daemon_options options;
    options.default_mode = "chat";
    options.turn_timeout_ms = 7000;
    options.inference_step_timeout_ms = 222;
    options.tool_timeout_ms = 333;
    options.mcp_connect_timeout_ms = 444;
    options.mcp_request_timeout_ms = 555;
    options.mcp_shutdown_timeout_ms = 666;

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

    common_agent_daemon_command list_sessions_command;
    if (!parse_agent_daemon_command(
            json{{"request_id", "sessions-1"}, {"command", "list_sessions"}},
            options,
            common_agent_runtime_host_mode::chat,
            list_sessions_command,
            error) ||
            list_sessions_command.type != common_agent_daemon_command_type::list_sessions) {
        std::fprintf(stderr, "failed to parse list_sessions command: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command get_session_command;
    if (!parse_agent_daemon_command(
            json{
                {"request_id", "session-1"},
                {"command", "get_session"},
                {"session_id", "session-a"},
                {"namespace_id", "namespace-a"},
            },
            options,
            common_agent_runtime_host_mode::chat,
            get_session_command,
            error) ||
            get_session_command.type != common_agent_daemon_command_type::get_session ||
            !get_session_command.session.has_value() ||
            get_session_command.session->key.session_id != "session-a") {
        std::fprintf(stderr, "failed to parse get_session command: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command list_resources_command;
    if (!parse_agent_daemon_command(
            json{
                {"request_id", "resources-1"},
                {"command", "list_resources"},
                {"session_id", "session-a"},
                {"namespace_id", "namespace-a"},
                {"project_id", "project-a"},
                {"turn_id", "turn-a"},
            },
            options,
            common_agent_runtime_host_mode::chat,
            list_resources_command,
            error) ||
            list_resources_command.type != common_agent_daemon_command_type::list_resources ||
            !list_resources_command.scope.has_value() ||
            list_resources_command.scope->authority.project_id != "project-a") {
        std::fprintf(stderr, "failed to parse list_resources command: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command list_memories_command = list_resources_command;
    if (!parse_agent_daemon_command(
            json{
                {"request_id", "memories-1"},
                {"command", "list_memories"},
                {"session_id", "session-a"},
                {"namespace_id", "namespace-a"},
            },
            options,
            common_agent_runtime_host_mode::chat,
            list_memories_command,
            error) ||
            list_memories_command.type != common_agent_daemon_command_type::list_memories ||
            !list_memories_command.scope.has_value()) {
        std::fprintf(stderr, "failed to parse list_memories command: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command list_plans_command = list_resources_command;
    if (!parse_agent_daemon_command(
            json{
                {"request_id", "plans-1"},
                {"command", "list_plans"},
                {"session_id", "session-a"},
                {"namespace_id", "namespace-a"},
            },
            options,
            common_agent_runtime_host_mode::chat,
            list_plans_command,
            error) ||
            list_plans_command.type != common_agent_daemon_command_type::list_plans ||
            !list_plans_command.scope.has_value()) {
        std::fprintf(stderr, "failed to parse list_plans command: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command read_resource_command;
    if (!parse_agent_daemon_command(
            json{
                {"request_id", "resource-1"},
                {"command", "read_resource"},
                {"uri", "agent-resource://resource/r-1"},
                {"session_id", "session-a"},
                {"namespace_id", "namespace-a"},
                {"project_id", "project-a"},
                {"turn_id", "turn-a"},
                {"max_bytes", 2048},
            },
            options,
            common_agent_runtime_host_mode::chat,
            read_resource_command,
            error) ||
            read_resource_command.type != common_agent_daemon_command_type::read_resource ||
            !read_resource_command.resource.has_value() ||
            read_resource_command.resource->uri != "agent-resource://resource/r-1" ||
            read_resource_command.resource->authority.project_id != "project-a" ||
            read_resource_command.resource->max_bytes != 2048) {
        std::fprintf(stderr, "failed to parse read_resource command: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command drain_command;
    if (!parse_agent_daemon_command(
            json{{"request_id", "drain-1"}, {"command", "drain"}},
            options,
            common_agent_runtime_host_mode::chat,
            drain_command,
            error) ||
            drain_command.type != common_agent_daemon_command_type::drain) {
        std::fprintf(stderr, "failed to parse drain command: %s\n", error.c_str());
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
                {"turn_timeout_ms", 12000},
                {"inference_step_timeout_ms", 777},
                {"tool_timeout_ms", 888},
                {"mcp_connect_timeout_ms", 999},
                {"mcp_request_timeout_ms", 1111},
                {"mcp_shutdown_timeout_ms", 2222},
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
            turn_command.turn->request.request_id != "turn-1" ||
            turn_command.turn->request.turn.mode != common_agent_runtime_host_mode::mini ||
            turn_command.turn->request.turn.memory_scope != common_memory_scope::project ||
            turn_command.turn->request.turn.plan_scope != common_plan_scope::project ||
            turn_command.turn->request.turn.n_predict != 42 ||
            turn_command.turn->request.turn.execution_control.timeout_policy.turn_timeout_ms != 12000 ||
            turn_command.turn->request.turn.execution_control.timeout_policy.inference_step_timeout_ms != 777 ||
            turn_command.turn->request.turn.execution_control.timeout_policy.tool_timeout_ms != 888 ||
            turn_command.turn->request.turn.execution_control.timeout_policy.mcp_connect_timeout_ms != 999 ||
            turn_command.turn->request.turn.execution_control.timeout_policy.mcp_request_timeout_ms != 1111 ||
            turn_command.turn->request.turn.execution_control.timeout_policy.mcp_shutdown_timeout_ms != 2222 ||
            !turn_command.turn->request.turn.execution_control.has_deadline()) {
        std::fprintf(stderr, "run_turn command did not preserve expected fields\n");
        return 1;
    }

    common_agent_daemon_command default_timeout_command;
    if (!parse_agent_daemon_command(
            json{
                {"request_id", "turn-2"},
                {"command", "run_turn"},
                {"prompt", "hello again"},
            },
            options,
            common_agent_runtime_host_mode::chat,
            default_timeout_command,
            error)) {
        std::fprintf(stderr, "failed to parse default-timeout run_turn command: %s\n", error.c_str());
        return 1;
    }
    if (!default_timeout_command.turn.has_value() ||
            default_timeout_command.turn->request.turn.execution_control.timeout_policy.turn_timeout_ms != 7000 ||
            default_timeout_command.turn->request.turn.execution_control.timeout_policy.inference_step_timeout_ms != 222 ||
            default_timeout_command.turn->request.turn.execution_control.timeout_policy.tool_timeout_ms != 333 ||
            default_timeout_command.turn->request.turn.execution_control.timeout_policy.mcp_connect_timeout_ms != 444 ||
            default_timeout_command.turn->request.turn.execution_control.timeout_policy.mcp_request_timeout_ms != 555 ||
            default_timeout_command.turn->request.turn.execution_control.timeout_policy.mcp_shutdown_timeout_ms != 666) {
        std::fprintf(stderr, "default daemon timeout policy was not preserved\n");
        return 1;
    }

    common_agent_daemon_service missing_payload_service({});
    common_agent_daemon_command missing_payload_command;
    missing_payload_command.request_id = "turn-missing";
    missing_payload_command.type = common_agent_daemon_command_type::run_turn;
    common_agent_daemon_command_result missing_payload_result;
    if (missing_payload_service.execute(missing_payload_command, missing_payload_result, error) ||
            missing_payload_result.event != "turn_failed" ||
            missing_payload_result.turn_result.error != "run_turn command missing turn payload" ||
            missing_payload_result.turn_result.cancelled ||
            missing_payload_result.turn_result.failure_class != common_agent_failure_class::execution ||
            missing_payload_result.turn_result.response_generation_status != common_agent_generation_status::errored ||
            missing_payload_result.turn_result.response_stop_reason != common_agent_generation_stop_reason::error) {
        std::fprintf(stderr, "missing-payload service failure did not preserve expected turn metadata\n");
        return 1;
    }

    common_agent_daemon_service rejected_turn_service({});
    common_agent_daemon_command rejected_turn_command = turn_command;
    rejected_turn_command.request_id = "turn-rejected";
    rejected_turn_command.turn->request.request_id = "turn-rejected";
    rejected_turn_command.turn->request.turn.execution_control.deadline =
        std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    common_agent_daemon_command_result rejected_turn_result;
    if (rejected_turn_service.execute(rejected_turn_command, rejected_turn_result, error) ||
            rejected_turn_result.event != "turn_rejected" ||
            rejected_turn_result.turn_result.error != "daemon is not accepting new turns" ||
            !rejected_turn_result.turn_result.cancelled ||
            rejected_turn_result.turn_result.failure_class != common_agent_failure_class::timeout ||
            rejected_turn_result.turn_result.response_generation_status != common_agent_generation_status::cancelled ||
            rejected_turn_result.turn_result.response_stop_reason != common_agent_generation_stop_reason::cancelled) {
        std::fprintf(stderr, "rejected-turn service failure did not preserve expected timeout metadata\n");
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
    status_result.status.active_request_id = "request-a";
    status_result.status.active_turn_id = "turn-active";
    status_result.status.active_turn_phase = "awaiting_inference";
    status_result.status.active_turn_disposition = "continue_immediately";
    common_agent_runtime_session_descriptor session_descriptor;
    session_descriptor.key = {"namespace-a", "session-a"};
    session_descriptor.project_id = "project-a";
    session_descriptor.memory_scope = common_memory_scope::project;
    session_descriptor.plan_scope = common_plan_scope::project;
    session_descriptor.lane_state = "running";
    session_descriptor.queued_turn_count = 0;
    session_descriptor.has_active_turn = true;
    session_descriptor.active_request_id = "request-a";
    session_descriptor.active_turn_id = "turn-active";
    session_descriptor.active_turn_phase = "awaiting_inference";
    session_descriptor.active_turn_disposition = "continue_immediately";
    session_descriptor.last_turn_id = "turn-a";
    session_descriptor.last_turn_phase = "completed";
    session_descriptor.last_turn_disposition = "completed";
    status_result.status.sessions.push_back(std::move(session_descriptor));

    const json status_response = make_agent_daemon_command_response(status_result);
    if (!status_response.value("ok", false) ||
            status_response.value("event", "") != "status" ||
            status_response.value("sessions", 0) != 1 ||
            status_response.value("active_request_id", "") != "request-a" ||
            status_response.value("active_turn_id", "") != "turn-active" ||
            status_response.value("active_turn_phase", "") != "awaiting_inference" ||
            status_response.value("active_turn_disposition", "") != "continue_immediately" ||
            !status_response.contains("session_keys") ||
            !status_response["session_keys"].is_array() ||
            status_response["session_keys"].size() != 1 ||
            status_response["session_keys"][0].value("lane_state", "") != "running" ||
            status_response["session_keys"][0].value("active_request_id", "") != "request-a" ||
            status_response["session_keys"][0].value("active_turn_id", "") != "turn-active" ||
            status_response["session_keys"][0].value("active_turn_phase", "") != "awaiting_inference" ||
            status_response["session_keys"][0].value("active_turn_disposition", "") != "continue_immediately" ||
            status_response["session_keys"][0].value("last_turn_id", "") != "turn-a" ||
            status_response["session_keys"][0].value("last_turn_phase", "") != "completed" ||
            status_response["session_keys"][0].value("last_turn_disposition", "") != "completed") {
        std::fprintf(stderr, "status response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result list_sessions_result = status_result;
    list_sessions_result.request_id = "sessions-1";
    list_sessions_result.event = "sessions_listed";
    const json list_sessions_response = make_agent_daemon_command_response(list_sessions_result);
    if (!list_sessions_response.value("ok", false) ||
            list_sessions_response.value("event", "") != "sessions_listed" ||
            !list_sessions_response.contains("session_keys") ||
            list_sessions_response["session_keys"].size() != 1) {
        std::fprintf(stderr, "list_sessions response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result get_session_result = status_result;
    get_session_result.request_id = "session-1";
    get_session_result.event = "session_found";
    get_session_result.status.sessions = {status_result.status.sessions.front()};
    get_session_result.status.session_count = 1;
    get_session_result.status.session_snapshot_populated = true;
    const json get_session_response = make_agent_daemon_command_response(get_session_result);
    if (!get_session_response.value("ok", false) ||
            get_session_response.value("event", "") != "session_found" ||
            !get_session_response.contains("session_keys") ||
            get_session_response["session_keys"].size() != 1 ||
            get_session_response["session_keys"][0].value("session_id", "") != "session-a") {
        std::fprintf(stderr, "get_session response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result resources_result;
    resources_result.ok = true;
    resources_result.request_id = "resources-1";
    resources_result.response_kind = common_agent_daemon_response_kind::listing;
    resources_result.event = "resources_listed";
    resources_result.status.state = common_agent_daemon_state::ready;
    resources_result.status.live = true;
    resources_result.status.ready = true;
    resources_result.listing_result.resources.push_back({
        "agent-resource://resource/r-1",
        "search-results.json",
        "Stored search payload",
        "application/json",
        21,
        common_runtime_resource_scope::session,
        {
            "Preserve full payload beyond inline summary.",
            "Full search results payload.",
            "Read when the top inline hits are insufficient.",
            "May be truncated by max_bytes.",
            {"search", "results"},
            {"repository_search"},
        },
        "r-1",
        "",
        "namespace-a",
        "session-a",
        "project-a",
        "turn-a",
        "",
        "",
        "",
        0,
        0,
    });
    const json resources_response = make_agent_daemon_command_response(resources_result);
    if (!resources_response.value("ok", false) ||
            resources_response.value("event", "") != "resources_listed" ||
            !resources_response.contains("resources") ||
            !resources_response["resources"].is_array() ||
            resources_response["resources"].size() != 1) {
        std::fprintf(stderr, "list_resources response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result memories_result;
    memories_result.ok = true;
    memories_result.request_id = "memories-1";
    memories_result.response_kind = common_agent_daemon_response_kind::listing;
    memories_result.event = "memories_listed";
    memories_result.status.state = common_agent_daemon_state::ready;
    memories_result.status.live = true;
    memories_result.status.ready = true;
    memories_result.listing_result.memories.push_back({
        "mem-1",
        "decision",
        "session",
        "Tool results should stay host-owned.",
        "session-a",
        "project-a",
        "turn-a",
        1,
    });
    const json memories_response = make_agent_daemon_command_response(memories_result);
    if (!memories_response.value("ok", false) ||
            memories_response.value("event", "") != "memories_listed" ||
            !memories_response.contains("memories") ||
            memories_response["memories"].size() != 1 ||
            memories_response["memories"][0].value("id", "") != "mem-1") {
        std::fprintf(stderr, "list_memories response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result plans_result;
    plans_result.ok = true;
    plans_result.request_id = "plans-1";
    plans_result.response_kind = common_agent_daemon_response_kind::listing;
    plans_result.event = "plans_listed";
    plans_result.status.state = common_agent_daemon_state::ready;
    plans_result.status.live = true;
    plans_result.status.ready = true;
    plans_result.listing_result.plans.push_back({
        "plan-1",
        "Verify daemon session flow.",
        "Run one admin smoke.",
        "active",
        "session",
        "session-a",
        "project-a",
        "turn-a",
        "step-1",
        "Inspect resource output.",
        2,
        3,
        1,
    });
    const json plans_response = make_agent_daemon_command_response(plans_result);
    if (!plans_response.value("ok", false) ||
            plans_response.value("event", "") != "plans_listed" ||
            !plans_response.contains("plans") ||
            plans_response["plans"].size() != 1 ||
            plans_response["plans"][0].value("plan_id", "") != "plan-1") {
        std::fprintf(stderr, "list_plans response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result turn_result;
    turn_result.ok = true;
    turn_result.request_id = "turn-1";
    turn_result.response_kind = common_agent_daemon_response_kind::turn;
    turn_result.event = "turn_completed";
    turn_result.turn_result.response = "DONE";
    turn_result.turn_result.runtime_reused = true;
    turn_result.turn_result.failure_class = common_agent_failure_class::execution;
    turn_result.turn_result.response_generation_status = common_agent_generation_status::completed;
    turn_result.turn_result.response_stop_reason = common_agent_generation_stop_reason::eos;
    turn_result.status.state = common_agent_daemon_state::ready;
    turn_result.status.live = true;
    turn_result.status.ready = true;
    turn_result.status.worker_running = true;
    turn_result.status.accepting_commands = true;
    turn_result.status.max_queue_size = 8;
    turn_result.status.queue_capacity_remaining = 8;
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
            turn_response.value("event", "") != "turn_completed" ||
            turn_response.value("response", "") != "DONE" ||
            turn_response.value("state", "") != "ready" ||
            !turn_response.value("runtime_reused", false) ||
            turn_response.value("failure_class", "") != "execution" ||
            turn_response.value("response_generation_status", "") != "completed" ||
            turn_response.value("response_stop_reason", "") != "eos" ||
            !turn_response.contains("trace") ||
            !turn_response["trace"].is_array() ||
            turn_response["trace"].size() != 1 ||
            turn_response["trace"][0].value("tool_name", "") != "calculator") {
        std::fprintf(stderr, "turn response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result resource_result;
    resource_result.ok = true;
    resource_result.request_id = "resource-1";
    resource_result.response_kind = common_agent_daemon_response_kind::resource;
    resource_result.event = "resource_read";
    resource_result.status.state = common_agent_daemon_state::ready;
    resource_result.status.live = true;
    resource_result.status.ready = true;
    resource_result.status.worker_running = true;
    resource_result.status.accepting_commands = true;
    resource_result.status.max_queue_size = 8;
    resource_result.status.queue_capacity_remaining = 8;
    resource_result.resource_result.resource.uri = "agent-resource://resource/r-1";
    resource_result.resource_result.resource.name = "search-results.json";
    resource_result.resource_result.resource.description = "Stored search payload";
    resource_result.resource_result.resource.mime_type = "application/json";
    resource_result.resource_result.resource.size_bytes = 21;
    resource_result.resource_result.resource.metadata.usage_hint = "Read the full payload when inline summary is insufficient.";
    resource_result.resource_result.content = "{\"results\":[\"stub\"]}";
    const json resource_response = make_agent_daemon_command_response(resource_result);
    if (!resource_response.value("ok", false) ||
            resource_response.value("event", "") != "resource_read" ||
            !resource_response.contains("resource") ||
            resource_response["resource"].value("uri", "") != "agent-resource://resource/r-1" ||
            resource_response.value("content", "") != "{\"results\":[\"stub\"]}") {
        std::fprintf(stderr, "resource response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result shutdown_result;
    shutdown_result.ok = true;
    shutdown_result.request_id = "shutdown-1";
    shutdown_result.response_kind = common_agent_daemon_response_kind::lifecycle;
    shutdown_result.event = "shutdown";
    shutdown_result.status.state = common_agent_daemon_state::draining;
    shutdown_result.status.live = true;
    shutdown_result.status.ready = false;
    shutdown_result.status.worker_running = true;
    shutdown_result.status.accepting_commands = false;
    shutdown_result.status.shutdown_requested = true;
    shutdown_result.status.queued_command_count = 0;
    shutdown_result.status.max_queue_size = 8;
    shutdown_result.status.queue_capacity_remaining = 8;
    const json shutdown_response = make_agent_daemon_command_response(shutdown_result);
    if (!shutdown_response.value("ok", false) ||
            shutdown_response.value("event", "") != "shutdown" ||
            shutdown_response.value("state", "") != "draining" ||
            shutdown_response.value("shutdown_requested", false) != true ||
            shutdown_response.value("accepting_commands", true) != false) {
        std::fprintf(stderr, "shutdown lifecycle response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result drain_result;
    drain_result.ok = true;
    drain_result.request_id = "drain-1";
    drain_result.response_kind = common_agent_daemon_response_kind::lifecycle;
    drain_result.event = "drain";
    drain_result.status.state = common_agent_daemon_state::draining;
    drain_result.status.live = true;
    drain_result.status.ready = false;
    drain_result.status.worker_running = true;
    drain_result.status.accepting_commands = true;
    drain_result.status.shutdown_requested = false;
    drain_result.status.queued_command_count = 0;
    drain_result.status.max_queue_size = 8;
    drain_result.status.queue_capacity_remaining = 8;
    const json drain_response = make_agent_daemon_command_response(drain_result);
    if (!drain_response.value("ok", false) ||
            drain_response.value("event", "") != "drain" ||
            drain_response.value("state", "") != "draining" ||
            drain_response.value("shutdown_requested", true) != false) {
        std::fprintf(stderr, "drain lifecycle response mismatch\n");
        return 1;
    }

    std::printf("daemon_protocol_ready=%s\n", ready.dump().c_str());
    std::printf("daemon_protocol_status=%s\n", status_response.dump().c_str());
    std::printf("daemon_protocol_turn=%s\n", turn_response.dump().c_str());
    std::printf("daemon_protocol_shutdown=%s\n", shutdown_response.dump().c_str());
    return 0;
}
