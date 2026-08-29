#include "tools/agent/daemon/agent-daemon-adapter.h"

#include <chrono>
#include <cstdio>

using json = nlohmann::ordered_json;

int main() {
    if (common_agent_daemon_event_category_for_type(
                common_agent_daemon_event_type::memories_listed) != common_agent_daemon_event_category::memory ||
            common_agent_daemon_event_category_for_type(
                common_agent_daemon_event_type::memories_list_failed) != common_agent_daemon_event_category::memory ||
            common_agent_daemon_event_category_for_type(
                common_agent_daemon_event_type::plans_listed) != common_agent_daemon_event_category::plan ||
            common_agent_daemon_event_category_for_type(
                common_agent_daemon_event_type::plans_list_failed) != common_agent_daemon_event_category::plan ||
            common_agent_daemon_event_category_for_type(
                common_agent_daemon_event_type::resource_chunk_planned) != common_agent_daemon_event_category::resource ||
            common_agent_daemon_event_category_for_type(
                common_agent_daemon_event_type::resource_chunk_processed) != common_agent_daemon_event_category::resource ||
            std::string(common_agent_daemon_event_type_name(
                common_agent_daemon_event_type::resource_chunk_processed)) != "resource.chunk_processed") {
        std::fprintf(stderr, "list event categories are incomplete\n");
        return 1;
    }

    daemon_options options;
    options.default_mode = "chat";
    options.turn_timeout_ms = 7000;
    options.inference_step_timeout_ms = 222;
    options.tool_timeout_ms = 333;
    options.mcp_connect_timeout_ms = 444;
    options.mcp_request_timeout_ms = 555;
    options.mcp_shutdown_timeout_ms = 666;
    options.model_profile = "agent-default";

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

    common_agent_daemon_command put_resource_command;
    if (!parse_agent_daemon_command(
            json{
                {"request_id", "resource-put-1"},
                {"command", "put_resource"},
                {"name", "notes.md"},
                {"description", "Uploaded notes"},
                {"mime_type", "text/markdown"},
                {"text", "# Notes\n"},
                {"scope", "session"},
                {"session_id", "session-a"},
                {"namespace_id", "namespace-a"},
                {"project_id", "project-a"},
                {"turn_id", "turn-a"},
            },
            options,
            common_agent_runtime_host_mode::chat,
            put_resource_command,
            error) ||
            put_resource_command.type != common_agent_daemon_command_type::put_resource ||
            !put_resource_command.resource_put.has_value() ||
            put_resource_command.resource_put->request.name != "notes.md" ||
            put_resource_command.resource_put->request.scope != common_runtime_resource_scope::session ||
            put_resource_command.resource_put->request.text != "# Notes\n") {
        std::fprintf(stderr, "failed to parse put_resource command: %s\n", error.c_str());
        return 1;
    }
    if (status_command.type != common_agent_daemon_command_type::get_status) {
        std::fprintf(stderr, "status command parsed to wrong type\n");
        return 1;
    }

    common_agent_daemon_command binary_resource_command;
    if (!parse_agent_daemon_command(
            json{
                {"request_id", "resource-put-binary-1"},
                {"command", "put_resource"},
                {"name", "image.bin"},
                {"mime_type", "application/octet-stream"},
                {"bytes_base64", "AAEC"},
                {"scope", "session"},
            },
            options,
            common_agent_runtime_host_mode::chat,
            binary_resource_command,
            error) ||
            !binary_resource_command.resource_put.has_value() ||
            !binary_resource_command.resource_put->request.bytes_are_authoritative ||
            binary_resource_command.resource_put->request.bytes != std::string("\x00\x01\x02", 3)) {
        std::fprintf(stderr, "failed to parse binary put_resource command: %s\n", error.c_str());
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

    common_agent_daemon_command shutdown_command;
    if (!parse_agent_daemon_command(
            json{{"request_id", "shutdown-1"}, {"command", "shutdown"}},
            options,
            common_agent_runtime_host_mode::chat,
            shutdown_command,
            error) ||
            shutdown_command.type != common_agent_daemon_command_type::shutdown) {
        std::fprintf(stderr, "failed to parse shutdown command: %s\n", error.c_str());
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
                {"mode", "agent"},
                {"include_summary", true},
                {"memory_scope", "project"},
                {"plan_scope", "project"},
                {"resource_refs", json::array({"agent-resource://resource/r-1"})},
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
            !turn_command.turn->include_summary ||
            turn_command.turn->request.request_id != "turn-1" ||
            turn_command.turn->request.turn.mode != common_agent_runtime_host_mode::agent ||
            turn_command.turn->request.turn.memory_scope != common_memory_scope::project ||
            turn_command.turn->request.turn.plan_scope != common_plan_scope::project ||
            turn_command.turn->request.turn.input_resources.size() != 1 ||
            turn_command.turn->request.turn.input_resources[0].resource.uri != "agent-resource://resource/r-1" ||
            turn_command.turn->request.turn.model_profile_id != "agent-default" ||
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

    for (const auto & field : {"tool_profile", "allowed_tools", "allow_writes", "enable_shell"}) {
        common_agent_daemon_command rejected_command;
        if (parse_agent_daemon_command(
                json{{"request_id", "turn-rejected"}, {"command", "run_turn"},
                     {"prompt", "hello"}, {field, field == "allowed_tools" ? json::array() : json(true)}},
                options,
                common_agent_runtime_host_mode::chat,
                rejected_command,
                error) || error.find("client-controlled field is not allowed") == std::string::npos) {
            std::fprintf(stderr, "client override field was not rejected: %s\n", field);
            return 1;
        }
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

    common_agent_daemon_command reload_command;
    if (!parse_agent_daemon_command(
            json{{"request_id", "reload-1"}, {"command", "reload_config"}, {"path", "fake-config.json"}},
            options,
            common_agent_runtime_host_mode::chat,
            reload_command,
            error) ||
            reload_command.type != common_agent_daemon_command_type::reload_config ||
            reload_command.reload_path != "fake-config.json") {
        std::fprintf(stderr, "reload_config command did not preserve its path: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_runtime reload_runtime;
    reload_runtime.reload_config = [](
            const std::string & path,
            common_agent_daemon_reload_result & result,
            std::string & callback_error) {
        if (path == "fake-restart-required.json") {
            result.restart_required = {"model.path", "limits.worker_count"};
            result.warning = "configuration was not applied; restart the daemon to change the listed fields";
            callback_error.clear();
            return true;
        }
        if (path == "fake-provider-change.json") {
            result.config_version = 3;
            result.applied_fields = {"tools.providers"};
            result.providers_added = {"new-provider"};
            result.providers_removed = {"old-provider"};
            result.providers_replaced = {"changed-provider"};
            callback_error.clear();
            return true;
        }
        result.config_version = 2;
        result.applied_fields = {"limits.tool_timeout_ms"};
        callback_error.clear();
        return true;
    };
    common_agent_daemon_service reload_service(std::move(reload_runtime));
    common_agent_daemon_command_result reload_result;
    if (!reload_service.execute(reload_command, reload_result, error) ||
            reload_result.event != "config.reload.completed" ||
            reload_result.reload_result.config_version != 2 ||
            reload_result.reload_result.applied_fields.size() != 1 ||
            reload_result.events.size() != 2 ||
            reload_result.events[0].event_type != common_agent_daemon_event_type::config_reload_started ||
            reload_result.events[1].event_type != common_agent_daemon_event_type::config_reload_completed) {
        std::fprintf(stderr, "fake config reload did not complete as expected: %s\n", error.c_str());
        return 1;
    }
    reload_command.reload_path = "fake-restart-required.json";
    if (reload_service.execute(reload_command, reload_result, error) ||
            reload_result.event != "config.reload.rejected" ||
            reload_result.reload_result.restart_required.size() != 2 ||
            reload_result.reload_result.warning.empty() ||
            reload_result.events.size() != 2 ||
            reload_result.events[1].event_type != common_agent_daemon_event_type::config_reload_rejected) {
        std::fprintf(stderr, "fake restart-required reload was not rejected as expected: %s\n", error.c_str());
        return 1;
    }
    reload_command.reload_path = "fake-provider-change.json";
    if (!reload_service.execute(reload_command, reload_result, error) ||
            reload_result.event != "config.reload.completed" ||
            reload_result.reload_result.providers_added.size() != 1 ||
            reload_result.reload_result.providers_removed.size() != 1 ||
            reload_result.reload_result.providers_replaced.size() != 1) {
        std::fprintf(stderr, "fake provider change did not complete as expected: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_service missing_payload_service({});
    common_agent_daemon_command missing_payload_command;
    missing_payload_command.request_id = "turn-missing";
    missing_payload_command.type = common_agent_daemon_command_type::run_turn;
    common_agent_daemon_command_result missing_payload_result;
    if (missing_payload_service.execute(missing_payload_command, missing_payload_result, error) ||
            missing_payload_result.event != "turn_failed" ||
            missing_payload_result.events.size() != 1 ||
            missing_payload_result.events[0].event_type != common_agent_daemon_event_type::turn_failed ||
            missing_payload_result.turn_result.error != "run_turn command missing turn payload" ||
            missing_payload_result.turn_result.cancelled ||
            missing_payload_result.turn_result.failure_class != common_agent_failure_class::execution ||
            missing_payload_result.turn_result.response_generation_status != common_agent_generation_status::errored ||
            missing_payload_result.turn_result.response_stop_reason != common_agent_generation_stop_reason::error) {
        std::fprintf(stderr, "missing-payload service failure did not preserve expected turn metadata\n");
        return 1;
    }

    common_agent_daemon_command cancel_command;
    cancel_command.request_id = "cancel-service-1";
    cancel_command.type = common_agent_daemon_command_type::cancel_turn;
    common_agent_daemon_command_result cancel_service_result;
    if (missing_payload_service.execute(cancel_command, cancel_service_result, error) ||
            cancel_service_result.events.size() != 1 ||
            cancel_service_result.events[0].event_type != common_agent_daemon_event_type::turn_cancel_rejected) {
        std::fprintf(stderr, "service cancel rejection did not preserve typed event metadata\n");
        return 1;
    }

    common_agent_daemon_command list_sessions_service_command;
    list_sessions_service_command.request_id = "sessions-service-1";
    list_sessions_service_command.type = common_agent_daemon_command_type::list_sessions;
    common_agent_daemon_command_result list_sessions_service_result;
    missing_payload_service.execute(list_sessions_service_command, list_sessions_service_result, error);
    if (list_sessions_service_result.events.size() != 1 ||
            list_sessions_service_result.events[0].event_type != common_agent_daemon_event_type::sessions_listed) {
        std::fprintf(stderr, "service list_sessions did not preserve typed event metadata\n");
        return 1;
    }

    common_agent_daemon_command get_session_service_command;
    get_session_service_command.request_id = "session-service-1";
    get_session_service_command.type = common_agent_daemon_command_type::get_session;
    get_session_service_command.session = common_agent_daemon_session_payload{{"namespace-a", "session-a"}};
    common_agent_daemon_command_result get_session_service_result;
    if (missing_payload_service.execute(get_session_service_command, get_session_service_result, error) ||
            get_session_service_result.event != "session_lookup_failed" ||
            get_session_service_result.events.size() != 1 ||
            get_session_service_result.events[0].event_type != common_agent_daemon_event_type::session_lookup_failed) {
        std::fprintf(stderr, "service get_session failure did not preserve typed event metadata\n");
        return 1;
    }

    common_agent_daemon_command read_resource_service_command;
    read_resource_service_command.request_id = "resource-service-1";
    read_resource_service_command.type = common_agent_daemon_command_type::read_resource;
    read_resource_service_command.resource = common_agent_daemon_resource_payload{
        "agent-resource://resource/r-1",
        {"namespace-a", "session-a", "project-a", "turn-a"},
        1024,
    };
    common_agent_daemon_command_result read_resource_service_result;
    if (missing_payload_service.execute(read_resource_service_command, read_resource_service_result, error) ||
            read_resource_service_result.event != "resource_read_failed" ||
            read_resource_service_result.events.size() != 1 ||
            read_resource_service_result.events[0].event_type != common_agent_daemon_event_type::resource_read_failed) {
        std::fprintf(stderr, "service read_resource failure did not preserve typed event metadata\n");
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
            rejected_turn_result.events.size() != 1 ||
            rejected_turn_result.events[0].type != "turn.rejected" ||
            rejected_turn_result.turn_result.error != "daemon is not accepting new turns" ||
            !rejected_turn_result.turn_result.cancelled ||
            rejected_turn_result.turn_result.failure_class != common_agent_failure_class::timeout ||
            rejected_turn_result.turn_result.response_generation_status != common_agent_generation_status::cancelled ||
            rejected_turn_result.turn_result.response_stop_reason != common_agent_generation_stop_reason::cancelled) {
        std::fprintf(stderr, "rejected-turn service failure did not preserve expected timeout metadata\n");
        return 1;
    }

    common_agent_daemon_command_result service_status_result;
    service_status_result.request_id = "status-live-1";
    rejected_turn_service.populate_status(service_status_result, error);
    if (service_status_result.event != "status" ||
            service_status_result.events.size() != 1 ||
            service_status_result.events[0].event_type != common_agent_daemon_event_type::status_reported) {
        std::fprintf(stderr, "service status did not preserve typed status event metadata: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command_result drain_service_result;
    if (!rejected_turn_service.execute(drain_command, drain_service_result, error) ||
            drain_service_result.event != "drain" ||
            drain_service_result.events.size() != 1 ||
            drain_service_result.events[0].event_type != common_agent_daemon_event_type::drain_requested) {
        std::fprintf(stderr, "drain service result did not preserve typed daemon event metadata: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_command_result shutdown_service_result;
    if (!rejected_turn_service.execute(shutdown_command, shutdown_service_result, error) ||
            shutdown_service_result.event != "shutdown" ||
            shutdown_service_result.events.size() != 1 ||
            shutdown_service_result.events[0].event_type != common_agent_daemon_event_type::shutdown_requested) {
        std::fprintf(stderr, "shutdown service result did not preserve typed daemon event metadata: %s\n", error.c_str());
        return 1;
    }

    const json ready = make_agent_daemon_ready_response(options);
    if (!ready.value("ok", false) ||
            ready.value("event", "") != "ready" ||
            ready.value("protocol_version", 0) != 1 ||
            ready.value("tooling", json::object()).value("profile", "") != "minimal" ||
            ready.value("tooling", json::object()).value("tools", json::array()).size() != 2) {
        std::fprintf(stderr, "ready response mismatch\n");
        return 1;
    }
    if (ready.value("readiness", json::object()).value("health", "") != "unknown" ||
            ready.value("readiness", json::object()).value("warnings", json::array()).empty()) {
        std::fprintf(stderr, "ready response readiness handshake mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result status_result;
    status_result.ok = true;
    status_result.request_id = "status-1";
    status_result.response_kind = common_agent_daemon_response_kind::status;
    status_result.event = "status";
    status_result.daemon_event_count = 1;
    status_result.events.push_back({
        "status.reported",
        "status-1",
        "",
        "",
        "",
        "",
        "",
        "ready",
        common_agent_daemon_event_type::status_reported,
        7,
    });
    status_result.status.state = common_agent_daemon_state::ready;
    status_result.status.live = true;
    status_result.status.ready = true;
    status_result.status.readiness.health = "ready";
    status_result.status.readiness.model = "loaded";
    status_result.status.readiness.inference = "available";
    status_result.status.readiness.memory_store = "ready";
    status_result.status.readiness.plan_store = "ready";
    status_result.status.readiness.resource_store = "ready";
    status_result.status.readiness.tool_profile = "minimal";
    status_result.status.readiness.providers.push_back({
        "local-mcp", "ready", false, {}});
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
    status_result.status.active_pending_operation_kind = "inference";
    status_result.status.active_pending_operation_detail = "session host turn execution";
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
    session_descriptor.pending_operation_kind = "inference";
    session_descriptor.pending_operation_detail = "session host turn execution";
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
            status_response.value("active_pending_operation_kind", "") != "inference" ||
            status_response.value("active_pending_operation_detail", "") != "session host turn execution" ||
            status_response.value("readiness", json::object()).value("health", "") != "ready" ||
            status_response.value("readiness", json::object()).value("stores", json::object()).value("memory", "") != "ready" ||
            status_response.value("readiness", json::object()).value("providers", json::array()).size() != 1 ||
            !status_response.contains("session_keys") ||
            !status_response["session_keys"].is_array() ||
            status_response["session_keys"].size() != 1 ||
            status_response["session_keys"][0].value("lane_state", "") != "running" ||
            status_response["session_keys"][0].value("active_request_id", "") != "request-a" ||
            status_response["session_keys"][0].value("active_turn_id", "") != "turn-active" ||
            status_response["session_keys"][0].value("active_turn_phase", "") != "awaiting_inference" ||
            status_response["session_keys"][0].value("active_turn_disposition", "") != "continue_immediately" ||
            status_response["session_keys"][0].value("pending_operation_kind", "") != "inference" ||
            status_response["session_keys"][0].value("pending_operation_detail", "") != "session host turn execution" ||
            status_response["session_keys"][0].value("last_turn_id", "") != "turn-a" ||
            status_response["session_keys"][0].value("last_turn_phase", "") != "completed" ||
            status_response["session_keys"][0].value("last_turn_disposition", "") != "completed" ||
            !status_response.contains("events") ||
            !status_response["events"].is_array() ||
            status_response["events"][0].value("event_type", "") != "status.reported" ||
            status_response["events"][0].value("sequence", 0) != 7) {
        std::fprintf(stderr, "status response mismatch\n");
        return 1;
    }

    common_agent_daemon_command_result event_result;
    event_result.ok = true;
    event_result.request_id = "turn-events-1";
    event_result.response_kind = common_agent_daemon_response_kind::turn;
    event_result.event = "turn_completed";
    event_result.events = {
        {
            "plan.created",
            "turn-events-1",
            "turn-a",
            "",
            "",
            "",
            "",
            "plan-a",
            common_agent_daemon_event_type::plan_created,
            11,
        },
        {
            "plan.step_started",
            "turn-events-1",
            "turn-a",
            "",
            "",
            "",
            "",
            "step-a",
            common_agent_daemon_event_type::plan_step_started,
            12,
        },
        {
            "observation.recorded",
            "turn-events-1",
            "turn-a",
            "",
            "",
            "",
            "",
            "obs-a",
            common_agent_daemon_event_type::observation_recorded,
            13,
        },
        {
            "resource.created",
            "turn-events-1",
            "turn-a",
            "",
            "",
            "",
            "",
            "agent-resource://resource/r-1",
            common_agent_daemon_event_type::resource_created,
            14,
        },
        {
            "resource.attached",
            "turn-events-1",
            "turn-a",
            "",
            "",
            "",
            "",
            "agent-resource://resource/r-1",
            common_agent_daemon_event_type::resource_attached,
            15,
        },
    };
    const json event_response = make_agent_daemon_command_response(event_result);
    if (!event_response.contains("events") ||
            !event_response["events"].is_array() ||
            event_response["events"].size() != 5 ||
            event_response["events"][0].value("event_type", "") != "plan.created" ||
            event_response["events"][1].value("event_type", "") != "plan.step_started" ||
            event_response["events"][2].value("event_type", "") != "observation.recorded" ||
            event_response["events"][3].value("event_type", "") != "resource.created" ||
            event_response["events"][4].value("event_type", "") != "resource.attached" ||
            event_response["events"][4].value("detail", "") != "agent-resource://resource/r-1") {
        std::fprintf(stderr, "event response mismatch\n");
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
            {"repository.search"},
        },
        {},
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
    turn_result.turn_result.continuation_checkpoint = common_agent_continuation_checkpoint{
        "checkpoint-1",
        "request-1",
        "turn-1",
        "plan-1",
        "step-2",
        "resume",
        4,
        2,
        common_agent_continuation_reason::completion_limit,
        {"step-1"},
        {{"workspace://checkpoint/log", "log", "checkpoint log", "text/plain", 12}},
    };
    turn_result.turn_result.continuation_checkpoint->chunk_parent_uri =
        "agent-resource://resource/original";
    turn_result.turn_result.continuation_checkpoint->chunk_count = 3;
    turn_result.turn_result.continuation_checkpoint->completed_chunk_indexes = {0, 1};
    common_agent_dataset_ref checkpoint_dataset;
    checkpoint_dataset.uri = "dataset://sales";
    checkpoint_dataset.name = "sales";
    checkpoint_dataset.row_count = 2;
    checkpoint_dataset.column_count = 1;
    checkpoint_dataset.source_resource_uri = "resource://turn/turn-1/sales.json";
    checkpoint_dataset.source_provider = "sales-api";
    checkpoint_dataset.source_operation = "listSales";
    checkpoint_dataset.source_request_json = "{}";
    checkpoint_dataset.retrieved_at = 123;
    checkpoint_dataset.content_hash = "sha256:sales";
    turn_result.turn_result.continuation_checkpoint->dataset_refs.push_back(checkpoint_dataset);
    common_runtime_resource_ref checkpoint_resource;
    checkpoint_resource.uri = "workspace://checkpoint/resource";
    checkpoint_resource.name = "checkpoint resource";
    checkpoint_resource.description = "checkpoint resource";
    checkpoint_resource.mime_type = "text/plain";
    checkpoint_resource.size_bytes = 24;
    turn_result.turn_result.continuation_checkpoint->working_state = common_agent_working_state{
        "continue the bounded operation",
        "verification",
        {"inspect", "implement"},
        "verify",
        {"synthesize"},
        {"preserve host authority"},
        {"use bounded resources"},
        {"chunk 2 remains pending"},
        {checkpoint_resource},
        {"agent-resource://resource/original[0/3];status=completed;observation=obs-0"},
        {"build passed"},
        "resume synthesis",
    };
    turn_result.turn_summary = common_agent_turn_summary{
        "agent",
        "completed",
        "Choose a tool",
        {"plan", "response"},
        {"calculator"},
        0,
        0,
        0,
        0,
        true,
        "eos",
        {},
    };

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
            turn_response["trace"][0].value("tool_name", "") != "calculator" ||
            !turn_response.contains("continuation_checkpoint") ||
            turn_response["continuation_checkpoint"].value("checkpoint_id", "") != "checkpoint-1" ||
            turn_response["continuation_checkpoint"].value("request_id", "") != "request-1" ||
            turn_response["continuation_checkpoint"].value("reason", "") != "completion_limit" ||
            turn_response["continuation_checkpoint"].value("chunk_parent_uri", "") !=
                "agent-resource://resource/original" ||
            turn_response["continuation_checkpoint"].value("chunk_count", 0) != 3 ||
            turn_response["continuation_checkpoint"].value("completed_chunk_indexes", json::array()) !=
                json::array({0, 1}) ||
            !turn_response["continuation_checkpoint"].contains("working_state") ||
            turn_response["continuation_checkpoint"]["working_state"].value("goal", "") !=
                "continue the bounded operation" ||
            turn_response["continuation_checkpoint"]["working_state"].value("current_phase", "") !=
                "verification" ||
            turn_response["continuation_checkpoint"]["working_state"].value("active_step", "") !=
                "verify" ||
            turn_response["continuation_checkpoint"]["working_state"].value("continuation_action", "") !=
                "resume synthesis" ||
            turn_response["continuation_checkpoint"]["working_state"]["completed_steps"] !=
                json::array({"inspect", "implement"}) ||
            turn_response["continuation_checkpoint"]["working_state"]["resource_refs"].size() != 1 ||
            turn_response["continuation_checkpoint"]["working_state"]["resource_refs"][0].value("uri", "") !=
                "workspace://checkpoint/resource" ||
            turn_response["continuation_checkpoint"]["dataset_refs"].size() != 1 ||
            turn_response["continuation_checkpoint"]["dataset_refs"][0].value("source_provider", "") !=
                "sales-api" ||
            turn_response["continuation_checkpoint"]["dataset_refs"][0].value("content_hash", "") !=
                "sha256:sales" ||
            !turn_response.contains("turn_summary") ||
            turn_response["turn_summary"].value("mode", "") != "agent" ||
            turn_response["turn_summary"].value("status", "") != "completed") {
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
