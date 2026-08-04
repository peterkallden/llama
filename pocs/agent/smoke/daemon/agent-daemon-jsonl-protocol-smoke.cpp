#include "tools/agent/daemon/agent-daemon-client-status.h"
#include "tools/agent/daemon/agent-daemon-jsonl-protocol.h"

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
    turn_request_spec.mode = "agent";
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
            turn_request.value("mode", "") != "agent" ||
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

    const json readiness_probe_payload = {
        {"ok", true},
        {"event", "status"},
        {"state", "ready"},
        {"live", true},
        {"ready", true},
        {"readiness", {
            {"health", "ready"},
            {"inference", "available"},
        }},
    };
    agent_daemon_jsonl_status_response parsed_status;
    if (!parse_agent_daemon_jsonl_status_response(readiness_probe_payload, parsed_status, error) ||
            parsed_status.readiness.value("health", "") != "ready" ||
            parsed_status.readiness.value("inference", "") != "available") {
        std::fprintf(stderr, "status readiness contract mismatch: %s\n", error.c_str());
        return 1;
    }
    const auto verbose_status = render_agent_daemon_client_status_verbose(parsed_status);
    if (verbose_status.find("readiness") == std::string::npos ||
            verbose_status.find("inference") == std::string::npos) {
        std::fprintf(stderr, "verbose status rendering omitted readiness details\n");
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

    const auto list_sessions_request = make_agent_daemon_jsonl_list_sessions_request({});
    if (!list_sessions_request.is_object() ||
            list_sessions_request.value("command", "") != "list_sessions") {
        std::fprintf(stderr, "list sessions request contract mismatch\n");
        return 1;
    }

    const auto get_session_request = make_agent_daemon_jsonl_get_session_request({
        "session-a",
        "namespace-a",
    });
    if (!get_session_request.is_object() ||
            get_session_request.value("command", "") != "get_session" ||
            get_session_request.value("session_id", "") != "session-a" ||
            get_session_request.value("namespace_id", "") != "namespace-a") {
        std::fprintf(stderr, "get session request contract mismatch\n");
        return 1;
    }

    const auto list_resources_request = make_agent_daemon_jsonl_list_resources_request({
        "session-a",
        "namespace-a",
        "project-a",
        "turn-a",
    });
    if (!list_resources_request.is_object() ||
            list_resources_request.value("command", "") != "list_resources" ||
            list_resources_request.value("project_id", "") != "project-a") {
        std::fprintf(stderr, "list resources request contract mismatch\n");
        return 1;
    }

    const auto list_memories_request = make_agent_daemon_jsonl_list_memories_request({
        "session-a",
        "namespace-a",
        "project-a",
        "turn-a",
    });
    if (!list_memories_request.is_object() ||
            list_memories_request.value("command", "") != "list_memories" ||
            list_memories_request.value("turn_id", "") != "turn-a") {
        std::fprintf(stderr, "list memories request contract mismatch\n");
        return 1;
    }

    const auto list_plans_request = make_agent_daemon_jsonl_list_plans_request({
        "session-a",
        "namespace-a",
        "project-a",
        "turn-a",
    });
    if (!list_plans_request.is_object() ||
            list_plans_request.value("command", "") != "list_plans" ||
            list_plans_request.value("namespace_id", "") != "namespace-a") {
        std::fprintf(stderr, "list plans request contract mismatch\n");
        return 1;
    }

    const auto read_resource_request = make_agent_daemon_jsonl_read_resource_request({
        "agent-resource://resource/r-1",
        "session-a",
        "namespace-a",
        "project-a",
        "turn-a",
        2048,
    });
    if (!read_resource_request.is_object() ||
            read_resource_request.value("command", "") != "read_resource" ||
            read_resource_request.value("uri", "") != "agent-resource://resource/r-1" ||
            read_resource_request.value("project_id", "") != "project-a" ||
            read_resource_request.value("turn_id", "") != "turn-a" ||
            read_resource_request.value("max_bytes", 0) != 2048) {
        std::fprintf(stderr, "read resource request contract mismatch\n");
        return 1;
    }

    const auto drain_request = make_agent_daemon_jsonl_drain_request({});
    if (!drain_request.is_object() ||
            drain_request.value("command", "") != "drain") {
        std::fprintf(stderr, "drain request contract mismatch\n");
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
                {"capabilities", json::array({"chat", "agent"})},
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
                {"daemon_event_count", 1},
                {"events", json::array({
                    {
                        {"type", "daemon.shutdown_requested"},
                        {"event_type", "daemon.shutdown_requested"},
                        {"sequence", 3},
                        {"request_id", "shutdown-1"},
                        {"detail", "shutdown requested"},
                    }
                })},
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
                {"daemon_event_count", 1},
                {"events", json::array({
                    {
                        {"type", "daemon.shutdown_requested"},
                        {"event_type", "daemon.shutdown_requested"},
                        {"sequence", 4},
                        {"request_id", "shutdown-1"},
                    }
                })},
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
            lifecycle_response.daemon_event_count != 1 ||
            lifecycle_response.events.size() != 1 ||
            lifecycle_response.events[0].event_type != "daemon.shutdown_requested" ||
            lifecycle_response.status.state != "draining" ||
            !lifecycle_response.status.shutdown_requested) {
        std::fprintf(stderr, "lifecycle response contract mismatch: %s\n", error.c_str());
        return 1;
    }

    agent_daemon_jsonl_lifecycle_response drain_response;
    if (!parse_agent_daemon_jsonl_lifecycle_response(
            {
                {"ok", true},
                {"event", "drain"},
                {"daemon_event_count", 1},
                {"events", json::array({
                    {
                        {"type", "daemon.drain_requested"},
                        {"event_type", "daemon.drain_requested"},
                        {"sequence", 5},
                    }
                })},
                {"state", "draining"},
                {"live", true},
                {"ready", false},
                {"worker_running", true},
                {"accepting_commands", true},
                {"shutdown_requested", false},
                {"sessions", 1},
                {"queued_commands", 0},
                {"max_queue_size", 8},
                {"queue_capacity_remaining", 8},
            },
            drain_response,
            error) ||
            drain_response.event != "drain" ||
            drain_response.status.state != "draining" ||
            drain_response.status.shutdown_requested) {
        std::fprintf(stderr, "drain response contract mismatch: %s\n", error.c_str());
        return 1;
    }

    agent_daemon_jsonl_status_response status_response;
    if (!parse_agent_daemon_jsonl_status_response(
            {
                {"ok", true},
                {"event", "status"},
                {"daemon_event_count", 1},
                {"events", json::array({
                    {
                        {"type", "status.reported"},
                        {"event_type", "status.reported"},
                        {"sequence", 6},
                        {"request_id", "status-1"},
                        {"detail", "ready"},
                    }
                })},
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
                {"active_pending_operation_kind", "inference"},
                {"active_pending_operation_detail", "session host turn execution"},
                {"session_keys", json::array({
                    {
                        {"namespace_id", "namespace-a"},
                        {"session_id", "session-a"},
                        {"project_id", "project-a"},
                        {"memory_scope", "project"},
                        {"plan_scope", "project"},
                        {"policy_pack_id", "pack-a"},
                        {"lane_state", "running"},
                        {"queued_turn_count", 0},
                        {"active_request_id", "request-a"},
                        {"active_turn_id", "turn-active"},
                        {"active_turn_phase", "awaiting_inference"},
                        {"active_turn_disposition", "continue_immediately"},
                        {"pending_operation_kind", "inference"},
                        {"pending_operation_detail", "session host turn execution"},
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
            status_response.active_pending_operation_kind != "inference" ||
            status_response.active_pending_operation_detail != "session host turn execution" ||
            status_response.daemon_event_count != 1 ||
            status_response.events.size() != 1 ||
            status_response.events[0].event_type != "status.reported" ||
            status_response.session_keys.size() != 1 ||
            status_response.session_keys[0].policy_pack_id != "pack-a" ||
            status_response.session_keys[0].lane_state != "running" ||
            status_response.session_keys[0].active_request_id != "request-a" ||
            status_response.session_keys[0].active_turn_id != "turn-active" ||
            status_response.session_keys[0].active_turn_phase != "awaiting_inference" ||
            status_response.session_keys[0].active_turn_disposition != "continue_immediately" ||
            status_response.session_keys[0].pending_operation_kind != "inference" ||
            status_response.session_keys[0].pending_operation_detail != "session host turn execution" ||
            status_response.session_keys[0].last_turn_id != "turn-a" ||
            status_response.session_keys[0].last_turn_phase != "completed" ||
            status_response.session_keys[0].last_turn_disposition != "completed") {
        std::fprintf(stderr, "status response contract mismatch: %s\n", error.c_str());
        return 1;
    }
    const auto summary = make_agent_daemon_client_status_summary(status_response);
    const std::string rendered_status = render_agent_daemon_client_status_summary(summary);
    if (rendered_status.find("state=ready") == std::string::npos ||
            rendered_status.find("active_turn=turn-active/awaiting_inference:continue_immediately pending=inference(session host turn execution)") == std::string::npos ||
            rendered_status.find(
                "session_bindings=namespace-a/session-a@project-a#pack-a{state=running}[active=turn-active/awaiting_inference:continue_immediately pending=inference(session host turn execution)]") == std::string::npos) {
        std::fprintf(stderr, "status render mismatch: %s\n", rendered_status.c_str());
        return 1;
    }

    agent_daemon_jsonl_turn_response turn_response;
    parse_agent_daemon_jsonl_turn_response(
        {
            {"ok", false},
            {"daemon_event_count", 1},
            {"events", json::array({
                {
                    {"type", "turn.rejected"},
                    {"event_type", "turn.rejected"},
                    {"sequence", 8},
                    {"request_id", "turn-1"},
                }
            })},
            {"cancelled", true},
            {"failure_class", "timeout"},
            {"response_generation_status", "cancelled"},
            {"response_stop_reason", "cancelled"},
            {"error", "turn deadline exceeded"},
            {"runtime_reused", true},
            {"event_count", 2},
            {"continuation_checkpoint", {
                {"checkpoint_id", "checkpoint-1"},
                {"request_id", "request-1"},
                {"turn_id", "turn-1"},
                {"plan_id", "plan-1"},
                {"active_step_id", "step-2"},
                {"next_action", "resume"},
                {"plan_version", 4},
                {"sequence", 2},
                {"reason", "completion_limit"},
                {"completed_step_ids", json::array({"step-1"})},
                {"resource_refs", json::array({
                    {{"uri", "workspace://checkpoint/log"}, {"name", "log"},
                     {"description", "checkpoint log"}, {"mime_type", "text/plain"},
                     {"size_bytes", 12}}
                })},
            }},
        },
        turn_response,
        error);
    if (error != "turn deadline exceeded" ||
            turn_response.event != "" ||
            !turn_response.cancelled ||
            turn_response.failure_class != "timeout" ||
            turn_response.response_generation_status != "cancelled" ||
            turn_response.response_stop_reason != "cancelled" ||
            !turn_response.continuation_checkpoint ||
            turn_response.continuation_checkpoint->checkpoint_id != "checkpoint-1" ||
            turn_response.continuation_checkpoint->request_id != "request-1" ||
            turn_response.continuation_checkpoint->turn_id != "turn-1" ||
            turn_response.continuation_checkpoint->reason !=
                common_agent_continuation_reason::completion_limit ||
            turn_response.continuation_checkpoint->resource_refs.size() != 1 ||
            turn_response.continuation_checkpoint->resource_refs[0].uri !=
                "workspace://checkpoint/log") {
        std::fprintf(stderr, "turn response contract mismatch: %s\n", error.c_str());
        return 1;
    }
    if (turn_response.daemon_event_count != 1 ||
            turn_response.events.size() != 1 ||
            turn_response.events[0].type != "turn.rejected") {
        std::fprintf(stderr, "turn response event parsing mismatch\n");
        return 1;
    }

    agent_daemon_jsonl_turn_response turn_failure_response;
    turn_failure_response.ok = false;
    turn_failure_response.event = "turn_rejected";
    turn_failure_response.cancelled = true;
    turn_failure_response.error = "daemon is not accepting new turns";
    turn_failure_response.failure_class = "timeout";
    turn_failure_response.response_generation_status = "cancelled";
    turn_failure_response.response_stop_reason = "cancelled";
    turn_failure_response.runtime_reused = false;
    turn_failure_response.event_count = 0;
    const auto turn_failure_summary =
        make_agent_daemon_client_turn_failure_summary(turn_failure_response, {});
    const std::string rendered_turn_failure =
        render_agent_daemon_client_turn_failure_summary(turn_failure_summary);
    if (rendered_turn_failure.find("turn_rejected") == std::string::npos ||
            rendered_turn_failure.find("class=timeout") == std::string::npos ||
            rendered_turn_failure.find("status=cancelled") == std::string::npos ||
            rendered_turn_failure.find("stop=cancelled") == std::string::npos ||
            rendered_turn_failure.find("cancelled=yes") == std::string::npos) {
        std::fprintf(stderr, "turn failure summary mismatch: %s\n", rendered_turn_failure.c_str());
        return 1;
    }

    agent_daemon_jsonl_resource_response resource_response;
    if (!parse_agent_daemon_jsonl_resource_response(
            {
                {"ok", true},
                {"event", "resource_read"},
                {"daemon_event_count", 1},
                {"events", json::array({
                    {
                        {"type", "resource.read"},
                        {"event_type", "resource.read"},
                        {"sequence", 9},
                    }
                })},
                {"content", "{\"results\":[\"stub\"]}"},
                {"resource", {
                    {"resource_id", "r-1"},
                    {"uri", "agent-resource://resource/r-1"},
                    {"name", "search-results.json"},
                    {"description", "Stored search payload"},
                    {"mime_type", "application/json"},
                    {"size_bytes", 21},
                    {"metadata", {
                        {"purpose", "Preserve full payload beyond inline summary."},
                        {"content_summary", "Full search results payload."},
                        {"usage_hint", "Read when the top inline hits are insufficient."},
                        {"limitations", "May be truncated by max_bytes."},
                        {"keywords", json::array({"search", "results"})},
                        {"entities", json::array({"repository.search"})},
                    }},
                }},
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
            },
            resource_response,
            error) ||
            resource_response.event != "resource_read" ||
            resource_response.resource.uri != "agent-resource://resource/r-1" ||
            resource_response.resource.metadata.usage_hint !=
                "Read when the top inline hits are insufficient." ||
            resource_response.daemon_event_count != 1 ||
            resource_response.events.size() != 1 ||
            resource_response.events[0].event_type != "resource.read" ||
            resource_response.content != "{\"results\":[\"stub\"]}" ||
            resource_response.status.state != "ready") {
        std::fprintf(stderr, "resource response contract mismatch: %s\n", error.c_str());
        return 1;
    }

    agent_daemon_jsonl_listing_response listing_response;
    if (!parse_agent_daemon_jsonl_listing_response(
            {
                {"ok", true},
                {"event", "resources_listed"},
                {"daemon_event_count", 1},
                {"events", json::array({
                    {
                        {"type", "resources.listed"},
                        {"event_type", "resources.listed"},
                        {"sequence", 10},
                    }
                })},
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
                {"resources", json::array({
                    {
                        {"uri", "agent-resource://resource/r-1"},
                        {"name", "search-results.json"},
                        {"description", "Stored search payload"},
                        {"mime_type", "application/json"},
                        {"size_bytes", 21},
                        {"metadata", {
                            {"purpose", "Preserve full payload beyond inline summary."},
                            {"content_summary", "Full search results payload."},
                            {"usage_hint", "Read when the top inline hits are insufficient."},
                            {"limitations", "May be truncated by max_bytes."},
                            {"keywords", json::array({"search", "results"})},
                        {"entities", json::array({"repository.search"})},
                        }},
                    }
                })},
            },
            listing_response,
            error) ||
            listing_response.event != "resources_listed" ||
            !listing_response.payload.contains("resources") ||
            !listing_response.payload["resources"].is_array() ||
            listing_response.payload["resources"].size() != 1 ||
            listing_response.daemon_event_count != 1 ||
            listing_response.events.size() != 1 ||
            listing_response.events[0].event_type != "resources.listed" ||
            listing_response.status.state != "ready") {
        std::fprintf(stderr, "listing response contract mismatch: %s\n", error.c_str());
        return 1;
    }

    std::printf("daemon_jsonl_mode=%s\n", parsed.value("mode", "").c_str());
    std::printf("daemon_jsonl_status=%s\n", status_request.value("command", "").c_str());
    std::printf("daemon_jsonl_cancel=%s\n", cancel_request.value("command", "").c_str());
    std::printf("daemon_jsonl_close=%s\n", close_request.value("command", "").c_str());
    std::printf("daemon_jsonl_list_sessions=%s\n", list_sessions_request.value("command", "").c_str());
    std::printf("daemon_jsonl_get_session=%s\n", get_session_request.value("command", "").c_str());
    std::printf("daemon_jsonl_list_resources=%s\n", list_resources_request.value("command", "").c_str());
    std::printf("daemon_jsonl_list_memories=%s\n", list_memories_request.value("command", "").c_str());
    std::printf("daemon_jsonl_list_plans=%s\n", list_plans_request.value("command", "").c_str());
    std::printf("daemon_jsonl_read_resource=%s\n", read_resource_request.value("command", "").c_str());
    std::printf("daemon_jsonl_drain=%s\n", drain_request.value("command", "").c_str());
    std::printf("daemon_jsonl_status_summary=%s\n", rendered_status.c_str());
    std::printf("daemon_jsonl_turn_failure=%s\n", rendered_turn_failure.c_str());
    std::printf("daemon_jsonl_shutdown=%s\n", shutdown_request.value("command", "").c_str());
    return 0;
}
