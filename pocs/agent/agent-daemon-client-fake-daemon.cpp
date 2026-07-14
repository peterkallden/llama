#include "agent-daemon-jsonl-protocol.h"

#include <cstdio>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using json = nlohmann::ordered_json;

namespace {

json make_session_status() {
    return {
        {"namespace_id", "namespace-a"},
        {"session_id", "session-a"},
        {"project_id", "project-a"},
        {"memory_scope", "session"},
        {"plan_scope", "session"},
        {"policy_pack_id", "pack-a"},
        {"lane_state", "running"},
        {"queued_turn_count", 0},
        {"active_request_id", ""},
        {"active_turn_id", ""},
        {"active_turn_phase", ""},
        {"active_turn_disposition", ""},
        {"last_turn_id", "turn-a"},
        {"last_turn_phase", "completed"},
        {"last_turn_disposition", "completed"},
    };
}

json make_status_payload(const std::string & event) {
    return {
        {"ok", true},
        {"event", event},
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
        {"session_keys", json::array({make_session_status()})},
    };
}

json make_lifecycle_payload(
        const std::string & event,
        const std::string & state,
        bool accepting_commands,
        bool shutdown_requested) {
    return {
        {"ok", true},
        {"event", event},
        {"state", state},
        {"live", true},
        {"ready", state == "ready"},
        {"worker_running", true},
        {"accepting_commands", accepting_commands},
        {"shutdown_requested", shutdown_requested},
        {"sessions", 1},
        {"queued_commands", 0},
        {"max_queue_size", 8},
        {"queue_capacity_remaining", 8},
        {"session_keys", json::array({make_session_status()})},
    };
}

} // namespace

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    std::string error;
    if (!write_agent_daemon_jsonl_message(
                stdout,
                {
                    {"ok", true},
                    {"event", "ready"},
                    {"default_mode", "chat"},
                    {"protocol_version", 1},
                    {"capabilities", json::array({
                        "chat",
                        "mini",
                        "planning",
                        "reflection",
                        "memory_learning",
                        "scoped_sessions",
                    })},
                },
                error)) {
        std::fprintf(stderr, "fake-daemon: failed to write ready payload: %s\n", error.c_str());
        return 1;
    }

    json request;
    while (read_agent_daemon_jsonl_message(stdin, request, error)) {
        const std::string command = request.value("command", "");
        json response;
        if (command == "run_turn") {
            response = {
                {"ok", true},
                {"event", "turn_completed"},
                {"cancelled", false},
                {"response", "stub turn response"},
                {"failure_class", "execution"},
                {"response_generation_status", "completed"},
                {"response_stop_reason", "eos"},
                {"runtime_reused", true},
                {"event_count", 1},
            };
        } else if (command == "status") {
            response = make_status_payload("status");
        } else if (command == "list_sessions") {
            response = make_status_payload("sessions_listed");
        } else if (command == "get_session") {
            response = make_status_payload("session_found");
        } else if (command == "list_resources") {
            response = make_status_payload("resources_listed");
            response["resources"] = json::array({
                {
                    {"resource_id", "r-1"},
                    {"uri", "agent-resource://resource/r-1"},
                    {"name", "search-results.json"},
                    {"description", "Stored search payload"},
                    {"mime_type", "application/json"},
                    {"size_bytes", 21},
                    {"scope", "session"},
                    {"metadata", {
                        {"purpose", "Preserve full payload beyond inline summary."},
                        {"content_summary", "Full search results payload."},
                        {"usage_hint", "Read when the top inline hits are insufficient."},
                        {"limitations", "May be truncated by max_bytes."},
                        {"keywords", json::array({"search", "results"})},
                        {"entities", json::array({"repository_search"})},
                    }},
                }
            });
        } else if (command == "list_memories") {
            response = make_status_payload("memories_listed");
            response["memories"] = json::array({
                {
                    {"id", "mem-1"},
                    {"kind", "decision"},
                    {"scope", "session"},
                    {"summary", "Tool results should stay host-owned."},
                    {"session_id", "session-a"},
                    {"project_id", "project-a"},
                    {"turn_id", "turn-a"},
                    {"created_at", 1},
                }
            });
        } else if (command == "list_plans") {
            response = make_status_payload("plans_listed");
            response["plans"] = json::array({
                {
                    {"plan_id", "plan-1"},
                    {"purpose", "Verify daemon session flow."},
                    {"goal", "Run one admin smoke."},
                    {"status", "active"},
                    {"scope", "session"},
                    {"session_id", "session-a"},
                    {"project_id", "project-a"},
                    {"turn_id", "turn-a"},
                    {"active_step_id", "step-1"},
                    {"next_action", "Inspect resource output."},
                    {"version", 2},
                    {"step_count", 3},
                    {"observation_count", 1},
                }
            });
        } else if (command == "read_resource") {
            response = {
                {"ok", true},
                {"event", "resource_read"},
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
                {"resource", {
                    {"resource_id", "r-1"},
                    {"uri", request.value("uri", "agent-resource://resource/r-1")},
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
                        {"entities", json::array({"repository_search"})},
                    }},
                }},
                {"content", R"({"results":["stub"]})"},
            };
        } else if (command == "drain") {
            response = make_lifecycle_payload("drain", "draining", true, false);
        } else if (command == "shutdown") {
            response = make_lifecycle_payload("shutdown", "stopping", false, true);
        } else if (command == "reset_session") {
            response = make_lifecycle_payload("session_reset", "ready", true, false);
        } else if (command == "close_session") {
            response = make_lifecycle_payload("session_closed", "ready", true, false);
        } else {
            response = {
                {"ok", false},
                {"event", "error"},
                {"error", "unsupported command: " + command},
            };
        }

        if (!write_agent_daemon_jsonl_message(stdout, response, error)) {
            std::fprintf(stderr, "fake-daemon: failed to write response: %s\n", error.c_str());
            return 1;
        }
        if (command == "shutdown") {
            return 0;
        }
    }

    if (!error.empty()) {
        std::fprintf(stderr, "fake-daemon: read loop failed: %s\n", error.c_str());
        return 1;
    }
    return 0;
}
