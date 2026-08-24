#pragma once

#include "../../common/runtime-resource.h"
#include "../../../common/agent/agent-continuation.h"
#include "../../../common/agent/turn-summary.h"

#include "agent-daemon-events.h"
#include "../../../common/agent/protocol/agent-jsonl.h"

#include <cstdio>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

using agent_daemon_jsonl_turn_request = common_agent_jsonl_turn_request;

struct agent_daemon_jsonl_status_request {};

struct agent_daemon_jsonl_list_sessions_request {};

struct agent_daemon_jsonl_get_session_request {
    std::string session_id;
    std::string namespace_id;
};

struct agent_daemon_jsonl_scope_request {
    std::string session_id;
    std::string namespace_id;
    std::string project_id;
    std::string turn_id;
};

struct agent_daemon_jsonl_list_resources_request : agent_daemon_jsonl_scope_request {};

struct agent_daemon_jsonl_list_memories_request : agent_daemon_jsonl_scope_request {};

struct agent_daemon_jsonl_list_plans_request : agent_daemon_jsonl_scope_request {};

struct agent_daemon_jsonl_read_resource_request {
    std::string uri;
    std::string session_id;
    std::string namespace_id;
    std::string project_id;
    std::string turn_id;
    size_t max_bytes = 8192;
};

struct agent_daemon_jsonl_put_resource_request {
    std::string name;
    std::string description;
    std::string mime_type = "text/plain";
    std::string text;
    std::string scope = "turn";
    std::string namespace_id;
    std::string session_id;
    std::string project_id;
    std::string turn_id;
};

struct agent_daemon_jsonl_drain_request {};

struct agent_daemon_jsonl_shutdown_request {};

enum class agent_daemon_jsonl_session_command {
    reset,
    close,
};

struct agent_daemon_jsonl_session_request {
    agent_daemon_jsonl_session_command command = agent_daemon_jsonl_session_command::reset;
    std::string session_id;
    std::string namespace_id;
};

struct agent_daemon_jsonl_cancel_request {
    std::string target_request_id;
    std::string target_turn_id;
};

struct agent_daemon_jsonl_ready_response {
    std::string default_mode;
    int protocol_version = 0;
    std::vector<std::string> capabilities;
};

using agent_daemon_jsonl_event_entry = common_agent_jsonl_event_entry;

struct agent_daemon_jsonl_turn_response {
    bool ok = false;
    std::string event;
    int daemon_event_count = 0;
    std::vector<agent_daemon_jsonl_event_entry> events;
    bool cancelled = false;
    std::string response;
    std::string error;
    std::string failure_class;
    std::string response_generation_status;
    std::string response_stop_reason;
    bool runtime_reused = false;
    int event_count = 0;
    std::optional<common_agent_continuation_checkpoint> continuation_checkpoint;
    std::optional<common_agent_turn_summary> turn_summary;
};

struct agent_daemon_jsonl_session_status {
    std::string namespace_id;
    std::string session_id;
    std::string project_id;
    std::string memory_scope;
    std::string plan_scope;
    std::string policy_pack_id;
    std::string lane_state;
    int queued_turn_count = 0;
    bool has_active_turn = false;
    std::string active_request_id;
    std::string active_turn_id;
    std::string active_turn_phase;
    std::string active_turn_disposition;
    bool active_cancel_requested = false;
    std::string pending_operation_kind;
    std::string pending_operation_detail;
    std::string last_turn_id;
    std::string last_turn_phase;
    std::string last_turn_disposition;
};

struct agent_daemon_jsonl_status_response {
    bool ok = false;
    std::string event;
    int daemon_event_count = 0;
    std::vector<agent_daemon_jsonl_event_entry> events;
    std::string state;
    bool live = false;
    bool ready = false;
    nlohmann::ordered_json readiness;
    bool worker_running = false;
    int worker_count = 1;
    int workers_running = 0;
    bool accepting_commands = false;
    bool shutdown_requested = false;
    int sessions = 0;
    int queued_commands = 0;
    int max_queue_size = 0;
    int queue_capacity_remaining = 0;
    uint64_t commands_accepted = 0;
    uint64_t commands_completed = 0;
    uint64_t commands_failed = 0;
    uint64_t turns_completed = 0;
    uint64_t tools_completed = 0;
    std::string active_request_id;
    std::string active_turn_id;
    std::string active_turn_phase;
    std::string active_turn_disposition;
    std::string active_pending_operation_kind;
    std::string active_pending_operation_detail;
    std::vector<agent_daemon_jsonl_session_status> session_keys;
    nlohmann::ordered_json payload;
    std::string error;
};

struct agent_daemon_jsonl_event_response {
    bool ok = false;
    std::string event;
    int daemon_event_count = 0;
    std::vector<agent_daemon_jsonl_event_entry> events;
    std::string error;
};

struct agent_daemon_jsonl_lifecycle_response {
    bool ok = false;
    std::string event;
    int daemon_event_count = 0;
    std::vector<agent_daemon_jsonl_event_entry> events;
    std::string target_request_id;
    std::string target_turn_id;
    agent_daemon_jsonl_status_response status;
    std::string error;
};

struct agent_daemon_jsonl_resource_response {
    bool ok = false;
    std::string event;
    int daemon_event_count = 0;
    std::vector<agent_daemon_jsonl_event_entry> events;
    agent_resource_descriptor resource;
    std::string content;
    agent_daemon_jsonl_status_response status;
    nlohmann::ordered_json payload;
    std::string error;
};

struct agent_daemon_jsonl_listing_response {
    bool ok = false;
    std::string event;
    int daemon_event_count = 0;
    std::vector<agent_daemon_jsonl_event_entry> events;
    agent_daemon_jsonl_status_response status;
    nlohmann::ordered_json payload;
    std::string error;
};

bool read_agent_daemon_jsonl_message(
    FILE * stream,
    nlohmann::ordered_json & out,
    std::string & error);

bool write_agent_daemon_jsonl_message(
        FILE * stream,
        const nlohmann::ordered_json & message,
        std::string & error);

nlohmann::ordered_json make_agent_daemon_jsonl_event_message(
        const std::string & subscription_id,
        const common_agent_event_stream_delivery & delivery);

nlohmann::ordered_json make_agent_daemon_jsonl_turn_request(
    const agent_daemon_jsonl_turn_request & request);

nlohmann::ordered_json make_agent_daemon_jsonl_status_request(
    const agent_daemon_jsonl_status_request & request = {});

nlohmann::ordered_json make_agent_daemon_jsonl_list_sessions_request(
    const agent_daemon_jsonl_list_sessions_request & request = {});

nlohmann::ordered_json make_agent_daemon_jsonl_get_session_request(
    const agent_daemon_jsonl_get_session_request & request);

nlohmann::ordered_json make_agent_daemon_jsonl_list_resources_request(
    const agent_daemon_jsonl_list_resources_request & request);

nlohmann::ordered_json make_agent_daemon_jsonl_list_memories_request(
    const agent_daemon_jsonl_list_memories_request & request);

nlohmann::ordered_json make_agent_daemon_jsonl_list_plans_request(
    const agent_daemon_jsonl_list_plans_request & request);

nlohmann::ordered_json make_agent_daemon_jsonl_read_resource_request(
    const agent_daemon_jsonl_read_resource_request & request);

nlohmann::ordered_json make_agent_daemon_jsonl_put_resource_request(
    const agent_daemon_jsonl_put_resource_request & request);

nlohmann::ordered_json make_agent_daemon_jsonl_drain_request(
    const agent_daemon_jsonl_drain_request & request = {});

nlohmann::ordered_json make_agent_daemon_jsonl_shutdown_request(
    const agent_daemon_jsonl_shutdown_request & request = {});

nlohmann::ordered_json make_agent_daemon_jsonl_session_request(
    const agent_daemon_jsonl_session_request & request);

nlohmann::ordered_json make_agent_daemon_jsonl_reset_session_request(
    const std::string & session_id,
    const std::string & namespace_id);

nlohmann::ordered_json make_agent_daemon_jsonl_close_session_request(
    const std::string & session_id,
    const std::string & namespace_id);

nlohmann::ordered_json make_agent_daemon_jsonl_cancel_request(
    const agent_daemon_jsonl_cancel_request & request);

bool parse_agent_daemon_jsonl_ready_response(
    const nlohmann::ordered_json & message,
    agent_daemon_jsonl_ready_response & response,
    std::string & error);

bool parse_agent_daemon_jsonl_turn_response(
    const nlohmann::ordered_json & message,
    agent_daemon_jsonl_turn_response & response,
    std::string & error);

bool parse_agent_daemon_jsonl_resource_response(
    const nlohmann::ordered_json & message,
    agent_daemon_jsonl_resource_response & response,
    std::string & error);

bool parse_agent_daemon_jsonl_listing_response(
    const nlohmann::ordered_json & message,
    agent_daemon_jsonl_listing_response & response,
    std::string & error);

bool parse_agent_daemon_jsonl_status_response(
    const nlohmann::ordered_json & message,
    agent_daemon_jsonl_status_response & response,
    std::string & error);

bool parse_agent_daemon_jsonl_event_response(
    const nlohmann::ordered_json & message,
    agent_daemon_jsonl_event_response & response,
    std::string & error);

bool parse_agent_daemon_jsonl_lifecycle_response(
    const nlohmann::ordered_json & message,
    agent_daemon_jsonl_lifecycle_response & response,
    std::string & error);

bool parse_agent_daemon_jsonl_event_response(
    const nlohmann::ordered_json & message,
    const std::string & expected_event,
    std::string & error);
