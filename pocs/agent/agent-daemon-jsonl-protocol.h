#pragma once

#include <cstdio>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

struct agent_daemon_jsonl_turn_request {
    std::string prompt;
    std::string session_id;
    std::string namespace_id;
    std::string project_id;
    std::string turn_id;
    std::string memory_scope;
    std::string plan_scope;
    int n_predict = 0;
    std::string mode = "chat";
    std::optional<size_t> turn_timeout_ms;
    std::optional<uint32_t> inference_step_timeout_ms;
    std::optional<uint32_t> tool_timeout_ms;
    std::optional<uint32_t> mcp_connect_timeout_ms;
    std::optional<uint32_t> mcp_request_timeout_ms;
    std::optional<uint32_t> mcp_shutdown_timeout_ms;
};

struct agent_daemon_jsonl_status_request {};

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

struct agent_daemon_jsonl_turn_response {
    bool ok = false;
    std::string event;
    bool cancelled = false;
    std::string response;
    std::string error;
    std::string failure_class;
    std::string response_generation_status;
    std::string response_stop_reason;
    bool runtime_reused = false;
    int event_count = 0;
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
    std::string last_turn_id;
    std::string last_turn_phase;
    std::string last_turn_disposition;
};

struct agent_daemon_jsonl_status_response {
    bool ok = false;
    std::string event;
    std::string state;
    bool live = false;
    bool ready = false;
    bool worker_running = false;
    bool accepting_commands = false;
    bool shutdown_requested = false;
    int sessions = 0;
    int queued_commands = 0;
    int max_queue_size = 0;
    int queue_capacity_remaining = 0;
    std::string active_request_id;
    std::string active_turn_id;
    std::string active_turn_phase;
    std::string active_turn_disposition;
    std::vector<agent_daemon_jsonl_session_status> session_keys;
    nlohmann::ordered_json payload;
    std::string error;
};

struct agent_daemon_jsonl_event_response {
    bool ok = false;
    std::string event;
    std::string error;
};

struct agent_daemon_jsonl_lifecycle_response {
    bool ok = false;
    std::string event;
    std::string target_request_id;
    std::string target_turn_id;
    agent_daemon_jsonl_status_response status;
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

nlohmann::ordered_json make_agent_daemon_jsonl_turn_request(
    const agent_daemon_jsonl_turn_request & request);

nlohmann::ordered_json make_agent_daemon_jsonl_status_request(
    const agent_daemon_jsonl_status_request & request = {});

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
