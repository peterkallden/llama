#pragma once

#include "agent-daemon-lifecycle.h"
#include "agent-daemon-events.h"
#include "agent-runtime-session-manager.h"
#include "agent-resource-store.h"

#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

enum class common_agent_daemon_command_type {
    run_turn,
    cancel_turn,
    list_sessions,
    get_session,
    list_resources,
    list_memories,
    list_plans,
    reset_session,
    close_session,
    read_resource,
    get_status,
    drain,
    shutdown,
};

struct common_agent_daemon_runtime {
    std::unique_ptr<common_memory_store> memory_store;
    std::unique_ptr<common_plan_store> plan_store;
    std::unique_ptr<agent_resource_store> resource_store;
    common_agent_runtime_host_mode default_mode = common_agent_runtime_host_mode::chat;
    std::unique_ptr<common_agent_runtime_session_manager> host;
};

struct common_agent_daemon_turn_payload {
    common_agent_runtime_session_manager_turn_request request;
};

struct common_agent_daemon_session_payload {
    common_agent_runtime_session_key key;
};

struct common_agent_daemon_cancel_payload {
    std::string target_request_id;
    std::string target_turn_id;
};

struct common_agent_daemon_resource_payload {
    std::string uri;
    agent_resource_read_authority authority;
    size_t max_bytes = 8192;
};

struct common_agent_daemon_scope_payload {
    agent_resource_read_authority authority;
};

struct common_agent_daemon_command {
    std::string request_id;
    common_agent_daemon_command_type type = common_agent_daemon_command_type::run_turn;
    std::optional<common_agent_daemon_turn_payload> turn;
    std::optional<common_agent_daemon_session_payload> session;
    std::optional<common_agent_daemon_cancel_payload> cancel;
    std::optional<common_agent_daemon_resource_payload> resource;
    std::optional<common_agent_daemon_scope_payload> scope;
};

struct common_agent_daemon_active_turn_status {
    std::string request_id;
    std::string turn_id;
    std::string phase;
    std::string disposition;
    bool cancellation_requested = false;
};

struct common_agent_daemon_status {
    common_agent_daemon_state state = common_agent_daemon_state::starting;
    bool live = false;
    bool ready = false;
    bool worker_running = false;
    bool accepting_commands = false;
    bool shutdown_requested = false;
    size_t queued_command_count = 0;
    size_t max_queue_size = 0;
    size_t queue_capacity_remaining = 0;
    size_t session_count = 0;
    std::optional<common_agent_daemon_active_turn_status> active_turn;
    std::string active_request_id;
    std::string active_turn_id;
    std::string active_turn_phase;
    std::string active_turn_disposition;
    bool active_cancel_requested = false;
    std::vector<common_agent_runtime_session_descriptor> sessions;
    bool session_snapshot_populated = false;
};

enum class common_agent_daemon_response_kind {
    turn,
    status,
    lifecycle,
    resource,
    listing,
};

struct common_agent_daemon_resource_read_result {
    agent_resource_descriptor resource;
    std::string content;
};

struct common_agent_daemon_memory_summary {
    std::string id;
    std::string kind;
    std::string scope;
    std::string summary;
    std::string session_id;
    std::string project_id;
    std::string turn_id;
    int64_t created_at = 0;
};

struct common_agent_daemon_plan_summary {
    std::string plan_id;
    std::string purpose;
    std::string goal;
    std::string status;
    std::string scope;
    std::string session_id;
    std::string project_id;
    std::string turn_id;
    std::string active_step_id;
    std::string next_action;
    uint64_t version = 0;
    size_t step_count = 0;
    size_t observation_count = 0;
};

struct common_agent_daemon_listing_result {
    std::vector<agent_resource_descriptor> resources;
    std::vector<common_agent_daemon_memory_summary> memories;
    std::vector<common_agent_daemon_plan_summary> plans;
};

struct common_agent_daemon_command_result {
    bool ok = false;
    std::string request_id;
    common_agent_daemon_response_kind response_kind = common_agent_daemon_response_kind::turn;
    std::string event;
    std::string target_request_id;
    std::string target_turn_id;
    common_agent_daemon_status status;
    size_t daemon_event_count = 0;
    std::vector<common_agent_daemon_event> events;
    common_agent_runtime_session_manager_turn_result turn_result;
    common_agent_daemon_resource_read_result resource_result;
    common_agent_daemon_listing_result listing_result;
    std::string error;
};

class common_agent_daemon_service {
public:
    explicit common_agent_daemon_service(common_agent_daemon_runtime runtime);

    bool execute(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error);

    bool shutdown_requested() const { return shutdown_requested_flag; }
    common_agent_runtime_host_mode default_mode() const { return runtime.default_mode; }
    common_agent_daemon_state state() const { return state_value; }
    common_agent_daemon_shutdown_mode shutdown_mode() const { return shutdown_mode_value; }
    std::vector<common_agent_runtime_session_descriptor> list_sessions() const;

    void mark_stopping();
    void mark_stopped();

    bool populate_status(
        common_agent_daemon_command_result & result,
        std::string & error) const;

    std::optional<common_agent_runtime_active_turn_descriptor> describe_active_turn() const;

    bool request_cancel_active_turn(
        const std::string & target_request_id,
        const std::string & target_turn_id,
        common_agent_runtime_active_turn_descriptor & active_turn,
        std::string & error);

    void emit_internal_event(
        common_agent_daemon_event_type type,
        const std::string & request_id,
        const std::string & turn_id,
        const std::string & detail);

    std::vector<common_agent_daemon_event> take_internal_events();

private:
    void initialize_command_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result) const;

    void initialize_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result) const;

    void initialize_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result) const;

    bool fail_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        std::string daemon_event_type) const;

    bool fail_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        std::string daemon_event_type) const;

    bool succeed_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        std::string daemon_event_type,
        std::string detail = {}) const;

    common_agent_daemon_runtime runtime;
    common_agent_daemon_state state_value = common_agent_daemon_state::starting;
    common_agent_daemon_shutdown_mode shutdown_mode_value = common_agent_daemon_shutdown_mode::drain;
    bool shutdown_requested_flag = false;
    mutable std::mutex event_mutex;
    std::vector<common_agent_daemon_event> pending_events;
    uint64_t next_event_sequence = 1;
};
