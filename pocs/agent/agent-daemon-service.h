#pragma once

#include "agent-daemon-lifecycle.h"
#include "agent-runtime-session-manager.h"
#include "agent-resource-store.h"

#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

enum class common_agent_daemon_command_type {
    run_turn,
    cancel_turn,
    reset_session,
    close_session,
    get_status,
    shutdown,
};

struct common_agent_daemon_runtime {
    std::unique_ptr<common_memory_store> memory_store;
    std::unique_ptr<common_plan_store> plan_store;
    std::unique_ptr<agent_resource_store> resource_store;
    common_agent_runtime_host_mode default_mode = common_agent_runtime_host_mode::chat;
    std::unique_ptr<common_agent_runtime_session_manager> host;
};

struct common_agent_daemon_command {
    std::string request_id;
    common_agent_daemon_command_type type = common_agent_daemon_command_type::run_turn;
    std::optional<common_agent_runtime_session_manager_turn_request> turn;
    std::optional<common_agent_runtime_session_key> session;
    std::string target_request_id;
    std::string target_turn_id;
};

struct common_agent_daemon_event {
    std::string type;
    std::string request_id;
    std::string turn_id;
    std::string detail;
};

struct common_agent_daemon_command_result {
    bool ok = false;
    std::string request_id;
    std::string event;
    std::string target_request_id;
    std::string target_turn_id;
    std::string active_request_id;
    std::string active_turn_id;
    std::string state;
    bool live = false;
    bool ready = false;
    bool worker_running = false;
    bool accepting_commands = false;
    bool shutdown_requested = false;
    size_t queued_command_count = 0;
    size_t max_queue_size = 0;
    size_t queue_capacity_remaining = 0;
    size_t session_count = 0;
    size_t daemon_event_count = 0;
    std::vector<common_agent_runtime_session_descriptor> sessions;
    std::vector<common_agent_daemon_event> events;
    common_agent_runtime_session_manager_turn_result turn_result;
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

    void mark_stopping();
    void mark_stopped();

    bool populate_status(
        common_agent_daemon_command_result & result,
        std::string & error) const;

private:
    common_agent_daemon_runtime runtime;
    common_agent_daemon_state state_value = common_agent_daemon_state::starting;
    common_agent_daemon_shutdown_mode shutdown_mode_value = common_agent_daemon_shutdown_mode::drain;
    bool shutdown_requested_flag = false;
};
