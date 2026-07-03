#pragma once

#include "agent-runtime-session-manager.h"

#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <memory>
#include <optional>
#include <string>

enum class common_agent_daemon_command_type {
    run_turn,
    shutdown,
};

struct common_agent_daemon_runtime {
    std::unique_ptr<common_memory_store> memory_store;
    std::unique_ptr<common_plan_store> plan_store;
    common_agent_runtime_host_mode default_mode = common_agent_runtime_host_mode::chat;
    std::unique_ptr<common_agent_runtime_daemon_host> host;
};

struct common_agent_daemon_command {
    std::string request_id;
    common_agent_daemon_command_type type = common_agent_daemon_command_type::run_turn;
    std::optional<common_agent_runtime_daemon_turn_request> turn;
};

struct common_agent_daemon_command_result {
    bool ok = false;
    std::string request_id;
    std::string event;
    common_agent_runtime_daemon_turn_result turn_result;
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

private:
    common_agent_daemon_runtime runtime;
    bool shutdown_requested_flag = false;
};
