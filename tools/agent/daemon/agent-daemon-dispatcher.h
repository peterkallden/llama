#pragma once

#include "agent-daemon-service.h"
#include "agent-runtime-control.h"

#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class common_agent_daemon_dispatcher {
public:
    explicit common_agent_daemon_dispatcher(
        common_agent_daemon_runtime runtime,
        size_t max_queue_size = 8);
    ~common_agent_daemon_dispatcher();

    bool execute(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error);

    bool shutdown_requested() const;
    common_agent_runtime_host_mode default_mode() const;
    size_t queued_command_count() const;
    size_t max_queue_size_value() const { return max_queue_size; }

private:
    struct queued_result {
        bool ok = false;
        common_agent_daemon_command_result result;
        std::string error;
    };

    struct queued_command {
        common_agent_daemon_command command;
        std::vector<common_agent_daemon_event> events;
        std::promise<queued_result> promise;
    };

    bool execute_cancel_turn(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error);

    bool execute_session_lifecycle(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error);

    bool populate_status_locked(
        common_agent_daemon_command_result & result,
        std::string & error) const;

    void initialize_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result) const;

    void initialize_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result) const;

    bool fail_lifecycle_result_locked(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        common_agent_daemon_event_type event_type,
        std::string turn_id = {}) const;

    bool succeed_lifecycle_result_locked(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        common_agent_daemon_event_type event_type,
        std::string detail = {},
        std::string turn_id = {}) const;

    void cancel_queued_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string error) const;

    void reject_queued_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string event,
        common_agent_daemon_event_type event_type,
        std::string error) const;

    void finalize_lifecycle_result_locked(
        common_agent_daemon_command_result & result) const;

    void fill_status_snapshot_locked(
        common_agent_daemon_status & status) const;

    void worker_loop();

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::shared_ptr<queued_command>> queue;
    common_agent_daemon_service service;
    std::thread worker;
    size_t max_queue_size = 0;
    bool worker_running = false;
    bool accepting_commands = true;
    bool stop_requested = false;
};
