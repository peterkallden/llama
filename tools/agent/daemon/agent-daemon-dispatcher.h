#pragma once

#include "agent-daemon-service.h"
#include "../runtime/agent-runtime-control.h"
#include "agent/adaptation/flydelta/flydelta-worker.h"
#include "agent/adaptation/flydelta/flydelta-worker-budget.h"
#include "agent/adaptation/flydelta/flydelta-model-adapter.h"

#include <condition_variable>
#include <chrono>
#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

struct daemon_options;

struct common_agent_daemon_flydelta_worker_config {
    bool enabled = false;
    size_t worker_count = 0;
    std::filesystem::path queue_root;
    common_flydelta_experiment_queue_limits queue_limits;
    // Host-owned model adapter. The worker only consumes its bounded callback;
    // model contexts and reference resolution remain inside the adapter.
    std::shared_ptr<const common_flydelta_model_adapter> model_adapter;
    common_flydelta_experiment_worker_callback callback;
    // Optional host scheduler seam. FlyDelta decides next_action; this hook
    // may enqueue one reference-only follow-up slice after the current job
    // has completed. It must not execute the next slice recursively.
    std::function<bool(
            const common_flydelta_experiment_worker_report &,
            std::string &)> schedule_next_action;
    std::chrono::milliseconds poll_interval{250};
};

class common_agent_daemon_dispatcher {
public:
    explicit common_agent_daemon_dispatcher(
        common_agent_daemon_runtime runtime,
        size_t max_queue_size = 8,
        size_t worker_count = 1,
        common_agent_daemon_flydelta_worker_config flydelta_config = {});
    ~common_agent_daemon_dispatcher();

    bool execute(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error);

    bool shutdown_requested() const;
    common_agent_runtime_host_mode default_mode() const;
    size_t queued_command_count() const;
    size_t max_queue_size_value() const { return max_queue_size; }
    size_t worker_count_value() const { return worker_count; }
    size_t total_worker_count_value() const { return total_worker_count; }
    size_t flydelta_worker_count_value() const { return flydelta_worker_count; }
    size_t flydelta_workers_running_value() const;
    std::string subscribe_events(common_agent_event_stream_subscription subscription);
    void unsubscribe_events(const std::string & subscription_id);
    common_agent_event_stream_wait_status wait_for_event(
        const std::string & subscription_id,
        common_agent_event_stream_delivery & delivery,
        std::chrono::milliseconds timeout);
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
    void flydelta_worker_loop();

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::shared_ptr<queued_command>> queue;
    common_agent_daemon_flydelta_worker_config flydelta_config;
    common_agent_daemon_service service;
    std::vector<std::thread> workers;
    std::vector<std::thread> flydelta_workers;
    size_t max_queue_size = 0;
    size_t total_worker_count = 1;
    size_t worker_count = 1;
    size_t flydelta_worker_count = 0;
    size_t workers_running = 0;
    size_t flydelta_workers_running = 0;
    bool worker_running = false;
    bool accepting_commands = true;
    bool stop_requested = false;
    uint64_t commands_accepted = 0;
    uint64_t commands_completed = 0;
    uint64_t commands_failed = 0;
    uint64_t turns_completed = 0;
    uint64_t tools_completed = 0;
};
