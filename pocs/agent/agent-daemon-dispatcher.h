#pragma once

#include "agent-daemon-service.h"

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

private:
    struct queued_result {
        bool ok = false;
        common_agent_daemon_command_result result;
        std::string error;
    };

    struct queued_command {
        common_agent_daemon_command command;
        std::promise<queued_result> promise;
    };

    void worker_loop();

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::shared_ptr<queued_command>> queue;
    common_agent_daemon_service service;
    std::thread worker;
    size_t max_queue_size = 0;
    bool accepting_commands = true;
    bool stop_requested = false;
};
