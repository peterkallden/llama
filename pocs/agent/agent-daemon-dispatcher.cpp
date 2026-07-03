#include "agent-daemon-dispatcher.h"

#include <utility>

common_agent_daemon_dispatcher::common_agent_daemon_dispatcher(
        common_agent_daemon_runtime runtime,
        size_t max_queue_size)
    : service(std::move(runtime))
    , max_queue_size(max_queue_size) {
    worker = std::thread([this]() {
        worker_loop();
    });
}

common_agent_daemon_dispatcher::~common_agent_daemon_dispatcher() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        accepting_commands = false;
        stop_requested = true;
    }
    condition.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

bool common_agent_daemon_dispatcher::execute(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error) {
    auto item = std::make_shared<queued_command>();
    item->command = command;
    auto future = item->promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!accepting_commands) {
            error = "daemon dispatcher is not accepting new commands";
            result = {};
            result.request_id = command.request_id;
            result.error = error;
            return false;
        }
        if (queue.size() >= max_queue_size) {
            error = "daemon command queue is full";
            result = {};
            result.request_id = command.request_id;
            result.error = error;
            return false;
        }
        queue.push_back(item);
    }

    condition.notify_one();

    auto queued = future.get();
    result = std::move(queued.result);
    error = std::move(queued.error);
    return queued.ok;
}

bool common_agent_daemon_dispatcher::shutdown_requested() const {
    std::lock_guard<std::mutex> lock(mutex);
    return service.shutdown_requested();
}

common_agent_runtime_host_mode common_agent_daemon_dispatcher::default_mode() const {
    return service.default_mode();
}

void common_agent_daemon_dispatcher::worker_loop() {
    while (true) {
        std::shared_ptr<queued_command> item;
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [&]() {
                return stop_requested || !queue.empty();
            });
            if (queue.empty()) {
                if (stop_requested) {
                    break;
                }
                continue;
            }

            item = queue.front();
            queue.pop_front();
        }

        queued_result queued;
        queued.ok = service.execute(item->command, queued.result, queued.error);
        if (!queued.error.empty() && queued.result.error.empty()) {
            queued.result.error = queued.error;
        }

        if (service.shutdown_requested()) {
            std::lock_guard<std::mutex> lock(mutex);
            accepting_commands = false;
            stop_requested = true;
        }

        item->promise.set_value(std::move(queued));
        condition.notify_all();
    }
}
