#include "agent-daemon-dispatcher.h"

#include <utility>

common_agent_daemon_dispatcher::common_agent_daemon_dispatcher(
        common_agent_daemon_runtime runtime,
        size_t max_queue_size)
    : service(std::move(runtime))
    , max_queue_size(max_queue_size) {
    worker_running = true;
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
    {
        std::lock_guard<std::mutex> lock(mutex);
        worker_running = false;
    }
}

bool common_agent_daemon_dispatcher::execute(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error) {
    if (command.type == common_agent_daemon_command_type::cancel_turn) {
        return execute_cancel_turn(command, result, error);
    }

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

size_t common_agent_daemon_dispatcher::queued_command_count() const {
    std::lock_guard<std::mutex> lock(mutex);
    return queue.size();
}

bool common_agent_daemon_dispatcher::execute_cancel_turn(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error) {
    result = {};
    result.request_id = command.request_id;
    result.target_request_id = command.target_request_id;
    result.target_turn_id = command.target_turn_id;

    std::shared_ptr<queued_command> cancelled_item;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = queue.begin();
        for (; it != queue.end(); ++it) {
            const auto & queued = *it;
            if (queued->command.type != common_agent_daemon_command_type::run_turn) {
                continue;
            }

            const bool request_match =
                !command.target_request_id.empty() &&
                queued->command.request_id == command.target_request_id;
            const bool turn_match =
                !command.target_turn_id.empty() &&
                queued->command.turn.has_value() &&
                queued->command.turn->turn_id == command.target_turn_id;
            if (!request_match && !turn_match) {
                continue;
            }

            cancelled_item = queued;
            queue.erase(it);
            break;
        }

        if (!cancelled_item) {
            if ((!command.target_request_id.empty() && command.target_request_id == active_request_id) ||
                    (!command.target_turn_id.empty() && command.target_turn_id == active_turn_id)) {
                error = "active turn cancellation is not supported yet";
                result.error = error;
                return false;
            }
        }
    }

    if (!cancelled_item) {
        error = "target turn is not queued";
        result.error = error;
        return false;
    }

    queued_result queued;
    queued.ok = false;
    queued.result.request_id = cancelled_item->command.request_id;
    queued.result.turn_result.cancelled = true;
    queued.result.turn_result.error = "turn cancelled before execution";
    queued.result.error = queued.result.turn_result.error;
    cancelled_item->promise.set_value(std::move(queued));

    result.ok = true;
    result.event = "turn_cancelled";
    error.clear();
    return true;
}

bool common_agent_daemon_dispatcher::populate_status_locked(
        common_agent_daemon_command_result & result,
        std::string & error) const {
    const size_t queued_count = queue.size();
    const std::string active_request = active_request_id;
    const std::string active_turn = active_turn_id;

    if (!service.populate_status(result, error)) {
        return false;
    }

    result.active_request_id = active_request;
    result.active_turn_id = active_turn;
    result.queued_command_count = queued_count;
    result.worker_running = worker_running;
    result.accepting_commands = accepting_commands;
    result.shutdown_requested = service.shutdown_requested();
    result.max_queue_size = max_queue_size;
    result.queue_capacity_remaining =
        max_queue_size > queued_count ? (max_queue_size - queued_count) : 0;
    result.state = service.shutdown_requested()
        ? "draining"
        : (worker_running ? "ready" : "stopped");
    result.event = "status";
    result.live = worker_running;
    result.ready = result.ready && accepting_commands && worker_running;
    return true;
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
            if (item->command.type == common_agent_daemon_command_type::run_turn) {
                active_request_id = item->command.request_id;
                active_turn_id =
                    item->command.turn.has_value()
                        ? item->command.turn->turn_id
                        : std::string();
            } else {
                active_request_id.clear();
                active_turn_id.clear();
            }
        }

        queued_result queued;
        if (item->command.type == common_agent_daemon_command_type::get_status) {
            std::lock_guard<std::mutex> lock(mutex);
            queued.ok = populate_status_locked(queued.result, queued.error);
        } else {
            queued.ok = service.execute(item->command, queued.result, queued.error);
        }
        if (!queued.error.empty() && queued.result.error.empty()) {
            queued.result.error = queued.error;
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            active_request_id.clear();
            active_turn_id.clear();
            if (service.shutdown_requested()) {
                accepting_commands = false;
                stop_requested = true;
            }
        }

        item->promise.set_value(std::move(queued));
        condition.notify_all();
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        worker_running = false;
    }
}
