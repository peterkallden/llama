#include "agent-daemon-dispatcher.h"

#include <utility>

namespace {

void assign_active_turn_status(
        common_agent_daemon_status & status,
        const common_agent_runtime_active_turn_descriptor & active_turn) {
    status.active_turn = common_agent_daemon_active_turn_status{
        active_turn.request_id,
        active_turn.turn_id,
        active_turn.phase,
        active_turn.disposition,
        active_turn.cancellation_requested,
    };
    status.active_request_id = active_turn.request_id;
    status.active_turn_id = active_turn.turn_id;
    status.active_turn_phase = active_turn.phase;
    status.active_turn_disposition = active_turn.disposition;
    status.active_cancel_requested = active_turn.cancellation_requested;
}

void append_daemon_event(
        common_agent_daemon_command_result & result,
        std::string type,
        std::string request_id,
        std::string turn_id,
        std::string detail = {}) {
    result.events.push_back(common_agent_daemon_event{
        std::move(type),
        std::move(request_id),
        std::move(turn_id),
        std::move(detail),
    });
    result.daemon_event_count = result.events.size();
}

std::string command_turn_id(const common_agent_daemon_command & command) {
    if (!command.turn.has_value()) {
        return {};
    }
    return command.turn->request.turn.turn_id;
}

} // namespace

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
        service.mark_stopping();
    }
    condition.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        worker_running = false;
        service.mark_stopped();
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
            return fail_lifecycle_result_locked(
                command,
                result,
                error,
                "command_rejected",
                "command.rejected",
                command_turn_id(command));
        }
        if (queue.size() >= max_queue_size) {
            error = "daemon command queue is full";
            return fail_lifecycle_result_locked(
                command,
                result,
                error,
                "command_rejected",
                "command.rejected",
                command_turn_id(command));
        }
        item->events.push_back(common_agent_daemon_event{
            "command.queued",
            command.request_id,
            command_turn_id(command),
            {},
        });
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
    initialize_lifecycle_result(command, result);
    result.target_request_id = command.cancel.has_value() ? command.cancel->target_request_id : std::string();
    result.target_turn_id = command.cancel.has_value() ? command.cancel->target_turn_id : std::string();

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
                command.cancel.has_value() &&
                !command.cancel->target_request_id.empty() &&
                queued->command.request_id == command.cancel->target_request_id;
            const bool turn_match =
                command.cancel.has_value() &&
                !command.cancel->target_turn_id.empty() &&
                queued->command.turn.has_value() &&
                queued->command.turn->request.turn.turn_id == command.cancel->target_turn_id;
            if (!request_match && !turn_match) {
                continue;
            }

            cancelled_item = queued;
            queue.erase(it);
            break;
        }

        if (!cancelled_item) {
            common_agent_runtime_active_turn_descriptor active_turn;
            if (service.request_cancel_active_turn(
                    command.cancel.has_value() ? command.cancel->target_request_id : std::string(),
                    command.cancel.has_value() ? command.cancel->target_turn_id : std::string(),
                    active_turn,
                    error)) {
                return succeed_lifecycle_result_locked(
                    command,
                    result,
                    error,
                    "turn_cancel_requested",
                    "turn.cancel_requested",
                    "active turn cancellation requested",
                    !command.cancel.has_value() || command.cancel->target_turn_id.empty()
                        ? active_turn.turn_id
                        : command.cancel->target_turn_id);
            }
            error.clear();
        }
    }

    if (!cancelled_item) {
        error = "target turn is not queued";
        std::lock_guard<std::mutex> lock(mutex);
        return fail_lifecycle_result_locked(
            command,
            result,
            error,
            "turn_cancel_rejected",
            "turn.cancel_rejected",
            command.cancel.has_value() ? command.cancel->target_turn_id : std::string());
    }

    queued_result queued;
    queued.ok = false;
    cancel_queued_turn_result(
        cancelled_item->command,
        queued.result,
        "turn cancelled before execution");
    queued.error = queued.result.error;
    {
        std::lock_guard<std::mutex> lock(mutex);
        fill_status_snapshot_locked(queued.result.status);
    }
    cancelled_item->promise.set_value(std::move(queued));

    {
        std::lock_guard<std::mutex> lock(mutex);
        return succeed_lifecycle_result_locked(
            command,
            result,
            error,
            "turn_cancelled",
            "turn.cancelled",
            "queued turn cancelled",
            !command.cancel.has_value() || command.cancel->target_turn_id.empty()
                ? command_turn_id(cancelled_item->command)
                : command.cancel->target_turn_id);
    }
}

bool common_agent_daemon_dispatcher::populate_status_locked(
        common_agent_daemon_command_result & result,
        std::string & error) const {
    if (!service.populate_status(result, error)) {
        return false;
    }
    fill_status_snapshot_locked(result.status);
    result.event = "status";
    return true;
}

void common_agent_daemon_dispatcher::initialize_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result) const {
    auto target_request_id = std::move(result.target_request_id);
    auto target_turn_id = std::move(result.target_turn_id);
    result = {};
    result.request_id = command.request_id;
    result.response_kind = common_agent_daemon_response_kind::lifecycle;
    result.target_request_id = std::move(target_request_id);
    result.target_turn_id = std::move(target_turn_id);
}

void common_agent_daemon_dispatcher::initialize_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result) const {
    result = {};
    result.request_id = command.request_id;
    result.response_kind = common_agent_daemon_response_kind::turn;
}

bool common_agent_daemon_dispatcher::fail_lifecycle_result_locked(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        std::string daemon_event_type,
        std::string turn_id) const {
    initialize_lifecycle_result(command, result);
    result.ok = false;
    result.event = std::move(event);
    result.error = error;
    append_daemon_event(
        result,
        std::move(daemon_event_type),
        command.request_id,
        std::move(turn_id),
        error);
    finalize_lifecycle_result_locked(result);
    return false;
}

bool common_agent_daemon_dispatcher::succeed_lifecycle_result_locked(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        std::string daemon_event_type,
        std::string detail,
        std::string turn_id) const {
    initialize_lifecycle_result(command, result);
    result.ok = true;
    result.event = std::move(event);
    append_daemon_event(
        result,
        std::move(daemon_event_type),
        command.request_id,
        std::move(turn_id),
        std::move(detail));
    finalize_lifecycle_result_locked(result);
    error.clear();
    return true;
}

void common_agent_daemon_dispatcher::cancel_queued_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string error) const {
    initialize_turn_result(command, result);
    result.ok = false;
    result.event = "turn_cancelled";
    result.turn_result.cancelled = true;
    result.turn_result.error = error;
    result.error = std::move(error);
    append_daemon_event(
        result,
        "turn.cancelled",
        command.request_id,
        command_turn_id(command),
        result.turn_result.error);
}

void common_agent_daemon_dispatcher::finalize_lifecycle_result_locked(
        common_agent_daemon_command_result & result) const {
    fill_status_snapshot_locked(result.status);
}

void common_agent_daemon_dispatcher::fill_status_snapshot_locked(
        common_agent_daemon_status & status) const {
    const size_t queued_count = queue.size();
    if (const auto active_turn = service.describe_active_turn()) {
        assign_active_turn_status(status, *active_turn);
    } else {
        status.active_turn.reset();
        status.active_request_id.clear();
        status.active_turn_id.clear();
        status.active_turn_phase.clear();
        status.active_turn_disposition.clear();
        status.active_cancel_requested = false;
    }
    status.queued_command_count = queued_count;
    status.worker_running = worker_running;
    status.accepting_commands = accepting_commands;
    status.shutdown_requested = service.shutdown_requested();
    status.max_queue_size = max_queue_size;
    status.queue_capacity_remaining =
        max_queue_size > queued_count ? (max_queue_size - queued_count) : 0;
    status.state = service.state();
    status.session_count = status.sessions.size();
    status.live = status.state != common_agent_daemon_state::stopped && worker_running;
    status.ready = status.state == common_agent_daemon_state::ready &&
        accepting_commands &&
        worker_running;
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
                if (!item->command.turn->request.turn.execution_control.cancellation) {
                    item->command.turn->request.turn.execution_control =
                        make_common_agent_runtime_execution_control(
                            item->command.turn->request.turn.execution_control.timeout_policy);
                }
                item->command.turn->request.request_id = item->command.request_id;
            }
        }

        queued_result queued;
        queued.result.request_id = item->command.request_id;
        queued.result.events = item->events;
        queued.result.daemon_event_count = queued.result.events.size();
        append_daemon_event(
            queued.result,
            "command.started",
            item->command.request_id,
            command_turn_id(item->command));
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
            if (service.shutdown_requested()) {
                accepting_commands = false;
                stop_requested = true;
            }
            fill_status_snapshot_locked(queued.result.status);
        }

        item->promise.set_value(std::move(queued));
        condition.notify_all();
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        service.mark_stopping();
        worker_running = false;
        service.mark_stopped();
    }
}
