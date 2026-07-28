#include "agent-daemon-dispatcher.h"

#include <iterator>
#include <utility>

namespace {

common_agent_event_context make_dispatcher_command_event_context(
        const common_agent_daemon_command & command) {
    common_agent_event_context context;
    context.request_id = command.request_id;
    if (command.turn.has_value()) {
        context.namespace_id = command.turn->request.turn.namespace_id;
        context.project_id = command.turn->request.turn.project_id;
        context.session_id = command.turn->request.turn.session_id;
        context.turn_id = command.turn->request.turn.turn_id;
        return context;
    }
    if (command.session.has_value()) {
        context.namespace_id = command.session->key.namespace_id;
        context.session_id = command.session->key.session_id;
        return context;
    }
    return context;
}

void assign_active_turn_status(
        common_agent_daemon_status & status,
        const common_agent_runtime_active_turn_descriptor & active_turn) {
    status.active_turn = common_agent_daemon_active_turn_status{
        active_turn.request_id,
        active_turn.turn_id,
        active_turn.phase,
        active_turn.disposition,
        active_turn.cancellation_requested,
        active_turn.pending_operation_kind,
        active_turn.pending_operation_detail,
    };
    status.active_request_id = active_turn.request_id;
    status.active_turn_id = active_turn.turn_id;
    status.active_turn_phase = active_turn.phase;
    status.active_turn_disposition = active_turn.disposition;
    status.active_cancel_requested = active_turn.cancellation_requested;
    status.active_pending_operation_kind = active_turn.pending_operation_kind;
    status.active_pending_operation_detail = active_turn.pending_operation_detail;
}

void append_daemon_event(
        common_agent_daemon_command_result & result,
        common_agent_daemon_event event) {
    result.events.push_back(std::move(event));
    result.daemon_event_count = result.events.size();
}

void emit_result_event(
        common_agent_daemon_command_result & result,
        const common_agent_daemon_command & command,
        common_agent_daemon_event_type type,
        std::string detail = {}) {
    common_agent_event_emitter emitter(
        [&result](common_agent_daemon_event event) {
            append_daemon_event(result, std::move(event));
        },
        make_dispatcher_command_event_context(command));
    emitter.emit(type, std::move(detail));
}

std::string command_turn_id(const common_agent_daemon_command & command) {
    if (!command.turn.has_value()) {
        return {};
    }
    return command.turn->request.turn.turn_id;
}

bool is_session_lifecycle_command(const common_agent_daemon_command & command) {
    return command.type == common_agent_daemon_command_type::reset_session ||
        command.type == common_agent_daemon_command_type::close_session;
}

bool queued_turn_matches_session(
        const common_agent_daemon_command & queued_command,
        const common_agent_runtime_session_key & key) {
    return queued_command.type == common_agent_daemon_command_type::run_turn &&
        queued_command.turn.has_value() &&
        queued_command.turn->request.turn.namespace_id == key.namespace_id &&
        queued_command.turn->request.turn.session_id == key.session_id;
}

const char * queued_turn_rejection_error(
        common_agent_daemon_command_type type) {
    switch (type) {
        case common_agent_daemon_command_type::reset_session:
            return "session reset before queued turn reached session lane";
        case common_agent_daemon_command_type::close_session:
            return "session closed before queued turn reached session lane";
        default:
            return "session lifecycle rejected queued turn";
    }
}

const char * queued_turn_rejection_event(
        common_agent_daemon_command_type type) {
    switch (type) {
        case common_agent_daemon_command_type::reset_session:
            return "turn_rejected";
        case common_agent_daemon_command_type::close_session:
            return "turn_rejected";
        default:
            return "turn_rejected";
    }
}

common_agent_daemon_event_type queued_turn_rejection_daemon_event_type(
        common_agent_daemon_command_type) {
    return common_agent_daemon_event_type::turn_rejected;
}

} // namespace

common_agent_daemon_dispatcher::common_agent_daemon_dispatcher(
        common_agent_daemon_runtime runtime,
        size_t max_queue_size,
        size_t worker_count)
    : service(std::move(runtime))
    , max_queue_size(max_queue_size)
    , worker_count(worker_count == 0 ? 1 : worker_count) {
    workers.reserve(this->worker_count);
    for (size_t i = 0; i < this->worker_count; ++i) {
        workers.emplace_back([this]() {
            worker_loop();
        });
    }
    workers_running = this->worker_count;
    worker_running = true;
}

common_agent_daemon_dispatcher::~common_agent_daemon_dispatcher() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        accepting_commands = false;
        stop_requested = true;
        service.mark_stopping();
    }
    condition.notify_all();
    for (auto & worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        worker_running = false;
        workers_running = 0;
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
    if (is_session_lifecycle_command(command)) {
        return execute_session_lifecycle(command, result, error);
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
                common_agent_daemon_event_type::command_rejected,
                command_turn_id(command));
        }
        if (queue.size() >= max_queue_size) {
            error = "daemon command queue is full";
            return fail_lifecycle_result_locked(
                command,
                result,
                error,
                "command_rejected",
                common_agent_daemon_event_type::command_rejected,
                command_turn_id(command));
        }
        common_agent_event_emitter(
            [&item](common_agent_daemon_event event) {
                item->events.push_back(std::move(event));
            },
            make_dispatcher_command_event_context(command))
            .emit(common_agent_daemon_event_type::command_queued);
        queue.push_back(item);
        ++commands_accepted;
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

std::string common_agent_daemon_dispatcher::subscribe_events(
        common_agent_event_stream_subscription subscription) {
    return service.subscribe_events(std::move(subscription));
}

void common_agent_daemon_dispatcher::unsubscribe_events(
        const std::string & subscription_id) {
    service.unsubscribe_events(subscription_id);
}

common_agent_event_stream_wait_status common_agent_daemon_dispatcher::wait_for_event(
        const std::string & subscription_id,
        common_agent_event_stream_delivery & delivery,
        std::chrono::milliseconds timeout) {
    return service.wait_for_event(subscription_id, delivery, timeout);
}

bool common_agent_daemon_dispatcher::build_http_tool_catalog(
        const daemon_options & options,
        agent_mcp_server_tool_registry & registry,
        std::string & error) const {
    return service.build_http_tool_catalog(options, registry, error);
}

size_t common_agent_daemon_dispatcher::queued_command_count() const {
    std::lock_guard<std::mutex> lock(mutex);
    return queue.size();
}

bool common_agent_daemon_dispatcher::execute_session_lifecycle(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error) {
    auto item = std::make_shared<queued_command>();
    item->command = command;
    auto future = item->promise.get_future();

    std::vector<std::shared_ptr<queued_command>> rejected_items;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!accepting_commands) {
            error = "daemon dispatcher is not accepting new commands";
            return fail_lifecycle_result_locked(
                command,
                result,
                error,
                "command_rejected",
                common_agent_daemon_event_type::command_rejected,
                command_turn_id(command));
        }
        if (queue.size() >= max_queue_size) {
            error = "daemon command queue is full";
            return fail_lifecycle_result_locked(
                command,
                result,
                error,
                "command_rejected",
                common_agent_daemon_event_type::command_rejected,
                command_turn_id(command));
        }
        if (command.session.has_value()) {
            auto it = queue.begin();
            while (it != queue.end()) {
                if (queued_turn_matches_session((*it)->command, command.session->key)) {
                    rejected_items.push_back(*it);
                    it = queue.erase(it);
                    continue;
                }
                ++it;
            }
        }
        common_agent_event_emitter(
            [&item](common_agent_daemon_event event) {
                item->events.push_back(std::move(event));
            },
            make_dispatcher_command_event_context(command))
            .emit(common_agent_daemon_event_type::command_queued);
        queue.push_back(item);
    }

    for (const auto & rejected_item : rejected_items) {
        queued_result rejected;
        rejected.ok = false;
        reject_queued_turn_result(
            rejected_item->command,
            rejected.result,
            queued_turn_rejection_event(command.type),
            queued_turn_rejection_daemon_event_type(command.type),
            queued_turn_rejection_error(command.type));
        {
            std::lock_guard<std::mutex> lock(mutex);
            fill_status_snapshot_locked(rejected.result.status);
        }
        rejected.error = rejected.result.error;
        rejected_item->promise.set_value(std::move(rejected));
    }

    condition.notify_one();

    auto queued = future.get();
    result = std::move(queued.result);
    error = std::move(queued.error);
    return queued.ok;
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
                    common_agent_daemon_event_type::turn_cancel_requested,
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
            common_agent_daemon_event_type::turn_cancel_rejected,
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
            common_agent_daemon_event_type::turn_cancelled,
            "queued turn cancelled",
            !command.cancel.has_value() || command.cancel->target_turn_id.empty()
                ? command_turn_id(cancelled_item->command)
                : command.cancel->target_turn_id);
    }
}

bool common_agent_daemon_dispatcher::populate_status_locked(
        common_agent_daemon_command_result & result,
        std::string & error) const {
    common_agent_daemon_command_execution execution;
    execution.outcome.request_id = result.request_id;
    if (!service.populate_status_outcome(execution.outcome, execution.events, error)) {
        result = project_agent_daemon_command_execution(std::move(execution));
        return false;
    }
    result = project_agent_daemon_command_execution(std::move(execution));
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
        common_agent_daemon_event_type event_type,
        std::string turn_id) const {
    initialize_lifecycle_result(command, result);
    result.ok = false;
    result.event = std::move(event);
    result.error = error;
    if (event_type != common_agent_daemon_event_type::unknown) {
        auto context = make_dispatcher_command_event_context(command);
        context.turn_id = std::move(turn_id);
        common_agent_event_emitter(
            [&result](common_agent_daemon_event event_value) {
                append_daemon_event(result, std::move(event_value));
            },
            std::move(context))
            .emit(event_type, error);
    }
    finalize_lifecycle_result_locked(result);
    return false;
}

bool common_agent_daemon_dispatcher::succeed_lifecycle_result_locked(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        common_agent_daemon_event_type event_type,
        std::string detail,
        std::string turn_id) const {
    initialize_lifecycle_result(command, result);
    result.ok = true;
    result.event = std::move(event);
    if (event_type != common_agent_daemon_event_type::unknown) {
        auto context = make_dispatcher_command_event_context(command);
        context.turn_id = std::move(turn_id);
        common_agent_event_emitter(
            [&result](common_agent_daemon_event event_value) {
                append_daemon_event(result, std::move(event_value));
            },
            std::move(context))
            .emit(event_type, std::move(detail));
    }
    finalize_lifecycle_result_locked(result);
    error.clear();
    return true;
}

void common_agent_daemon_dispatcher::cancel_queued_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string error) const {
    reject_queued_turn_result(
        command,
        result,
        "turn_cancelled",
        common_agent_daemon_event_type::turn_cancelled,
        std::move(error));
    result.turn_result.cancelled = true;
    result.turn_result.response_generation_status = common_agent_generation_status::cancelled;
    result.turn_result.response_stop_reason = common_agent_generation_stop_reason::cancelled;
}

void common_agent_daemon_dispatcher::reject_queued_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string event,
        common_agent_daemon_event_type event_type,
        std::string error) const {
    initialize_turn_result(command, result);
    result.ok = false;
    result.event = std::move(event);
    result.turn_result.cancelled = false;
    result.turn_result.failure_class = common_agent_failure_class::execution;
    result.turn_result.response_generation_status = common_agent_generation_status::errored;
    result.turn_result.response_stop_reason = common_agent_generation_stop_reason::error;
    result.turn_result.error = error;
    result.error = std::move(error);
    if (event_type != common_agent_daemon_event_type::unknown) {
        emit_result_event(
            result,
            command,
            event_type,
            result.turn_result.error);
    }
}

void common_agent_daemon_dispatcher::finalize_lifecycle_result_locked(
        common_agent_daemon_command_result & result) const {
    fill_status_snapshot_locked(result.status);
}

void common_agent_daemon_dispatcher::fill_status_snapshot_locked(
        common_agent_daemon_status & status) const {
    const size_t queued_count = queue.size();
    if (!status.session_snapshot_populated) {
        status.sessions = service.list_sessions();
        status.session_count = status.sessions.size();
    }
    if (const auto active_turn = service.describe_active_turn()) {
        assign_active_turn_status(status, *active_turn);
    } else {
        status.active_turn.reset();
        status.active_request_id.clear();
        status.active_turn_id.clear();
        status.active_turn_phase.clear();
        status.active_turn_disposition.clear();
        status.active_cancel_requested = false;
        status.active_pending_operation_kind.clear();
        status.active_pending_operation_detail.clear();
    }
    status.queued_command_count = queued_count;
    status.worker_running = worker_running;
    status.worker_count = worker_count;
    status.workers_running = workers_running;
    status.accepting_commands = accepting_commands;
    status.shutdown_requested = service.shutdown_requested();
    status.max_queue_size = max_queue_size;
    status.queue_capacity_remaining =
        max_queue_size > queued_count ? (max_queue_size - queued_count) : 0;
    status.commands_accepted = commands_accepted;
    status.commands_completed = commands_completed;
    status.commands_failed = commands_failed;
    status.turns_completed = turns_completed;
    status.tools_completed = tools_completed;
    status.state = service.state();
    status.readiness = service.readiness();
    status.live = status.state != common_agent_daemon_state::stopped && worker_running;
    status.ready = status.state == common_agent_daemon_state::ready &&
        accepting_commands &&
        worker_running &&
        status.readiness.health != "failed";
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
        common_agent_daemon_command_execution execution;
        execution.outcome.request_id = item->command.request_id;
        execution.events = item->events;
        common_agent_event_emitter(
            [&execution](common_agent_daemon_event event) {
                execution.events.push_back(std::move(event));
            },
            make_dispatcher_command_event_context(item->command))
            .emit(common_agent_daemon_event_type::command_started);
        if (item->command.type == common_agent_daemon_command_type::get_status) {
            std::lock_guard<std::mutex> lock(mutex);
            queued.ok = service.populate_status_outcome(execution.outcome, execution.events, queued.error);
        } else {
            queued.ok = service.execute_outcome(item->command, execution.outcome, execution.events, queued.error);
        }
        auto internal_events = service.take_internal_events();
        if (!internal_events.empty()) {
            execution.events.insert(
                execution.events.end(),
                std::make_move_iterator(internal_events.begin()),
                std::make_move_iterator(internal_events.end()));
        }
        queued.result = project_agent_daemon_command_execution(common_agent_daemon_command_execution{
            std::move(execution.outcome),
            std::move(execution.events),
        });
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++commands_completed;
            if (!queued.ok) ++commands_failed;
            if (item->command.type == common_agent_daemon_command_type::run_turn && queued.ok) ++turns_completed;
            if (item->command.type == common_agent_daemon_command_type::execute_tool && queued.ok) ++tools_completed;
        }
        if (!queued.error.empty() && queued.result.error.empty()) {
            queued.result.error = queued.error;
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            if (service.shutdown_requested()) {
                accepting_commands = false;
                if (queue.empty()) {
                    stop_requested = true;
                }
            }
            fill_status_snapshot_locked(queued.result.status);
        }

        item->promise.set_value(std::move(queued));
        condition.notify_all();
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (workers_running > 0) {
            --workers_running;
        }
        worker_running = workers_running > 0;
        if (workers_running == 0) {
            service.mark_stopping();
            service.mark_stopped();
        }
    }
}
