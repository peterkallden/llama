#include "agent-runtime-turn-driver.h"
#include "agent-inference-capacity-gate.h"

#include <chrono>
#include <memory>
#include <utility>

namespace {

common_agent_inference_priority inference_priority_for_request(
        const common_agent_runtime_session_host_turn_request & request) {
    if (request.mode == common_agent_runtime_host_mode::chat) {
        return common_agent_inference_priority::interactive;
    }
    if (request.deliberation_policy_override.has_value() &&
            request.deliberation_policy_override->mode == common_agent_thinking_mode::research) {
        return common_agent_inference_priority::background;
    }
    return common_agent_inference_priority::normal;
}

common_agent_daemon_event_type completed_event_for_operation_kind(
        common_agent_runtime_pending_operation_kind kind) {
    switch (kind) {
        case common_agent_runtime_pending_operation_kind::tool:
            return common_agent_daemon_event_type::tool_completed;
        case common_agent_runtime_pending_operation_kind::inference:
            return common_agent_daemon_event_type::inference_completed;
    }
    return common_agent_daemon_event_type::inference_completed;
}

} // namespace

common_agent_runtime_turn_disposition poll_common_agent_runtime_pending_operation(
        common_runtime_operation_manager & operation_manager,
        std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
        std::optional<common_agent_runtime_turn_execution> & active_turn,
        std::mutex & lane_mutex,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_session_host_turn_result & result,
        std::string & error,
        common_agent_runtime_turn_phase phase,
        const common_agent_event_emitter & turn_events) {
    std::string operation_id;
    common_agent_runtime_pending_operation_kind operation_kind =
        common_agent_runtime_pending_operation_kind::tool;
    std::string operation_detail;
    common_agent_runtime_turn_disposition waiting_disposition =
        common_agent_runtime_turn_disposition::failed;
    {
        std::lock_guard<std::mutex> lock(lane_mutex);
        if (pending_operation.has_value()) {
            operation_id = pending_operation->pending_operation.operation_id;
            operation_kind = pending_operation->pending_operation.kind;
            operation_detail = pending_operation->pending_operation.detail;
            waiting_disposition = pending_operation->waiting_disposition;
        }
    }

    if (request.execution_control.should_stop()) {
        if (!operation_id.empty()) {
            std::string ignored_error;
            operation_manager.cancel(operation_id, ignored_error);
            if (operation_kind ==
                    common_agent_runtime_pending_operation_kind::tool) {
                turn_events.with_operation(operation_id).emit(
                    common_agent_daemon_event_type::tool_cancelled,
                    request.execution_control.stop_reason());
            }
        }
        {
            std::lock_guard<std::mutex> lock(lane_mutex);
            pending_operation.reset();
            if (active_turn.has_value()) {
                active_turn->pending_operation.reset();
                active_turn->cancellation_requested = true;
                active_turn->disposition = common_agent_runtime_turn_disposition::cancelled;
                active_turn->phase = common_agent_runtime_turn_phase::cancelled;
            }
        }
        result = {};
        result.cancelled = true;
        result.error = request.execution_control.stop_reason();
        error = result.error;
        turn_events.emit(common_agent_daemon_event_type::turn_cancelled, result.error);
        return common_agent_runtime_turn_disposition::cancelled;
    }

    if (operation_id.empty()) {
        error = "lane phase is missing its pending operation";
        {
            std::lock_guard<std::mutex> lock(lane_mutex);
            if (active_turn.has_value()) {
                active_turn->pending_operation.reset();
                active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                active_turn->phase = common_agent_runtime_turn_phase::failed;
            }
        }
        turn_events.emit(common_agent_daemon_event_type::turn_failed, error);
        return common_agent_runtime_turn_disposition::failed;
    }

    bool ready = false;
    if (!operation_manager.poll(operation_id, ready, error)) {
        common_runtime_operation_status operation_status;
        const bool has_status = operation_manager.describe(operation_id, operation_status);
        {
            std::lock_guard<std::mutex> lock(lane_mutex);
            pending_operation.reset();
            if (active_turn.has_value()) {
                active_turn->pending_operation.reset();
                active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                active_turn->phase = common_agent_runtime_turn_phase::failed;
            }
        }
        if (operation_kind ==
                common_agent_runtime_pending_operation_kind::tool) {
            turn_events.with_operation(operation_id).emit(
                has_status && operation_status.state == common_runtime_operation_state::timed_out
                    ? common_agent_daemon_event_type::tool_timed_out
                    : common_agent_daemon_event_type::tool_failed,
                error);
        }
        turn_events.emit(common_agent_daemon_event_type::turn_failed, error);
        return common_agent_runtime_turn_disposition::failed;
    }
    if (!ready) {
        {
            std::lock_guard<std::mutex> lock(lane_mutex);
            if (active_turn.has_value()) {
                active_turn->disposition = waiting_disposition;
            }
        }
        return waiting_disposition;
    }

    const bool inference_capacity_granted =
        operation_kind == common_agent_runtime_pending_operation_kind::inference &&
        operation_detail == "waiting for inference capacity";
    {
        std::lock_guard<std::mutex> lock(lane_mutex);
        if (active_turn.has_value()) {
            active_turn->pending_operation.reset();
            active_turn->disposition = common_agent_runtime_turn_disposition::continue_immediately;
            active_turn->phase = phase;
        }
        pending_operation.reset();
    }
    turn_events.emit(
        inference_capacity_granted
            ? common_agent_daemon_event_type::inference_capacity_granted
            : completed_event_for_operation_kind(operation_kind),
        inference_capacity_granted
            ? "inference admission granted"
            : (operation_kind == common_agent_runtime_pending_operation_kind::tool
                ? "manager-owned pending tool operation completed"
                : "manager-owned pending inference operation completed"));
    if (operation_kind == common_agent_runtime_pending_operation_kind::tool) {
        turn_events.emit(
            common_agent_daemon_event_type::turn_resumed,
            "turn resumed after pending tool operation");
    }
    return common_agent_runtime_turn_disposition::continue_immediately;
}

common_agent_runtime_turn_disposition advance_common_agent_runtime_turn(
        common_agent_runtime_session_host * host,
        std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
        std::optional<common_agent_runtime_turn_execution> & active_turn,
        std::mutex & lane_mutex,
        common_runtime_operation_manager & operation_manager,
        const std::function<bool(
            const common_agent_runtime_session_host_turn_request & request,
            std::optional<common_agent_runtime_session_manager_pending_operation> & pending_operation,
            std::string & error)> & pending_operation_resolver,
        const std::shared_ptr<common_agent_inference_capacity_gate> & inference_gate,
        const std::shared_ptr<common_agent_runtime_inference_executor> & inference_executor,
        const std::string & request_id,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_session_host_turn_result & result,
        std::string & error,
        const common_agent_event_emitter & turn_events) {
    common_agent_runtime_turn_phase phase;
    {
        std::lock_guard<std::mutex> lock(lane_mutex);
        if (!active_turn.has_value()) {
            error = "lane does not have an active turn";
            return common_agent_runtime_turn_disposition::failed;
        }
        phase = active_turn->phase;
    }

    switch (phase) {
        case common_agent_runtime_turn_phase::queued:
            if (request.execution_control.should_stop()) {
                {
                    std::lock_guard<std::mutex> lock(lane_mutex);
                    if (active_turn.has_value()) {
                        active_turn->cancellation_requested = true;
                        active_turn->disposition = common_agent_runtime_turn_disposition::cancelled;
                        active_turn->phase = common_agent_runtime_turn_phase::cancelled;
                    }
                }
                result = {};
                result.cancelled = true;
                result.error = request.execution_control.stop_reason();
                error = result.error;
                turn_events.emit(common_agent_daemon_event_type::turn_cancelled, result.error);
                return common_agent_runtime_turn_disposition::cancelled;
            }
            {
                std::lock_guard<std::mutex> lock(lane_mutex);
                if (active_turn.has_value()) {
                    active_turn->phase = common_agent_runtime_turn_phase::preparing;
                }
            }
            return common_agent_runtime_turn_disposition::continue_immediately;

        case common_agent_runtime_turn_phase::preparing:
            if (pending_operation_resolver) {
                std::optional<common_agent_runtime_session_manager_pending_operation> resolved;
                if (!pending_operation_resolver(request, resolved, error)) {
                    {
                        std::lock_guard<std::mutex> lock(lane_mutex);
                        pending_operation.reset();
                        if (active_turn.has_value()) {
                            active_turn->pending_operation.reset();
                            active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                            active_turn->phase = common_agent_runtime_turn_phase::failed;
                        }
                    }
                    turn_events.emit(common_agent_daemon_event_type::turn_failed, error);
                    return common_agent_runtime_turn_disposition::failed;
                }
                if (resolved.has_value()) {
                    std::string operation_error;
                    if (!operation_manager.begin(
                            resolved->pending_operation,
                            resolved->take_poll_callback(),
                            resolved->take_cancel_callback(),
                            operation_error)) {
                        error = operation_error;
                        {
                            std::lock_guard<std::mutex> lock(lane_mutex);
                            pending_operation.reset();
                            if (active_turn.has_value()) {
                                active_turn->pending_operation.reset();
                                active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                                active_turn->phase = common_agent_runtime_turn_phase::failed;
                            }
                        }
                        turn_events.emit(common_agent_daemon_event_type::turn_failed, error);
                        return common_agent_runtime_turn_disposition::failed;
                    }
                    {
                        std::lock_guard<std::mutex> lock(lane_mutex);
                        pending_operation = std::move(resolved);
                        if (active_turn.has_value()) {
                            active_turn->phase = pending_operation->waiting_phase;
                            active_turn->disposition = pending_operation->waiting_disposition;
                            active_turn->pending_operation = pending_operation->pending_operation;
                        }
                    }
                    auto operation_events = turn_events.with_operation(
                        pending_operation->pending_operation.operation_id);
                    if (pending_operation->pending_operation.kind ==
                            common_agent_runtime_pending_operation_kind::tool) {
                        operation_events.emit(
                            common_agent_daemon_event_type::tool_queued,
                            pending_operation->pending_operation.detail.empty()
                                ? "tool operation queued"
                                : pending_operation->pending_operation.detail);
                    }
                    operation_events.emit(
                        pending_operation->pending_operation.kind ==
                            common_agent_runtime_pending_operation_kind::tool
                            ? common_agent_daemon_event_type::turn_waiting_for_tool
                            : common_agent_daemon_event_type::turn_waiting_for_inference,
                        pending_operation->pending_operation.detail.empty()
                            ? "lane entered manager-owned pending operation"
                            : pending_operation->pending_operation.detail);
                    return pending_operation->waiting_disposition;
                }
            }
            {
                std::lock_guard<std::mutex> lock(lane_mutex);
                if (active_turn.has_value()) {
                    active_turn->phase = common_agent_runtime_turn_phase::awaiting_inference;
                    common_agent_runtime_pending_operation pending;
                    pending.operation_id = "inference:" + request_id;
                    pending.kind = common_agent_runtime_pending_operation_kind::inference;
                    pending.detail = "session host turn execution";
                    if (request.execution_control.deadline.has_value()) {
                        pending.deadline = *request.execution_control.deadline;
                    }
                    active_turn->pending_operation = std::move(pending);
                }
            }
            turn_events
                .with_operation("inference:" + request_id)
                .emit(
                    common_agent_daemon_event_type::turn_waiting_for_inference,
                    "turn entered inference execution");
            return common_agent_runtime_turn_disposition::continue_immediately;

        case common_agent_runtime_turn_phase::awaiting_inference: {
            const bool waiting_for_inference_capacity = pending_operation.has_value() &&
                pending_operation->pending_operation.detail == "waiting for inference capacity";
            if (!pending_operation.has_value()) {
                if (inference_gate) {
                    bool acquired = false;
                    {
                        std::lock_guard<std::mutex> lock(lane_mutex);
                        acquired = active_turn.has_value() && active_turn->inference_capacity_acquired;
                    }
                    if (!acquired) {
                        const std::string waiter_id = "inference-capacity:" + request_id;
                        std::string admission_error;
                        if (!inference_gate->enqueue(
                                common_agent_inference_wait_request{
                                    waiter_id,
                                    inference_priority_for_request(request),
                                    request.execution_control.deadline.value_or(
                                        std::chrono::steady_clock::time_point{}),
                                },
                                admission_error)) {
                            {
                                std::lock_guard<std::mutex> lock(lane_mutex);
                                if (active_turn.has_value()) {
                                    active_turn->inference_capacity_acquired = false;
                                    active_turn->pending_operation.reset();
                                    active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                                    active_turn->phase = common_agent_runtime_turn_phase::failed;
                                }
                            }
                            error = admission_error;
                            return common_agent_runtime_turn_disposition::failed;
                        }
                        if (!inference_gate->try_acquire(waiter_id)) {
                        common_agent_runtime_session_manager_pending_operation pending;
                        pending.pending_operation.operation_id = waiter_id;
                        pending.pending_operation.kind = common_agent_runtime_pending_operation_kind::inference;
                        pending.pending_operation.detail = "waiting for inference capacity";
                        if (request.execution_control.deadline.has_value()) {
                            pending.pending_operation.deadline = *request.execution_control.deadline;
                        }
                        pending.waiting_phase = common_agent_runtime_turn_phase::awaiting_inference;
                        pending.waiting_disposition = common_agent_runtime_turn_disposition::wait_for_inference;
                        pending.poll = [inference_gate, waiter_id](bool & ready, std::string & poll_error) {
                            ready = inference_gate->try_acquire(waiter_id);
                            poll_error.clear();
                            return true;
                        };
                        pending.cancel = [inference_gate, waiter_id](std::string & cancel_error) {
                            return inference_gate->cancel(waiter_id, cancel_error);
                        };
                        std::string operation_error;
                        if (!operation_manager.begin(
                                pending.pending_operation,
                                pending.take_poll_callback(),
                                pending.take_cancel_callback(),
                                operation_error)) {
                            std::string ignored_error;
                            inference_gate->cancel(waiter_id, ignored_error);
                            {
                                std::lock_guard<std::mutex> lock(lane_mutex);
                                if (active_turn.has_value()) {
                                    active_turn->inference_capacity_acquired = false;
                                    active_turn->pending_operation.reset();
                                    active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                                    active_turn->phase = common_agent_runtime_turn_phase::failed;
                                }
                            }
                            error = operation_error;
                            return common_agent_runtime_turn_disposition::failed;
                        }
                        {
                            std::lock_guard<std::mutex> lock(lane_mutex);
                            pending_operation = std::move(pending);
                            if (active_turn.has_value()) {
                                active_turn->pending_operation = pending_operation->pending_operation;
                                active_turn->disposition = common_agent_runtime_turn_disposition::wait_for_inference;
                            }
                        }
                        turn_events.with_operation(waiter_id).emit(
                            common_agent_daemon_event_type::inference_queued,
                            "inference queued for admission");
                        turn_events.with_operation(waiter_id).emit(
                            common_agent_daemon_event_type::turn_waiting_for_inference,
                            "turn suspended while waiting for inference capacity");
                        return common_agent_runtime_turn_disposition::wait_for_inference;
                        }
                    }
                    {
                        std::lock_guard<std::mutex> lock(lane_mutex);
                        if (active_turn.has_value()) {
                            active_turn->inference_capacity_acquired = true;
                        }
                    }
                }
                const std::string lease_id = "inference-capacity:" + request_id;
                auto * result_ptr = &result;
                auto * error_ptr = &error;
                const auto executor = inference_executor
                    ? inference_executor
                    : make_common_agent_runtime_async_inference_executor();
                std::string executor_error;
                const auto inference_task = executor->submit(
                    host, request, inference_gate, lease_id, executor_error);
                if (!inference_task) {
                    if (inference_gate) inference_gate->release(lease_id);
                    {
                        std::lock_guard<std::mutex> lock(lane_mutex);
                        if (active_turn.has_value()) {
                            active_turn->inference_capacity_acquired = false;
                            active_turn->pending_operation.reset();
                            active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                            active_turn->phase = common_agent_runtime_turn_phase::failed;
                        }
                    }
                    error = executor_error;
                    return common_agent_runtime_turn_disposition::failed;
                }

                common_agent_runtime_session_manager_pending_operation pending;
                pending.pending_operation.operation_id = "inference:" + request_id;
                pending.pending_operation.kind = common_agent_runtime_pending_operation_kind::inference;
                pending.pending_operation.detail = "session host turn execution";
                if (request.execution_control.deadline.has_value()) {
                    pending.pending_operation.deadline = *request.execution_control.deadline;
                }
                pending.waiting_phase = common_agent_runtime_turn_phase::awaiting_inference;
                pending.waiting_disposition = common_agent_runtime_turn_disposition::wait_for_inference;
                pending.poll = [inference_task, result_ptr, error_ptr](bool & ready, std::string & poll_error) {
                    const bool ok = inference_task->poll(ready, *result_ptr, *error_ptr);
                    poll_error = *error_ptr;
                    return ok;
                };
                pending.cancel = [inference_task](std::string & cancel_error) {
                    return inference_task->cancel(cancel_error);
                };
                std::string operation_error;
                if (!operation_manager.begin(
                        pending.pending_operation,
                        pending.take_poll_callback(),
                        pending.take_cancel_callback(),
                        operation_error)) {
                    std::string ignored_cancel_error;
                    inference_task->cancel(ignored_cancel_error);
                    std::lock_guard<std::mutex> lock(lane_mutex);
                    if (active_turn.has_value()) {
                        active_turn->inference_capacity_acquired = false;
                        active_turn->pending_operation.reset();
                        active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                        active_turn->phase = common_agent_runtime_turn_phase::failed;
                    }
                    error = operation_error;
                    return common_agent_runtime_turn_disposition::failed;
                }
                {
                    std::lock_guard<std::mutex> lock(lane_mutex);
                    pending_operation = std::move(pending);
                    if (active_turn.has_value()) {
                        active_turn->pending_operation = pending_operation->pending_operation;
                        active_turn->disposition = common_agent_runtime_turn_disposition::wait_for_inference;
                    }
                }
                turn_events
                    .with_operation("inference:" + request_id)
                    .emit(
                        common_agent_daemon_event_type::inference_started,
                        "session host turn execution started");
                turn_events
                    .with_operation("inference:" + request_id)
                    .emit(
                        common_agent_daemon_event_type::turn_waiting_for_inference,
                        "turn suspended while session host turn executes");
                return common_agent_runtime_turn_disposition::wait_for_inference;
            }

            const bool host_turn_async = pending_operation->pending_operation.detail ==
                "session host turn execution";
            const auto pending_disposition = poll_common_agent_runtime_pending_operation(
                operation_manager,
                pending_operation,
                active_turn,
                lane_mutex,
                request,
                result,
                error,
                common_agent_runtime_turn_phase::awaiting_inference,
                turn_events);
            if (pending_disposition != common_agent_runtime_turn_disposition::continue_immediately) {
                return pending_disposition;
            }
            if (waiting_for_inference_capacity) {
                {
                    std::lock_guard<std::mutex> lock(lane_mutex);
                    if (active_turn.has_value()) active_turn->inference_capacity_acquired = true;
                }
                return advance_common_agent_runtime_turn(
                    host, pending_operation, active_turn, lane_mutex, operation_manager,
                    pending_operation_resolver, inference_gate, inference_executor,
                    request_id, request,
                    result, error, turn_events);
            }
            // Cancellation may arrive after the pending operation reports
            // ready but before the host turn is entered. Keep that boundary
            // explicit so a cancelled operation cannot launch a host turn.
            if (request.execution_control.should_stop()) {
                result = {};
                result.cancelled = true;
                result.error = request.execution_control.stop_reason();
                error = result.error;
                {
                    std::lock_guard<std::mutex> lock(lane_mutex);
                    if (active_turn.has_value()) {
                        active_turn->cancellation_requested = true;
                        active_turn->disposition = common_agent_runtime_turn_disposition::cancelled;
                        active_turn->phase = common_agent_runtime_turn_phase::cancelled;
                        active_turn->pending_operation.reset();
                    }
                }
                turn_events.emit(common_agent_daemon_event_type::turn_cancelled, result.error);
                return common_agent_runtime_turn_disposition::cancelled;
            }
            const bool ok = host_turn_async
                ? result.ok
                : host->run_turn(request, result, error);
            if (host_turn_async && inference_gate) {
                std::lock_guard<std::mutex> lock(lane_mutex);
                if (active_turn.has_value()) active_turn->inference_capacity_acquired = false;
            }
            const auto disposition = ok
                ? common_agent_runtime_turn_disposition::completed
                : (result.cancelled
                    ? common_agent_runtime_turn_disposition::cancelled
                    : common_agent_runtime_turn_disposition::failed);
            {
                std::lock_guard<std::mutex> lock(lane_mutex);
                if (active_turn.has_value()) {
                    active_turn->cancellation_requested = request.execution_control.is_cancel_requested();
                    active_turn->disposition = disposition;
                    active_turn->pending_operation.reset();
                    active_turn->phase = ok
                        ? common_agent_runtime_turn_phase::completing
                        : (result.cancelled
                            ? common_agent_runtime_turn_phase::cancelled
                            : common_agent_runtime_turn_phase::failed);
                }
            }
            turn_events.emit(
                ok
                    ? common_agent_daemon_event_type::turn_completed
                    : (result.cancelled
                        ? common_agent_daemon_event_type::turn_cancelled
                        : common_agent_daemon_event_type::turn_failed),
                ok ? "turn execution returned" : error);
            return disposition;
        }

        case common_agent_runtime_turn_phase::awaiting_tool:
            return poll_common_agent_runtime_pending_operation(
                operation_manager,
                pending_operation,
                active_turn,
                lane_mutex,
                request,
                result,
                error,
                common_agent_runtime_turn_phase::awaiting_inference,
                turn_events);

        case common_agent_runtime_turn_phase::completing:
        case common_agent_runtime_turn_phase::completed:
            if (active_turn.has_value()) {
                active_turn->disposition = common_agent_runtime_turn_disposition::completed;
            }
            return common_agent_runtime_turn_disposition::completed;

        case common_agent_runtime_turn_phase::failed:
            if (active_turn.has_value()) {
                active_turn->disposition = common_agent_runtime_turn_disposition::failed;
            }
            return common_agent_runtime_turn_disposition::failed;

        case common_agent_runtime_turn_phase::cancelled:
            if (active_turn.has_value()) {
                active_turn->disposition = common_agent_runtime_turn_disposition::cancelled;
            }
            return common_agent_runtime_turn_disposition::cancelled;
    }

    error = "unsupported lane turn phase";
    return common_agent_runtime_turn_disposition::failed;
}
