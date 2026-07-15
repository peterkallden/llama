#include "agent-runtime-session-manager.h"

#include <chrono>
#include <thread>
#include <utility>

common_agent_runtime_session_manager::common_agent_runtime_session_manager(
        common_agent_runtime_session_manager_config config)
    : config(std::move(config)) {}

void common_agent_runtime_session_manager::set_event_sink(
        common_agent_daemon_event_sink sink) {
    event_sink = std::move(sink);
}

common_agent_runtime_session_key common_agent_runtime_session_manager::make_session_key(
        const common_agent_runtime_session_manager_turn_request & request) const {
    return {
        request.turn.namespace_id,
        request.turn.session_id,
    };
}

common_agent_runtime_session_manager::common_agent_runtime_session_lane &
common_agent_runtime_session_manager::ensure_session_lane(
        const common_agent_runtime_session_key & key) {
    auto it = lanes.find(key);
    if (it == lanes.end()) {
        auto inserted = lanes.try_emplace(key);
        it = inserted.first;
        it->second.host = std::make_unique<common_agent_runtime_session_host>(config.host_config);
    }
    return it->second;
}

void common_agent_runtime_session_manager::emit_event(
        common_agent_daemon_event_type type,
        const std::string & request_id,
        const std::string & turn_id,
        const std::string & detail) const {
    if (event_sink) {
        event_sink(type, request_id, turn_id, detail);
    }
}

std::shared_ptr<common_agent_runtime_session_manager::common_agent_runtime_session_lane_message>
common_agent_runtime_session_manager::enqueue_lane_turn(
        common_agent_runtime_session_lane & lane,
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result &,
        std::string & error) {
    auto message = std::make_shared<common_agent_runtime_session_lane_message>();
    message->id = lane.next_message_id++;
    message->request = request;
    std::lock_guard<std::mutex> lock(lane.mutex);
    if (lane.state == common_agent_runtime_session_lane_state::resetting) {
        error = "session lane is resetting";
        return nullptr;
    }
    if (lane.state == common_agent_runtime_session_lane_state::closing) {
        error = "session lane is closing";
        return nullptr;
    }
    lane.mailbox.push_back(message);
    if (lane.state == common_agent_runtime_session_lane_state::running ||
            lane.state == common_agent_runtime_session_lane_state::running_with_waiters) {
        lane.state = common_agent_runtime_session_lane_state::running_with_waiters;
    }
    emit_event(
        common_agent_daemon_event_type::turn_accepted,
        request.request_id,
        request.turn.turn_id,
        "turn accepted into session lane mailbox");
    return message;
}

bool common_agent_runtime_session_manager::wait_for_message_completion(
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message,
        std::string & error) const {
    if (message == nullptr) {
        error = "lane message is missing";
        return false;
    }

    std::unique_lock<std::mutex> lock(message->mutex);
    message->condition.wait(lock, [&]() {
        return message->completed;
    });
    error = message->error;
    return message->ok;
}

void common_agent_runtime_session_manager::complete_lane_message(
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message,
        bool ok,
        const std::string & error,
        bool cancelled) const {
    if (message == nullptr) {
        return;
    }

    message->result.error = error;
    message->result.cancelled = cancelled;
    message->error = error;

    {
        std::lock_guard<std::mutex> lock(message->mutex);
        message->ok = ok;
        message->completed = true;
    }
    message->condition.notify_all();
}

void common_agent_runtime_session_manager::reconcile_lane_state(
        common_agent_runtime_session_lane & lane) const {
    std::lock_guard<std::mutex> lock(lane.mutex);
    if (lane.state == common_agent_runtime_session_lane_state::resetting ||
            lane.state == common_agent_runtime_session_lane_state::closing) {
        return;
    }
    if (lane.current_message != nullptr || lane.active_turn.has_value()) {
        lane.state = lane.mailbox.empty()
            ? common_agent_runtime_session_lane_state::running
            : common_agent_runtime_session_lane_state::running_with_waiters;
        return;
    }
    lane.state = lane.mailbox.empty()
        ? common_agent_runtime_session_lane_state::idle
        : common_agent_runtime_session_lane_state::running_with_waiters;
}

namespace {

bool is_waiting_disposition(common_agent_runtime_turn_disposition disposition) {
    return disposition == common_agent_runtime_turn_disposition::wait_for_inference ||
        disposition == common_agent_runtime_turn_disposition::wait_for_tool;
}

bool is_terminal_disposition(common_agent_runtime_turn_disposition disposition) {
    return disposition == common_agent_runtime_turn_disposition::completed ||
        disposition == common_agent_runtime_turn_disposition::failed ||
        disposition == common_agent_runtime_turn_disposition::cancelled;
}

common_agent_daemon_event_type completed_event_for_operation_kind(
        common_agent_runtime_pending_operation_kind kind) {
    switch (kind) {
        case common_agent_runtime_pending_operation_kind::tool:
            return common_agent_daemon_event_type::tool_completed;
        case common_agent_runtime_pending_operation_kind::inference:
            return common_agent_daemon_event_type::turn_started;
    }
    return common_agent_daemon_event_type::turn_started;
}

} // namespace

bool common_agent_runtime_session_manager::run_lane_turn(
        common_agent_runtime_session_lane & lane,
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message) {
    if (message == nullptr) {
        return false;
    }

    auto & request = message->request;
    auto & result = message->result;
    auto & error = message->error;
    {
        std::lock_guard<std::mutex> lock(lane.mutex);
        if (!lane.active_turn.has_value()) {
            lane.active_turn = common_agent_runtime_turn_execution{
                request.request_id,
                request.turn.turn_id,
                request.turn.mode,
                common_agent_runtime_turn_phase::queued,
                common_agent_runtime_turn_disposition::continue_immediately,
                request.turn.execution_control.is_cancel_requested(),
                request.turn.execution_control.cancellation,
            };
        }
    }

    const auto disposition = advance_lane_turn(lane, message);

    std::lock_guard<std::mutex> lock(lane.mutex);
    if (!lane.active_turn.has_value()) {
        return disposition == common_agent_runtime_turn_disposition::completed;
    }

    lane.last_turn_id = lane.active_turn->turn_id;
    lane.last_turn_phase = lane.active_turn->phase;
    lane.last_turn_disposition = disposition;
    if (disposition == common_agent_runtime_turn_disposition::completed) {
        lane.active_turn->phase = common_agent_runtime_turn_phase::completed;
        lane.last_turn_phase = lane.active_turn->phase;
    }
    if (is_terminal_disposition(disposition)) {
        lane.pending_operation.reset();
        lane.active_turn.reset();
    }

    return disposition == common_agent_runtime_turn_disposition::completed;
}

common_agent_runtime_turn_disposition common_agent_runtime_session_manager::poll_pending_operation(
        common_agent_runtime_session_lane & lane,
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message,
        common_agent_runtime_turn_phase phase) {
    auto & request = message->request;
    auto & result = message->result;
    auto & error = message->error;

    if (request.turn.execution_control.should_stop()) {
        {
            std::lock_guard<std::mutex> lock(lane.mutex);
            lane.pending_operation.reset();
            if (lane.active_turn.has_value()) {
                lane.active_turn->pending_operation.reset();
                lane.active_turn->cancellation_requested = true;
                lane.active_turn->disposition = common_agent_runtime_turn_disposition::cancelled;
                lane.active_turn->phase = common_agent_runtime_turn_phase::cancelled;
            }
        }
        result = {};
        result.cancelled = true;
        result.error = request.turn.execution_control.stop_reason();
        error = result.error;
        emit_event(
            common_agent_daemon_event_type::turn_cancelled,
            request.request_id,
            request.turn.turn_id,
            result.error);
        return common_agent_runtime_turn_disposition::cancelled;
    }

    if (!lane.pending_operation.has_value()) {
        error = "lane phase is missing its pending operation";
        {
            std::lock_guard<std::mutex> lock(lane.mutex);
            if (lane.active_turn.has_value()) {
                lane.active_turn->pending_operation.reset();
                lane.active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                lane.active_turn->phase = common_agent_runtime_turn_phase::failed;
            }
        }
        emit_event(
            common_agent_daemon_event_type::turn_failed,
            request.request_id,
            request.turn.turn_id,
            error);
        return common_agent_runtime_turn_disposition::failed;
    }

    bool ready = false;
    if (!lane.pending_operation->poll(ready, error)) {
        std::lock_guard<std::mutex> lock(lane.mutex);
        lane.pending_operation.reset();
        if (lane.active_turn.has_value()) {
            lane.active_turn->pending_operation.reset();
            lane.active_turn->disposition = common_agent_runtime_turn_disposition::failed;
            lane.active_turn->phase = common_agent_runtime_turn_phase::failed;
        }
        emit_event(
            common_agent_daemon_event_type::turn_failed,
            request.request_id,
            request.turn.turn_id,
            error);
        return common_agent_runtime_turn_disposition::failed;
    }
    if (!ready) {
        std::lock_guard<std::mutex> lock(lane.mutex);
        if (lane.active_turn.has_value()) {
            lane.active_turn->disposition = lane.pending_operation->waiting_disposition;
        }
        return lane.pending_operation->waiting_disposition;
    }

    const auto operation_kind = lane.pending_operation->pending_operation.kind;
    {
        std::lock_guard<std::mutex> lock(lane.mutex);
        if (lane.active_turn.has_value()) {
            lane.active_turn->pending_operation.reset();
            lane.active_turn->disposition = common_agent_runtime_turn_disposition::continue_immediately;
            lane.active_turn->phase = phase;
        }
        lane.pending_operation.reset();
    }
    emit_event(
        completed_event_for_operation_kind(operation_kind),
        request.request_id,
        request.turn.turn_id,
        operation_kind == common_agent_runtime_pending_operation_kind::tool
            ? "manager-owned pending tool operation completed"
            : "manager-owned pending inference operation completed");
    if (operation_kind == common_agent_runtime_pending_operation_kind::tool) {
        emit_event(
            common_agent_daemon_event_type::turn_started,
            request.request_id,
            request.turn.turn_id,
            "turn resumed after pending tool operation");
    }
    return common_agent_runtime_turn_disposition::continue_immediately;
}

common_agent_runtime_turn_disposition common_agent_runtime_session_manager::advance_lane_turn(
        common_agent_runtime_session_lane & lane,
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message) {
    if (message == nullptr) {
        return common_agent_runtime_turn_disposition::failed;
    }

    auto & request = message->request;
    auto & result = message->result;
    auto & error = message->error;
    common_agent_runtime_turn_phase phase;
    {
        std::lock_guard<std::mutex> lock(lane.mutex);
        if (!lane.active_turn.has_value()) {
            error = "lane does not have an active turn";
            return common_agent_runtime_turn_disposition::failed;
        }
        phase = lane.active_turn->phase;
    }

    switch (phase) {
        case common_agent_runtime_turn_phase::queued:
            if (request.turn.execution_control.should_stop()) {
                {
                    std::lock_guard<std::mutex> lock(lane.mutex);
                    if (lane.active_turn.has_value()) {
                        lane.active_turn->cancellation_requested = true;
                        lane.active_turn->disposition = common_agent_runtime_turn_disposition::cancelled;
                        lane.active_turn->phase = common_agent_runtime_turn_phase::cancelled;
                    }
                }
                result = {};
                result.cancelled = true;
                result.error = request.turn.execution_control.stop_reason();
                error = result.error;
                emit_event(
                    common_agent_daemon_event_type::turn_cancelled,
                    request.request_id,
                    request.turn.turn_id,
                    result.error);
                return common_agent_runtime_turn_disposition::cancelled;
            }
            {
                std::lock_guard<std::mutex> lock(lane.mutex);
                if (lane.active_turn.has_value()) {
                    lane.active_turn->phase = common_agent_runtime_turn_phase::preparing;
                }
            }
            return common_agent_runtime_turn_disposition::continue_immediately;

        case common_agent_runtime_turn_phase::preparing:
            if (config.pending_operation_resolver) {
                std::optional<common_agent_runtime_session_manager_pending_operation> pending_operation;
                if (!config.pending_operation_resolver(request.turn, pending_operation, error)) {
                    {
                        std::lock_guard<std::mutex> lock(lane.mutex);
                        lane.pending_operation.reset();
                        if (lane.active_turn.has_value()) {
                            lane.active_turn->pending_operation.reset();
                            lane.active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                            lane.active_turn->phase = common_agent_runtime_turn_phase::failed;
                        }
                    }
                    emit_event(
                        common_agent_daemon_event_type::turn_failed,
                        request.request_id,
                        request.turn.turn_id,
                        error);
                    return common_agent_runtime_turn_disposition::failed;
                }
                if (pending_operation.has_value()) {
                    {
                        std::lock_guard<std::mutex> lock(lane.mutex);
                        lane.pending_operation = std::move(pending_operation);
                        if (lane.active_turn.has_value()) {
                            lane.active_turn->phase = lane.pending_operation->waiting_phase;
                            lane.active_turn->disposition = lane.pending_operation->waiting_disposition;
                            lane.active_turn->pending_operation = lane.pending_operation->pending_operation;
                        }
                    }
                    emit_event(
                        lane.pending_operation->pending_operation.kind ==
                            common_agent_runtime_pending_operation_kind::tool
                            ? common_agent_daemon_event_type::turn_waiting_for_tool
                            : common_agent_daemon_event_type::turn_waiting_for_inference,
                        request.request_id,
                        request.turn.turn_id,
                        lane.pending_operation->pending_operation.detail.empty()
                            ? "lane entered manager-owned pending operation"
                            : lane.pending_operation->pending_operation.detail);
                    return lane.pending_operation->waiting_disposition;
                }
            }
            {
                std::lock_guard<std::mutex> lock(lane.mutex);
                if (lane.active_turn.has_value()) {
                    lane.active_turn->phase = common_agent_runtime_turn_phase::awaiting_inference;
                    common_agent_runtime_pending_operation pending;
                    pending.operation_id = "inference:" + request.request_id;
                    pending.kind = common_agent_runtime_pending_operation_kind::inference;
                    pending.detail = "session host turn execution";
                    if (request.turn.execution_control.deadline.has_value()) {
                        pending.deadline = *request.turn.execution_control.deadline;
                    }
                    lane.active_turn->pending_operation = std::move(pending);
                }
            }
            emit_event(
                common_agent_daemon_event_type::turn_waiting_for_inference,
                request.request_id,
                request.turn.turn_id,
                "turn entered inference execution");
            return common_agent_runtime_turn_disposition::continue_immediately;

        case common_agent_runtime_turn_phase::awaiting_inference: {
            if (lane.pending_operation.has_value()) {
                const auto pending_disposition = poll_pending_operation(
                    lane,
                    message,
                    common_agent_runtime_turn_phase::awaiting_inference);
                if (pending_disposition != common_agent_runtime_turn_disposition::continue_immediately) {
                    return pending_disposition;
                }
            }
            const bool ok = lane.host->run_turn(request.turn, result, error);
            const auto disposition = ok
                ? common_agent_runtime_turn_disposition::completed
                : (result.cancelled
                    ? common_agent_runtime_turn_disposition::cancelled
                    : common_agent_runtime_turn_disposition::failed);
            {
                std::lock_guard<std::mutex> lock(lane.mutex);
                if (lane.active_turn.has_value()) {
                    lane.active_turn->cancellation_requested = request.turn.execution_control.is_cancel_requested();
                    lane.active_turn->disposition = disposition;
                    lane.active_turn->pending_operation.reset();
                    lane.active_turn->phase = ok
                        ? common_agent_runtime_turn_phase::completing
                        : (result.cancelled
                            ? common_agent_runtime_turn_phase::cancelled
                            : common_agent_runtime_turn_phase::failed);
                }
            }
            emit_event(
                ok
                    ? common_agent_daemon_event_type::turn_completed
                    : (result.cancelled
                        ? common_agent_daemon_event_type::turn_cancelled
                        : common_agent_daemon_event_type::turn_failed),
                request.request_id,
                request.turn.turn_id,
                ok ? "turn execution returned" : error);
            return disposition;
        }

        case common_agent_runtime_turn_phase::awaiting_tool:
            return poll_pending_operation(
                lane,
                message,
                common_agent_runtime_turn_phase::awaiting_inference);

        case common_agent_runtime_turn_phase::completing:
            {
                std::lock_guard<std::mutex> lock(lane.mutex);
                if (lane.active_turn.has_value()) {
                    lane.active_turn->disposition = common_agent_runtime_turn_disposition::completed;
                }
            }
            return common_agent_runtime_turn_disposition::completed;

        case common_agent_runtime_turn_phase::completed:
            {
                std::lock_guard<std::mutex> lock(lane.mutex);
                if (lane.active_turn.has_value()) {
                    lane.active_turn->disposition = common_agent_runtime_turn_disposition::completed;
                }
            }
            return common_agent_runtime_turn_disposition::completed;

        case common_agent_runtime_turn_phase::failed:
            {
                std::lock_guard<std::mutex> lock(lane.mutex);
                if (lane.active_turn.has_value()) {
                    lane.active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                }
            }
            return common_agent_runtime_turn_disposition::failed;

        case common_agent_runtime_turn_phase::cancelled:
            {
                std::lock_guard<std::mutex> lock(lane.mutex);
                if (lane.active_turn.has_value()) {
                    lane.active_turn->disposition = common_agent_runtime_turn_disposition::cancelled;
                }
            }
            return common_agent_runtime_turn_disposition::cancelled;
    }

    error = "unsupported lane turn phase";
    return common_agent_runtime_turn_disposition::failed;
}

bool common_agent_runtime_session_manager::drain_lane(
        common_agent_runtime_session_lane & lane,
        const std::shared_ptr<common_agent_runtime_session_lane_message> & target_message,
        std::string & error) {
    const auto resolve_lane_error = [](
            const common_agent_runtime_session_lane_message & message) {
        if (!message.error.empty()) {
            return message.error;
        }
        if (!message.result.error.empty()) {
            return message.result.error;
        }
        return std::string();
    };

    bool already_draining = false;
    {
        std::lock_guard<std::mutex> lock(lane.mutex);
        if (lane.state != common_agent_runtime_session_lane_state::idle) {
            already_draining = true;
        } else {
            lane.state = common_agent_runtime_session_lane_state::running;
        }
    }
    if (already_draining) {
        return wait_for_message_completion(target_message, error);
    }
    while (true) {
        {
            std::lock_guard<std::mutex> lock(lane.mutex);
            if (lane.active_turn.has_value()) {
                lane.state = common_agent_runtime_session_lane_state::idle;
                error = "session already has an active turn";
                return false;
            }
        }

        std::shared_ptr<common_agent_runtime_session_lane_message> message;
        {
            std::lock_guard<std::mutex> lock(lane.mutex);
            if (lane.mailbox.empty()) {
                break;
            }
            message = std::move(lane.mailbox.front());
            lane.mailbox.pop_front();
            lane.current_message = message;
        }
        reconcile_lane_state(lane);
        if (message == nullptr) {
            std::lock_guard<std::mutex> lock(lane.mutex);
            lane.current_message.reset();
            lane.state = common_agent_runtime_session_lane_state::idle;
            error = "lane mailbox message is missing result storage";
            return false;
        }

        while (true) {
            const bool completed = run_lane_turn(lane, message);
            if (completed) {
                std::lock_guard<std::mutex> message_lock(message->mutex);
                message->ok = true;
                message->completed = true;
                break;
            }
            bool has_active_turn = false;
            common_agent_runtime_turn_disposition disposition = common_agent_runtime_turn_disposition::failed;
            {
                std::lock_guard<std::mutex> lock(lane.mutex);
                has_active_turn = lane.active_turn.has_value();
                if (has_active_turn) {
                    disposition = lane.active_turn->disposition;
                }
            }
            if (!has_active_turn) {
                {
                    std::lock_guard<std::mutex> message_lock(message->mutex);
                    message->ok = false;
                    message->completed = true;
                }
                message->condition.notify_all();
                if (message == target_message) {
                    error = resolve_lane_error(*message);
                }
                break;
            }
            if (is_waiting_disposition(disposition)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            if (disposition != common_agent_runtime_turn_disposition::continue_immediately) {
                {
                    std::lock_guard<std::mutex> message_lock(message->mutex);
                    message->ok = false;
                    message->completed = true;
                }
                message->condition.notify_all();
                if (message == target_message) {
                    error = resolve_lane_error(*message);
                }
                break;
            }
        }

        {
            std::lock_guard<std::mutex> message_lock(message->mutex);
            message->completed = true;
        }
        message->condition.notify_all();

        if (message == target_message) {
            error = resolve_lane_error(*message);
        }

        {
            std::lock_guard<std::mutex> lock(lane.mutex);
            if (lane.current_message == message) {
                lane.current_message.reset();
            }
        }
        reconcile_lane_state(lane);
    }

    reconcile_lane_state(lane);
    emit_event(
        common_agent_daemon_event_type::lane_drained,
        target_message ? target_message->request.request_id : std::string(),
        target_message ? target_message->request.turn.turn_id : std::string(),
        "session lane mailbox drained");
    if (target_message != nullptr) {
        return wait_for_message_completion(target_message, error);
    }
    error.clear();
    return true;
}

bool common_agent_runtime_session_manager::prepare_lane_transition(
        common_agent_runtime_session_lane & lane,
        common_agent_runtime_session_lane_state target_state,
        const char * pending_error,
        std::shared_ptr<common_agent_runtime_session_lane_message> & current_message,
        std::string & error) {
    std::deque<std::shared_ptr<common_agent_runtime_session_lane_message>> dropped_messages;
    {
        std::lock_guard<std::mutex> lock(lane.mutex);
        if (lane.state == common_agent_runtime_session_lane_state::resetting) {
            error = "session lane is already resetting";
            return false;
        }
        if (lane.state == common_agent_runtime_session_lane_state::closing) {
            error = "session lane is already closing";
            return false;
        }

        lane.state = target_state;
        current_message = lane.current_message;
        dropped_messages = std::move(lane.mailbox);
        lane.mailbox.clear();
    }

    for (const auto & message : dropped_messages) {
        complete_lane_message(message, false, pending_error);
    }

    error.clear();
    return true;
}

bool common_agent_runtime_session_manager::run_turn(
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error) {
    auto & lane = ensure_session_lane(make_session_key(request));
    auto message = enqueue_lane_turn(lane, request, result, error);
    if (message == nullptr) {
        result = {};
        result.error = error;
        return false;
    }
    const bool ok = drain_lane(lane, message, error);
    result = message->result;
    if (result.error.empty()) {
        result.error = message->error;
    }
    if (error.empty()) {
        error = message->error;
    }
    return ok;
}

bool common_agent_runtime_session_manager::reset_session(
        const common_agent_runtime_session_key & key,
        std::string & error) {
    auto it = lanes.find(key);
    if (it == lanes.end()) {
        error = "session is not active";
        return false;
    }
    emit_event(
        common_agent_daemon_event_type::session_reset_requested,
        {},
        {},
        key.namespace_id + "/" + key.session_id);

    std::shared_ptr<common_agent_runtime_session_lane_message> current_message;
    if (!prepare_lane_transition(
                it->second,
                common_agent_runtime_session_lane_state::resetting,
                "session lane reset before turn execution",
                current_message,
                error)) {
        return false;
    }

    std::string wait_error;
    if (current_message != nullptr && !wait_for_message_completion(current_message, wait_error)) {
        wait_error.clear();
    }

    it->second.host->reset();
    {
        std::lock_guard<std::mutex> lock(it->second.mutex);
        it->second.mailbox.clear();
        it->second.current_message.reset();
        it->second.active_turn.reset();
        it->second.pending_operation.reset();
        it->second.last_turn_id.clear();
        it->second.last_turn_phase = common_agent_runtime_turn_phase::queued;
        it->second.last_turn_disposition = common_agent_runtime_turn_disposition::continue_immediately;
        it->second.state = common_agent_runtime_session_lane_state::idle;
    }
    error.clear();
    emit_event(
        common_agent_daemon_event_type::session_reset,
        {},
        {},
        key.namespace_id + "/" + key.session_id);
    return true;
}

bool common_agent_runtime_session_manager::close_session(
        const common_agent_runtime_session_key & key,
        std::string & error) {
    auto it = lanes.find(key);
    if (it == lanes.end()) {
        error = "session is not active";
        return false;
    }
    emit_event(
        common_agent_daemon_event_type::session_close_requested,
        {},
        {},
        key.namespace_id + "/" + key.session_id);

    std::shared_ptr<common_agent_runtime_session_lane_message> current_message;
    if (!prepare_lane_transition(
                it->second,
                common_agent_runtime_session_lane_state::closing,
                "session lane closed before turn execution",
                current_message,
                error)) {
        return false;
    }

    std::string wait_error;
    if (current_message != nullptr && !wait_for_message_completion(current_message, wait_error)) {
        wait_error.clear();
    }

    it->second.host->reset();
    lanes.erase(it);
    error.clear();
    emit_event(
        common_agent_daemon_event_type::session_closed,
        {},
        {},
        key.namespace_id + "/" + key.session_id);
    return true;
}

bool common_agent_runtime_session_manager::request_cancel_active_turn(
        const std::string & target_request_id,
        const std::string & target_turn_id,
        common_agent_runtime_active_turn_descriptor & active_turn,
        std::string & error) {
    active_turn = {};
    for (auto & entry : lanes) {
        auto & lane = entry.second;
        std::lock_guard<std::mutex> lock(lane.mutex);
        if (!lane.active_turn.has_value()) {
            continue;
        }

        const bool request_match =
            !target_request_id.empty() &&
            lane.active_turn->request_id == target_request_id;
        const bool turn_match =
            !target_turn_id.empty() &&
            lane.active_turn->turn_id == target_turn_id;
        if (!request_match && !turn_match) {
            continue;
        }

        const auto descriptor = lane.host->describe_session();
        active_turn = {
            entry.first,
            descriptor.project_id,
            lane.current_message ? lane.current_message->request.request_id : lane.active_turn->request_id,
            lane.current_message ? lane.current_message->request.turn.turn_id : lane.active_turn->turn_id,
            common_agent_runtime_turn_phase_name(lane.active_turn->phase),
            common_agent_runtime_turn_disposition_name(lane.active_turn->disposition),
            lane.active_turn->cancellation_requested,
            lane.active_turn->pending_operation.has_value()
                ? common_agent_runtime_pending_operation_kind_name(
                    lane.active_turn->pending_operation->kind)
                : std::string(),
            lane.active_turn->pending_operation.has_value()
                ? lane.active_turn->pending_operation->detail
                : std::string(),
        };

        if (!lane.active_turn->cancellation) {
            error = "active turn does not expose a cancellation handle";
            return false;
        }

        lane.active_turn->cancellation->request_cancel("turn cancelled by host");
        lane.active_turn->cancellation_requested = lane.active_turn->cancellation->is_cancelled();
        active_turn.cancellation_requested = lane.active_turn->cancellation_requested;
        error.clear();
        return true;
    }

    error = "target turn is not active";
    return false;
}

std::optional<common_agent_runtime_active_turn_descriptor> common_agent_runtime_session_manager::describe_active_turn() const {
    for (const auto & entry : lanes) {
        const auto & lane = entry.second;
        std::lock_guard<std::mutex> lock(lane.mutex);
        if (!lane.active_turn.has_value()) {
            continue;
        }

        const auto descriptor = lane.host->describe_session();
        return common_agent_runtime_active_turn_descriptor{
            entry.first,
            descriptor.project_id,
            lane.current_message ? lane.current_message->request.request_id : lane.active_turn->request_id,
            lane.current_message ? lane.current_message->request.turn.turn_id : lane.active_turn->turn_id,
            common_agent_runtime_turn_phase_name(lane.active_turn->phase),
            common_agent_runtime_turn_disposition_name(lane.active_turn->disposition),
            lane.active_turn->cancellation_requested,
            lane.active_turn->pending_operation.has_value()
                ? common_agent_runtime_pending_operation_kind_name(
                    lane.active_turn->pending_operation->kind)
                : std::string(),
            lane.active_turn->pending_operation.has_value()
                ? lane.active_turn->pending_operation->detail
                : std::string(),
        };
    }

    return std::nullopt;
}

std::vector<common_agent_runtime_session_descriptor> common_agent_runtime_session_manager::list_sessions() const {
    std::vector<common_agent_runtime_session_descriptor> sessions;
    sessions.reserve(lanes.size());
    for (const auto & entry : lanes) {
        const auto descriptor = entry.second.host->describe_session();
        std::lock_guard<std::mutex> lock(entry.second.mutex);
        sessions.push_back({
            entry.first,
            descriptor.project_id,
            descriptor.memory_scope,
            descriptor.plan_scope,
            descriptor.policy_pack_id,
            common_agent_runtime_session_lane_state_name(entry.second.state),
            entry.second.mailbox.size(),
            entry.second.active_turn.has_value(),
            entry.second.active_turn.has_value()
                ? (entry.second.current_message
                    ? entry.second.current_message->request.request_id
                    : entry.second.active_turn->request_id)
                : std::string(),
            entry.second.active_turn.has_value()
                ? (entry.second.current_message
                    ? entry.second.current_message->request.turn.turn_id
                    : entry.second.active_turn->turn_id)
                : std::string(),
            entry.second.active_turn.has_value()
                ? common_agent_runtime_turn_phase_name(entry.second.active_turn->phase)
                : std::string(),
            entry.second.active_turn.has_value()
                ? common_agent_runtime_turn_disposition_name(entry.second.active_turn->disposition)
                : std::string(),
            entry.second.active_turn.has_value() && entry.second.active_turn->cancellation_requested,
            entry.second.active_turn.has_value() && entry.second.active_turn->pending_operation.has_value()
                ? common_agent_runtime_pending_operation_kind_name(
                    entry.second.active_turn->pending_operation->kind)
                : std::string(),
            entry.second.active_turn.has_value() && entry.second.active_turn->pending_operation.has_value()
                ? entry.second.active_turn->pending_operation->detail
                : std::string(),
            entry.second.last_turn_id,
            entry.second.last_turn_id.empty()
                ? std::string()
                : common_agent_runtime_turn_phase_name(entry.second.last_turn_phase),
            entry.second.last_turn_id.empty()
                ? std::string()
                : common_agent_runtime_turn_disposition_name(entry.second.last_turn_disposition),
        });
    }
    return sessions;
}

void common_agent_runtime_session_manager::reset_all() {
    std::vector<common_agent_runtime_session_key> keys;
    keys.reserve(lanes.size());
    for (const auto & entry : lanes) {
        keys.push_back(entry.first);
    }

    std::string error;
    for (const auto & key : keys) {
        close_session(key, error);
        error.clear();
    }
}
