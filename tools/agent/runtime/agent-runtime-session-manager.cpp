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

common_agent_event_emitter common_agent_runtime_session_manager::make_lane_emitter(
        const common_agent_runtime_session_key & key,
        const std::string & project_id,
        const std::string & request_id,
        const std::string & turn_id,
        const std::string & operation_id) const {
    return common_agent_event_emitter(
        event_sink,
        {
            key.namespace_id,
            project_id,
            key.session_id,
            request_id,
            turn_id,
            operation_id,
        });
}

std::shared_ptr<common_agent_runtime_session_manager::common_agent_runtime_session_lane_message>
common_agent_runtime_session_manager::enqueue_lane_turn(
        common_agent_runtime_session_lane & lane,
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result &,
        std::string & error) {
    auto message = std::make_shared<common_agent_runtime_session_lane_message>();
    message->request = request;
    message->request.turn.request_id = request.request_id;
    {
        std::lock_guard<std::mutex> lock(lane.mutex);
        if (lane.state == common_agent_runtime_session_lane_state::resetting) {
            error = "session lane is resetting";
            return nullptr;
        }
        if (lane.state == common_agent_runtime_session_lane_state::closing) {
            error = "session lane is closing";
            return nullptr;
        }
        message->id = lane.next_message_id++;
        lane.mailbox.push_back(message);
        if (lane.state == common_agent_runtime_session_lane_state::running ||
                lane.state == common_agent_runtime_session_lane_state::running_with_waiters) {
            lane.state = common_agent_runtime_session_lane_state::running_with_waiters;
        }
    }
    make_lane_emitter(
        make_session_key(request),
        request.turn.project_id,
        request.request_id,
        request.turn.turn_id)
        .emit(
            common_agent_daemon_event_type::turn_accepted,
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
            lane.active_turn = make_common_agent_runtime_turn_execution(
                request.request_id,
                request.turn.turn_id,
                request.turn.mode,
                request.turn.execution_control.is_cancel_requested(),
                request.turn.execution_control.cancellation);
        }
    }

    const auto disposition = advance_lane_turn(lane, message);

    const bool terminal = is_terminal_disposition(disposition);
    {
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
        if (terminal) {
            lane.pending_operation.reset();
            lane.active_turn.reset();
        }
    }
    if (terminal) {
        operation_manager.cleanup_terminal();
    }

    return disposition == common_agent_runtime_turn_disposition::completed;
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
    const auto turn_events = make_lane_emitter(
        make_session_key(request),
        request.turn.project_id,
        request.request_id,
        request.turn.turn_id);
    {
        std::lock_guard<std::mutex> lock(lane.mutex);
        if (!lane.active_turn.has_value()) {
            error = "lane does not have an active turn";
            return common_agent_runtime_turn_disposition::failed;
        }
    }
    return advance_common_agent_runtime_turn(
        lane.host.get(),
        lane.pending_operation,
        lane.active_turn,
        lane.mutex,
        operation_manager,
        config.pending_operation_resolver,
        config.inference_gate,
        config.inference_executor,
        request.request_id,
        request.turn,
        result,
        error,
        turn_events);

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
    if (target_message != nullptr) {
        make_lane_emitter(
            make_session_key(target_message->request),
            target_message->request.turn.project_id,
            target_message->request.request_id,
            target_message->request.turn.turn_id)
            .emit(
                common_agent_daemon_event_type::lane_drained,
                "session lane mailbox drained");
    } else {
        common_agent_event_emitter(event_sink)
            .emit(
                common_agent_daemon_event_type::lane_drained,
                "session lane mailbox drained");
    }
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
    make_lane_emitter(key, {}, {}, {})
        .emit(
            common_agent_daemon_event_type::session_reset_requested,
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
    make_lane_emitter(key, {}, {}, {})
        .emit(
            common_agent_daemon_event_type::session_reset,
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
    make_lane_emitter(key, {}, {}, {})
        .emit(
            common_agent_daemon_event_type::session_close_requested,
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
    make_lane_emitter(key, {}, {}, {})
        .emit(
            common_agent_daemon_event_type::session_closed,
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
