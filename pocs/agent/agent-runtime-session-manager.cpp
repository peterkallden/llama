#include "agent-runtime-session-manager.h"

#include <utility>

common_agent_runtime_session_manager::common_agent_runtime_session_manager(
        common_agent_runtime_session_manager_config config)
    : config(std::move(config)) {}

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
        it->second.host = std::make_unique<common_agent_runtime_session_host>(config);
    }
    return it->second;
}

std::shared_ptr<common_agent_runtime_session_manager::common_agent_runtime_session_lane_message>
common_agent_runtime_session_manager::enqueue_lane_turn(
        common_agent_runtime_session_lane & lane,
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error) {
    auto message = std::make_shared<common_agent_runtime_session_lane_message>();
    message->id = lane.next_message_id++;
    message->request = request;
    message->result = &result;
    message->error = &error;
    std::lock_guard<std::mutex> lock(lane.mutex);
    lane.mailbox.push_back(message);
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
    if (message->error != nullptr) {
        error = *message->error;
    } else {
        error.clear();
    }
    return message->ok;
}

bool common_agent_runtime_session_manager::run_lane_turn(
        common_agent_runtime_session_lane & lane,
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message) {
    if (message == nullptr || message->result == nullptr || message->error == nullptr) {
        return false;
    }

    auto & request = message->request;
    auto & result = *message->result;
    auto & error = *message->error;
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
    if (disposition != common_agent_runtime_turn_disposition::continue_immediately) {
        lane.active_turn.reset();
    }

    return disposition == common_agent_runtime_turn_disposition::completed;
}

common_agent_runtime_turn_disposition common_agent_runtime_session_manager::advance_lane_turn(
        common_agent_runtime_session_lane & lane,
        const std::shared_ptr<common_agent_runtime_session_lane_message> & message) {
    if (message == nullptr || message->result == nullptr || message->error == nullptr) {
        return common_agent_runtime_turn_disposition::failed;
    }

    auto & request = message->request;
    auto & result = *message->result;
    auto & error = *message->error;
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
            {
                std::lock_guard<std::mutex> lock(lane.mutex);
                if (lane.active_turn.has_value()) {
                    lane.active_turn->phase = common_agent_runtime_turn_phase::awaiting_inference;
                }
            }
            return common_agent_runtime_turn_disposition::continue_immediately;

        case common_agent_runtime_turn_phase::awaiting_inference: {
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
                    lane.active_turn->phase = ok
                        ? common_agent_runtime_turn_phase::completing
                        : (result.cancelled
                            ? common_agent_runtime_turn_phase::cancelled
                            : common_agent_runtime_turn_phase::failed);
                }
            }
            return disposition;
        }

        case common_agent_runtime_turn_phase::awaiting_tool:
            error = "awaiting_tool advancement is not implemented yet";
            {
                std::lock_guard<std::mutex> lock(lane.mutex);
                if (lane.active_turn.has_value()) {
                    lane.active_turn->disposition = common_agent_runtime_turn_disposition::failed;
                    lane.active_turn->phase = common_agent_runtime_turn_phase::failed;
                }
            }
            return common_agent_runtime_turn_disposition::failed;

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
        if (message.error != nullptr && !message.error->empty()) {
            return *message.error;
        }
        if (message.result != nullptr && !message.result->error.empty()) {
            return message.result->error;
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
        if (message == nullptr || message->result == nullptr || message->error == nullptr) {
            std::lock_guard<std::mutex> lock(lane.mutex);
            lane.current_message.reset();
            lane.state = common_agent_runtime_session_lane_state::idle;
            error = "lane mailbox message is missing result/error storage";
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
    }

    {
        std::lock_guard<std::mutex> lock(lane.mutex);
        lane.state = common_agent_runtime_session_lane_state::idle;
    }
    if (target_message != nullptr) {
        return wait_for_message_completion(target_message, error);
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
    return drain_lane(lane, message, error);
}

bool common_agent_runtime_session_manager::reset_session(
        const common_agent_runtime_session_key & key,
        std::string & error) {
    auto it = lanes.find(key);
    if (it == lanes.end()) {
        error = "session is not active";
        return false;
    }

    it->second.host->reset();
    {
        std::lock_guard<std::mutex> lock(it->second.mutex);
        it->second.mailbox.clear();
        it->second.current_message.reset();
        it->second.active_turn.reset();
        it->second.last_turn_id.clear();
        it->second.last_turn_phase = common_agent_runtime_turn_phase::queued;
        it->second.last_turn_disposition = common_agent_runtime_turn_disposition::continue_immediately;
        it->second.state = common_agent_runtime_session_lane_state::idle;
    }
    error.clear();
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

    lanes.erase(it);
    error.clear();
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
    lanes.clear();
}
