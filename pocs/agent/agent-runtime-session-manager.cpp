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
        common_agent_runtime_session_lane lane;
        lane.host = std::make_unique<common_agent_runtime_session_host>(config);
        it = lanes.emplace(key, std::move(lane)).first;
    }
    return it->second;
}

bool common_agent_runtime_session_manager::run_lane_turn(
        common_agent_runtime_session_lane & lane,
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error) {
    lane.active_turn = common_agent_runtime_turn_execution{
        request.request_id,
        request.turn.turn_id,
        request.turn.mode,
        common_agent_runtime_turn_phase::queued,
        common_agent_runtime_turn_disposition::continue_immediately,
        request.turn.execution_control.is_cancel_requested(),
        request.turn.execution_control.cancellation,
    };

    struct lane_guard {
        common_agent_runtime_session_lane & lane;

        ~lane_guard() {
            lane.active_turn.reset();
        }
    } guard{lane};

    const auto disposition = advance_lane_turn(lane, request, result, error);

    lane.last_turn_id = lane.active_turn->turn_id;
    lane.last_turn_phase = lane.active_turn->phase;
    lane.last_turn_disposition = disposition;
    if (disposition == common_agent_runtime_turn_disposition::completed) {
        lane.active_turn->phase = common_agent_runtime_turn_phase::completed;
        lane.last_turn_phase = lane.active_turn->phase;
    }

    return disposition == common_agent_runtime_turn_disposition::completed;
}

common_agent_runtime_turn_disposition common_agent_runtime_session_manager::advance_lane_turn(
        common_agent_runtime_session_lane & lane,
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error) {
    if (!lane.active_turn.has_value()) {
        error = "lane does not have an active turn";
        return common_agent_runtime_turn_disposition::failed;
    }

    auto & turn = *lane.active_turn;
    switch (turn.phase) {
        case common_agent_runtime_turn_phase::queued:
            if (request.turn.execution_control.should_stop()) {
                turn.cancellation_requested = true;
                turn.disposition = common_agent_runtime_turn_disposition::cancelled;
                turn.phase = common_agent_runtime_turn_phase::cancelled;
                result = {};
                result.cancelled = true;
                result.error = request.turn.execution_control.stop_reason();
                error = result.error;
                return turn.disposition;
            }
            turn.phase = common_agent_runtime_turn_phase::preparing;
            return common_agent_runtime_turn_disposition::continue_immediately;

        case common_agent_runtime_turn_phase::preparing:
            turn.phase = common_agent_runtime_turn_phase::awaiting_inference;
            return common_agent_runtime_turn_disposition::continue_immediately;

        case common_agent_runtime_turn_phase::awaiting_inference: {
            const bool ok = lane.host->run_turn(request.turn, result, error);
            turn.cancellation_requested = request.turn.execution_control.is_cancel_requested();
            turn.disposition = ok
                ? common_agent_runtime_turn_disposition::completed
                : (result.cancelled
                    ? common_agent_runtime_turn_disposition::cancelled
                    : common_agent_runtime_turn_disposition::failed);
            turn.phase = ok
                ? common_agent_runtime_turn_phase::completing
                : (result.cancelled
                    ? common_agent_runtime_turn_phase::cancelled
                    : common_agent_runtime_turn_phase::failed);
            return turn.disposition;
        }

        case common_agent_runtime_turn_phase::awaiting_tool:
            error = "awaiting_tool advancement is not implemented yet";
            turn.disposition = common_agent_runtime_turn_disposition::failed;
            turn.phase = common_agent_runtime_turn_phase::failed;
            return turn.disposition;

        case common_agent_runtime_turn_phase::completing:
            turn.disposition = common_agent_runtime_turn_disposition::completed;
            return turn.disposition;

        case common_agent_runtime_turn_phase::completed:
            turn.disposition = common_agent_runtime_turn_disposition::completed;
            return turn.disposition;

        case common_agent_runtime_turn_phase::failed:
            turn.disposition = common_agent_runtime_turn_disposition::failed;
            return turn.disposition;

        case common_agent_runtime_turn_phase::cancelled:
            turn.disposition = common_agent_runtime_turn_disposition::cancelled;
            return turn.disposition;
    }

    error = "unsupported lane turn phase";
    return common_agent_runtime_turn_disposition::failed;
}

bool common_agent_runtime_session_manager::drain_lane(
        common_agent_runtime_session_lane & lane,
        std::string & error) {
    while (!lane.mailbox.empty()) {
        if (lane.active_turn.has_value()) {
            error = "session already has an active turn";
            return false;
        }

        auto message = std::move(lane.mailbox.front());
        lane.mailbox.pop_front();
        if (message.result == nullptr || message.error == nullptr) {
            error = "lane mailbox message is missing result/error storage";
            return false;
        }

        while (true) {
            const bool completed = run_lane_turn(lane, message.request, *message.result, *message.error);
            if (completed) {
                break;
            }
            if (!lane.active_turn.has_value()) {
                error = *message.error;
                return false;
            }
            const auto disposition = lane.active_turn->disposition;
            if (disposition != common_agent_runtime_turn_disposition::continue_immediately) {
                error = *message.error;
                return false;
            }
        }
    }

    error.clear();
    return true;
}

bool common_agent_runtime_session_manager::run_turn(
        const common_agent_runtime_session_manager_turn_request & request,
        common_agent_runtime_session_manager_turn_result & result,
        std::string & error) {
    auto & lane = ensure_session_lane(make_session_key(request));
    if (lane.active_turn.has_value()) {
        result = {};
        result.error = "session already has an active turn";
        error = result.error;
        return false;
    }
    lane.mailbox.push_back({request, &result, &error});
    return drain_lane(lane, error);
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
    it->second.mailbox.clear();
    it->second.active_turn.reset();
    it->second.last_turn_id.clear();
    it->second.last_turn_phase = common_agent_runtime_turn_phase::queued;
    it->second.last_turn_disposition = common_agent_runtime_turn_disposition::continue_immediately;
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
            lane.active_turn->request_id,
            lane.active_turn->turn_id,
            common_agent_runtime_turn_phase_name(lane.active_turn->phase),
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
        if (!lane.active_turn.has_value()) {
            continue;
        }

        const auto descriptor = lane.host->describe_session();
        return common_agent_runtime_active_turn_descriptor{
            entry.first,
            descriptor.project_id,
            lane.active_turn->request_id,
            lane.active_turn->turn_id,
            common_agent_runtime_turn_phase_name(lane.active_turn->phase),
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
        sessions.push_back({
            entry.first,
            descriptor.project_id,
            descriptor.memory_scope,
            descriptor.plan_scope,
            descriptor.policy_pack_id,
            entry.second.mailbox.size(),
            entry.second.active_turn.has_value(),
            entry.second.active_turn.has_value() ? entry.second.active_turn->request_id : std::string(),
            entry.second.active_turn.has_value() ? entry.second.active_turn->turn_id : std::string(),
            entry.second.active_turn.has_value()
                ? common_agent_runtime_turn_phase_name(entry.second.active_turn->phase)
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
