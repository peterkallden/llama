#include "agent-daemon-service.h"

namespace {

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
    return command.turn->turn_id;
}

} // namespace

common_agent_daemon_service::common_agent_daemon_service(common_agent_daemon_runtime runtime)
    : runtime(std::move(runtime)) {
    state_value = this->runtime.host ? common_agent_daemon_state::ready : common_agent_daemon_state::failed;
}

void common_agent_daemon_service::mark_stopping() {
    if (state_value == common_agent_daemon_state::failed ||
            state_value == common_agent_daemon_state::stopped) {
        return;
    }
    state_value = common_agent_daemon_state::stopping;
}

void common_agent_daemon_service::mark_stopped() {
    if (state_value == common_agent_daemon_state::failed) {
        return;
    }
    state_value = common_agent_daemon_state::stopped;
}

bool common_agent_daemon_service::populate_status(
        common_agent_daemon_command_result & result,
        std::string & error) const {
    result.ok = runtime.host != nullptr;
    result.response_kind = common_agent_daemon_response_kind::status;
    result.event = "status";
    result.status.state = state_value;
    result.status.live = state_value != common_agent_daemon_state::stopped;
    result.status.ready = state_value == common_agent_daemon_state::ready;
    if (runtime.host) {
        result.status.sessions = runtime.host->list_sessions();
        result.status.session_count = result.status.sessions.size();
    }
    append_daemon_event(
        result,
        "status.reported",
        result.request_id,
        {},
        common_agent_daemon_state_name(result.status.state));
    error.clear();
    return result.ok;
}

bool common_agent_daemon_service::execute(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error) {
    auto existing_events = std::move(result.events);
    result = {};
    result.request_id = command.request_id;
    result.events = std::move(existing_events);
    result.daemon_event_count = result.events.size();

    switch (command.type) {
        case common_agent_daemon_command_type::get_status:
            return populate_status(result, error);

        case common_agent_daemon_command_type::cancel_turn:
            error = "cancel_turn is handled by the daemon dispatcher";
            result.error = error;
            append_daemon_event(result, "turn.cancel_rejected", command.request_id, {}, error);
            return false;

        case common_agent_daemon_command_type::reset_session:
            if (!command.session.has_value()) {
                error = "reset_session command missing session payload";
                result.error = error;
                result.response_kind = common_agent_daemon_response_kind::lifecycle;
                result.event = "session_reset_failed";
                append_daemon_event(result, "session.reset_failed", command.request_id, {}, error);
                return false;
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                result.error = error;
                result.response_kind = common_agent_daemon_response_kind::lifecycle;
                result.event = "session_reset_failed";
                append_daemon_event(result, "session.reset_failed", command.request_id, {}, error);
                return false;
            }

            result.ok = runtime.host->reset_session(*command.session, error);
            result.response_kind = common_agent_daemon_response_kind::lifecycle;
            result.event = result.ok ? "session_reset" : "session_reset_failed";
            if (!result.ok) {
                result.error = error;
                append_daemon_event(result, "session.reset_failed", command.request_id, {}, error);
                return false;
            }
            append_daemon_event(result, "session.reset", command.request_id, {}, "session reset");
            error.clear();
            return true;

        case common_agent_daemon_command_type::close_session:
            if (!command.session.has_value()) {
                error = "close_session command missing session payload";
                result.error = error;
                result.response_kind = common_agent_daemon_response_kind::lifecycle;
                result.event = "session_close_failed";
                append_daemon_event(result, "session.close_failed", command.request_id, {}, error);
                return false;
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                result.error = error;
                result.response_kind = common_agent_daemon_response_kind::lifecycle;
                result.event = "session_close_failed";
                append_daemon_event(result, "session.close_failed", command.request_id, {}, error);
                return false;
            }

            result.ok = runtime.host->close_session(*command.session, error);
            result.response_kind = common_agent_daemon_response_kind::lifecycle;
            result.event = result.ok ? "session_closed" : "session_close_failed";
            if (!result.ok) {
                result.error = error;
                append_daemon_event(result, "session.close_failed", command.request_id, {}, error);
                return false;
            }
            append_daemon_event(result, "session.closed", command.request_id, {}, "session closed");
            error.clear();
            return true;

        case common_agent_daemon_command_type::shutdown:
            shutdown_requested_flag = true;
            shutdown_mode_value = common_agent_daemon_shutdown_mode::drain;
            state_value = common_agent_daemon_state::draining;
            result.ok = true;
            result.response_kind = common_agent_daemon_response_kind::lifecycle;
            result.event = "shutdown";
            append_daemon_event(
                result,
                "daemon.shutdown_requested",
                command.request_id,
                {},
                "shutdown requested");
            error.clear();
            return true;

        case common_agent_daemon_command_type::run_turn:
            if (!command.turn.has_value()) {
                error = "run_turn command missing turn payload";
                result.error = error;
                result.response_kind = common_agent_daemon_response_kind::turn;
                append_daemon_event(result, "turn.failed", command.request_id, {}, error);
                return false;
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                result.error = error;
                result.response_kind = common_agent_daemon_response_kind::turn;
                append_daemon_event(
                    result,
                    "turn.failed",
                    command.request_id,
                    command_turn_id(command),
                    error);
                return false;
            }

            error.clear();
            runtime.host->run_turn(*command.turn, result.turn_result, error);
            result.response_kind = common_agent_daemon_response_kind::turn;
            result.ok = result.turn_result.ok;
            if (!error.empty() && result.turn_result.error.empty()) {
                result.turn_result.error = error;
            }
            if (!result.turn_result.error.empty()) {
                result.error = result.turn_result.error;
            }
            append_daemon_event(
                result,
                result.turn_result.cancelled
                    ? "turn.cancelled"
                    : (result.turn_result.ok ? "turn.completed" : "turn.failed"),
                command.request_id,
                command_turn_id(command),
                result.error);
            return result.turn_result.ok;
    }

    error = "unsupported daemon command";
    result.error = error;
    result.response_kind = common_agent_daemon_response_kind::lifecycle;
    append_daemon_event(result, "command.failed", command.request_id, {}, error);
    return false;
}
