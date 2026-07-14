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
    return command.turn->request.turn.turn_id;
}

common_agent_failure_class classify_daemon_turn_failure(
        const common_agent_daemon_command & command) {
    if (command.turn.has_value() &&
            command.turn->request.turn.execution_control.is_deadline_exceeded()) {
        return common_agent_failure_class::timeout;
    }
    return common_agent_failure_class::execution;
}

void populate_daemon_failed_turn_result(
        const common_agent_daemon_command & command,
        common_agent_runtime_session_manager_turn_result & turn_result,
        const std::string & error) {
    turn_result.error = error;
    turn_result.cancelled =
        command.turn.has_value() &&
        command.turn->request.turn.execution_control.should_stop();
    turn_result.failure_class = classify_daemon_turn_failure(command);
    turn_result.response_generation_status =
        turn_result.cancelled
            ? common_agent_generation_status::cancelled
            : common_agent_generation_status::errored;
    turn_result.response_stop_reason =
        turn_result.cancelled
            ? common_agent_generation_stop_reason::cancelled
            : common_agent_generation_stop_reason::error;
}

} // namespace

common_agent_daemon_service::common_agent_daemon_service(common_agent_daemon_runtime runtime)
    : runtime(std::move(runtime)) {
    state_value = this->runtime.host ? common_agent_daemon_state::ready : common_agent_daemon_state::failed;
}

void common_agent_daemon_service::initialize_command_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result) const {
    auto existing_events = std::move(result.events);
    result = {};
    result.request_id = command.request_id;
    result.events = std::move(existing_events);
    result.daemon_event_count = result.events.size();
}

void common_agent_daemon_service::initialize_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result) const {
    initialize_command_result(command, result);
    result.response_kind = common_agent_daemon_response_kind::lifecycle;
}

void common_agent_daemon_service::initialize_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result) const {
    initialize_command_result(command, result);
    result.response_kind = common_agent_daemon_response_kind::turn;
}

bool common_agent_daemon_service::fail_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        std::string daemon_event_type) const {
    initialize_lifecycle_result(command, result);
    result.ok = false;
    result.event = std::move(event);
    result.error = error;
    append_daemon_event(result, std::move(daemon_event_type), command.request_id, {}, error);
    return false;
}

bool common_agent_daemon_service::fail_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        std::string daemon_event_type) const {
    initialize_turn_result(command, result);
    result.ok = false;
    result.event = std::move(event);
    result.error = error;
    populate_daemon_failed_turn_result(command, result.turn_result, error);
    append_daemon_event(
        result,
        std::move(daemon_event_type),
        command.request_id,
        command_turn_id(command),
        error);
    return false;
}

bool common_agent_daemon_service::succeed_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error,
        std::string event,
        std::string daemon_event_type,
        std::string detail) const {
    initialize_lifecycle_result(command, result);
    result.ok = true;
    result.event = std::move(event);
    append_daemon_event(
        result,
        std::move(daemon_event_type),
        command.request_id,
        {},
        std::move(detail));
    error.clear();
    return true;
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

std::optional<common_agent_runtime_active_turn_descriptor> common_agent_daemon_service::describe_active_turn() const {
    if (!runtime.host) {
        return std::nullopt;
    }
    return runtime.host->describe_active_turn();
}

bool common_agent_daemon_service::request_cancel_active_turn(
        const std::string & target_request_id,
        const std::string & target_turn_id,
        common_agent_runtime_active_turn_descriptor & active_turn,
        std::string & error) {
    if (!runtime.host) {
        error = "daemon host is not initialized";
        return false;
    }
    return runtime.host->request_cancel_active_turn(
        target_request_id,
        target_turn_id,
        active_turn,
        error);
}

bool common_agent_daemon_service::execute(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error) {
    initialize_command_result(command, result);

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
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "session_reset_failed",
                    "session.reset_failed");
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "session_reset_failed",
                    "session.reset_failed");
            }

            result.ok = runtime.host->reset_session(command.session->key, error);
            if (!result.ok) {
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "session_reset_failed",
                    "session.reset_failed");
            }
            return succeed_lifecycle_result(
                command,
                result,
                error,
                "session_reset",
                "session.reset",
                "session reset");

        case common_agent_daemon_command_type::close_session:
            if (!command.session.has_value()) {
                error = "close_session command missing session payload";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "session_close_failed",
                    "session.close_failed");
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "session_close_failed",
                    "session.close_failed");
            }

            result.ok = runtime.host->close_session(command.session->key, error);
            if (!result.ok) {
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "session_close_failed",
                    "session.close_failed");
            }
            return succeed_lifecycle_result(
                command,
                result,
                error,
                "session_closed",
                "session.closed",
                "session closed");

        case common_agent_daemon_command_type::shutdown:
            shutdown_requested_flag = true;
            shutdown_mode_value = common_agent_daemon_shutdown_mode::drain;
            state_value = common_agent_daemon_state::draining;
            return succeed_lifecycle_result(
                command,
                result,
                error,
                "shutdown",
                "daemon.shutdown_requested",
                "shutdown requested");

        case common_agent_daemon_command_type::run_turn:
            if (!command.turn.has_value()) {
                error = "run_turn command missing turn payload";
                return fail_turn_result(
                    command,
                    result,
                    error,
                    "turn_failed",
                    "turn.failed");
            }
            if (shutdown_requested_flag || state_value != common_agent_daemon_state::ready) {
                error = "daemon is not accepting new turns";
                return fail_turn_result(
                    command,
                    result,
                    error,
                    "turn_rejected",
                    "turn.rejected");
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_turn_result(
                    command,
                    result,
                    error,
                    "turn_failed",
                    "turn.failed");
            }

            error.clear();
            runtime.host->run_turn(command.turn->request, result.turn_result, error);
            result.response_kind = common_agent_daemon_response_kind::turn;
            result.ok = result.turn_result.ok;
            if (!error.empty() && result.turn_result.error.empty()) {
                result.turn_result.error = error;
            }
            if (!result.turn_result.error.empty()) {
                result.error = result.turn_result.error;
            }
            result.event =
                result.turn_result.cancelled
                    ? "turn_cancelled"
                    : (result.turn_result.ok ? "turn_completed" : "turn_failed");
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
    return fail_lifecycle_result(
        command,
        result,
        error,
        {},
        "command.failed");
}
