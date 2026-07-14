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

namespace {

bool filter_session_descriptor(
        const common_agent_runtime_session_descriptor & session,
        const common_agent_runtime_session_key & key) {
    return session.key.namespace_id == key.namespace_id &&
        session.key.session_id == key.session_id;
}

std::optional<common_agent_runtime_session_descriptor> find_session_descriptor(
        common_agent_runtime_session_manager * host,
        const agent_resource_read_authority & authority) {
    if (host == nullptr) {
        return std::nullopt;
    }
    for (const auto & session : host->list_sessions()) {
        if (session.key.namespace_id == authority.namespace_id &&
                session.key.session_id == authority.session_id) {
            return session;
        }
    }
    return std::nullopt;
}

common_agent_daemon_memory_summary summarize_memory(
        const common_memory_record & memory) {
    return {
        memory.id,
        common_memory_kind_name(memory.kind),
        common_memory_scope_name(memory.scope),
        !memory.summary.empty() ? memory.summary : memory.content,
        memory.session_id,
        memory.project_id,
        memory.turn_id,
        memory.created_at,
    };
}

common_agent_daemon_plan_summary summarize_plan(
        const common_plan_state & plan) {
    const auto plan_status_name = [](common_plan_status status) {
        switch (status) {
            case common_plan_status::proposed: return "proposed";
            case common_plan_status::active: return "active";
            case common_plan_status::completed: return "completed";
            case common_plan_status::blocked: return "blocked";
            case common_plan_status::failed: return "failed";
            case common_plan_status::cancelled: return "cancelled";
        }
        return "proposed";
    };
    const auto plan_scope_name = [](common_plan_scope scope) {
        switch (scope) {
            case common_plan_scope::turn: return "turn";
            case common_plan_scope::session: return "session";
            case common_plan_scope::project: return "project";
            case common_plan_scope::global: return "global";
        }
        return "turn";
    };
    return {
        plan.id,
        plan.purpose,
        plan.goal,
        plan_status_name(plan.status),
        plan_scope_name(plan.scope),
        plan.session_id,
        plan.project_id,
        plan.turn_id,
        plan.active_step_id.value_or(""),
        plan.next_action.value_or(""),
        plan.version,
        plan.steps.size(),
        plan.observations.size(),
    };
}

}

std::optional<common_agent_runtime_active_turn_descriptor> common_agent_daemon_service::describe_active_turn() const {
    if (!runtime.host) {
        return std::nullopt;
    }
    return runtime.host->describe_active_turn();
}

std::vector<common_agent_runtime_session_descriptor> common_agent_daemon_service::list_sessions() const {
    if (!runtime.host) {
        return {};
    }
    return runtime.host->list_sessions();
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

        case common_agent_daemon_command_type::list_sessions:
            result.ok = runtime.host != nullptr;
            result.response_kind = common_agent_daemon_response_kind::status;
            result.event = "sessions_listed";
            result.status.state = state_value;
            result.status.live = state_value != common_agent_daemon_state::stopped;
            result.status.ready = state_value == common_agent_daemon_state::ready;
            result.status.session_snapshot_populated = true;
            if (runtime.host) {
                result.status.sessions = runtime.host->list_sessions();
                result.status.session_count = result.status.sessions.size();
            } else {
                result.error = "daemon host is not initialized";
            }
            append_daemon_event(
                result,
                "sessions.listed",
                command.request_id,
                {},
                std::to_string(result.status.session_count));
            error = result.error;
            return result.ok;

        case common_agent_daemon_command_type::get_session:
            if (!command.session.has_value()) {
                error = "get_session command missing session payload";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "session_lookup_failed",
                    "session.lookup_failed");
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "session_lookup_failed",
                    "session.lookup_failed");
            }
            result.response_kind = common_agent_daemon_response_kind::status;
            result.event = "session_found";
            result.status.state = state_value;
            result.status.live = state_value != common_agent_daemon_state::stopped;
            result.status.ready = state_value == common_agent_daemon_state::ready;
            result.status.session_snapshot_populated = true;
            for (const auto & session : runtime.host->list_sessions()) {
                if (filter_session_descriptor(session, command.session->key)) {
                    result.status.sessions.push_back(session);
                }
            }
            result.status.session_count = result.status.sessions.size();
            if (result.status.sessions.empty()) {
                error = "session is not active";
                result.error = error;
                result.event = "session_not_found";
                result.ok = false;
                append_daemon_event(
                    result,
                    "session.not_found",
                    command.request_id,
                    {},
                    error);
                return false;
            }
            result.ok = true;
            append_daemon_event(
                result,
                "session.found",
                command.request_id,
                {},
                result.status.sessions.front().key.session_id);
            error.clear();
            return true;

        case common_agent_daemon_command_type::list_resources:
            if (!command.scope.has_value()) {
                error = "list_resources command missing scope payload";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "resources_list_failed",
                    "resources.list_failed");
            }
            if (!runtime.resource_store) {
                error = "daemon resource store is not initialized";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "resources_list_failed",
                    "resources.list_failed");
            }
            result.response_kind = common_agent_daemon_response_kind::listing;
            result.event = "resources_listed";
            result.status.state = state_value;
            result.status.live = state_value != common_agent_daemon_state::stopped;
            result.status.ready = state_value == common_agent_daemon_state::ready;
            result.ok = runtime.resource_store->list(
                command.scope->authority,
                result.listing_result.resources,
                error);
            if (!result.ok) {
                result.error = error;
                append_daemon_event(
                    result,
                    "resources.list_failed",
                    command.request_id,
                    {},
                    error);
                return false;
            }
            append_daemon_event(
                result,
                "resources.listed",
                command.request_id,
                {},
                std::to_string(result.listing_result.resources.size()));
            error.clear();
            return true;

        case common_agent_daemon_command_type::list_memories:
            if (!command.scope.has_value()) {
                error = "list_memories command missing scope payload";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "memories_list_failed",
                    "memories.list_failed");
            }
            if (!runtime.memory_store) {
                error = "daemon memory store is not initialized";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "memories_list_failed",
                    "memories.list_failed");
            }
            result.response_kind = common_agent_daemon_response_kind::listing;
            result.event = "memories_listed";
            result.status.state = state_value;
            result.status.live = state_value != common_agent_daemon_state::stopped;
            result.status.ready = state_value == common_agent_daemon_state::ready;
            {
                std::vector<common_memory_record> collected;
                const auto session = find_session_descriptor(runtime.host.get(), command.scope->authority);

                common_memory_query query;
                query.namespace_id = command.scope->authority.namespace_id;
                query.session_id = command.scope->authority.session_id;
                query.project_id = command.scope->authority.project_id;
                query.turn_id = command.scope->authority.turn_id;
                query.limit = 1024;

                query.scope = common_memory_scope::session;
                auto session_memories = runtime.memory_store->list(query, error);
                if (!error.empty()) {
                    result.error = error;
                    append_daemon_event(result, "memories.list_failed", command.request_id, {}, error);
                    return false;
                }
                collected.insert(collected.end(), session_memories.begin(), session_memories.end());

                const std::string project_id =
                    !query.project_id.empty()
                        ? query.project_id
                        : (session.has_value() ? session->project_id : std::string());
                if (!project_id.empty()) {
                    query.scope = common_memory_scope::project;
                    query.project_id = project_id;
                    auto project_memories = runtime.memory_store->list(query, error);
                    if (!error.empty()) {
                        result.error = error;
                        append_daemon_event(result, "memories.list_failed", command.request_id, {}, error);
                        return false;
                    }
                    collected.insert(collected.end(), project_memories.begin(), project_memories.end());
                }

                for (const auto & memory : collected) {
                    result.listing_result.memories.push_back(summarize_memory(memory));
                }
                result.ok = true;
            }
            append_daemon_event(
                result,
                "memories.listed",
                command.request_id,
                {},
                std::to_string(result.listing_result.memories.size()));
            error.clear();
            return true;

        case common_agent_daemon_command_type::list_plans:
            if (!command.scope.has_value()) {
                error = "list_plans command missing scope payload";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "plans_list_failed",
                    "plans.list_failed");
            }
            if (!runtime.plan_store) {
                error = "daemon plan store is not initialized";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "plans_list_failed",
                    "plans.list_failed");
            }
            result.response_kind = common_agent_daemon_response_kind::listing;
            result.event = "plans_listed";
            result.status.state = state_value;
            result.status.live = state_value != common_agent_daemon_state::stopped;
            result.status.ready = state_value == common_agent_daemon_state::ready;
            {
                const auto session = find_session_descriptor(runtime.host.get(), command.scope->authority);
                const std::string project_id =
                    !command.scope->authority.project_id.empty()
                        ? command.scope->authority.project_id
                        : (session.has_value() ? session->project_id : std::string());
                auto plans = runtime.plan_store->list(error);
                if (!error.empty()) {
                    result.error = error;
                    append_daemon_event(result, "plans.list_failed", command.request_id, {}, error);
                    return false;
                }
                for (const auto & plan : plans) {
                    const bool matches_session = common_plan_scope_matches(
                        plan,
                        common_plan_scope::session,
                        command.scope->authority.namespace_id,
                        command.scope->authority.session_id,
                        project_id,
                        command.scope->authority.turn_id);
                    const bool matches_project = !project_id.empty() && common_plan_scope_matches(
                        plan,
                        common_plan_scope::project,
                        command.scope->authority.namespace_id,
                        command.scope->authority.session_id,
                        project_id,
                        command.scope->authority.turn_id);
                    if (!matches_session && !matches_project) {
                        continue;
                    }
                    result.listing_result.plans.push_back(summarize_plan(plan));
                }
                result.ok = true;
            }
            append_daemon_event(
                result,
                "plans.listed",
                command.request_id,
                {},
                std::to_string(result.listing_result.plans.size()));
            error.clear();
            return true;

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

        case common_agent_daemon_command_type::read_resource:
            if (!command.resource.has_value()) {
                error = "read_resource command missing resource payload";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "resource_read_failed",
                    "resource.read_failed");
            }
            if (!runtime.resource_store) {
                error = "daemon resource store is not initialized";
                return fail_lifecycle_result(
                    command,
                    result,
                    error,
                    "resource_read_failed",
                    "resource.read_failed");
            }
            result.response_kind = common_agent_daemon_response_kind::resource;
            if (!runtime.resource_store->stat(
                    command.resource->uri,
                    command.resource->authority,
                    result.resource_result.resource,
                    error)) {
                result.ok = false;
                result.event = "resource_not_found";
                result.error = error;
                append_daemon_event(
                    result,
                    "resource.not_found",
                    command.request_id,
                    {},
                    error);
                return false;
            }
            if (!runtime.resource_store->read_text(
                    command.resource->uri,
                    command.resource->authority,
                    command.resource->max_bytes,
                    result.resource_result.content,
                    error)) {
                result.ok = false;
                result.event = "resource_read_failed";
                result.error = error;
                append_daemon_event(
                    result,
                    "resource.read_failed",
                    command.request_id,
                    {},
                    error);
                return false;
            }
            result.ok = true;
            result.event = "resource_read";
            append_daemon_event(
                result,
                "resource.read",
                command.request_id,
                {},
                result.resource_result.resource.uri);
            error.clear();
            return true;

        case common_agent_daemon_command_type::drain:
            state_value = common_agent_daemon_state::draining;
            return succeed_lifecycle_result(
                command,
                result,
                error,
                "drain",
                "daemon.drain_requested",
                "drain requested");

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
