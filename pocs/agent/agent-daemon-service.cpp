#include "agent-daemon-service.h"

namespace {

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

void emit_plan_and_resource_events_from_turn(
        common_agent_daemon_service & service,
        const common_agent_daemon_command & command,
        const common_agent_runtime_session_manager_turn_result & turn_result) {
    const std::string turn_id = command.turn.has_value()
        ? command.turn->request.turn.turn_id
        : std::string();
    std::set<std::string> emitted_keys;

    const auto emit_once = [&](
            common_agent_daemon_event_type type,
            const std::string & detail) {
        const std::string key =
            std::string(common_agent_daemon_event_type_name(type)) + "|" + detail;
        if (!emitted_keys.insert(key).second) {
            return;
        }
        service.emit_internal_event(
            type,
            command.request_id,
            turn_id,
            detail);
    };

    for (const auto & event : turn_result.events) {
        switch (event.type) {
            case common_agent_event_type::plan_created:
                emit_once(
                    common_agent_daemon_event_type::plan_created,
                    event.plan_id.value_or(event.detail));
                break;
            case common_agent_event_type::plan_updated:
                emit_once(
                    common_agent_daemon_event_type::plan_updated,
                    event.plan_id.value_or(event.detail));
                break;
            case common_agent_event_type::observation_recorded:
                emit_once(
                    common_agent_daemon_event_type::observation_recorded,
                    !event.observation_id.empty() ? event.observation_id : event.detail);
                break;
            case common_agent_event_type::resource_created:
                emit_once(
                    common_agent_daemon_event_type::resource_created,
                    !event.resource_uri.empty() ? event.resource_uri : event.detail);
                break;
            case common_agent_event_type::resource_attached:
                emit_once(
                    common_agent_daemon_event_type::resource_attached,
                    !event.resource_uri.empty() ? event.resource_uri : event.detail);
                break;
            default:
                break;
        }
    }

    for (const auto & trace : turn_result.trace) {
        if (trace.stage == common_runtime_trace_stage::step) {
            if (trace.kind == common_runtime_trace_kind::started) {
                emit_once(
                    common_agent_daemon_event_type::plan_step_started,
                    !trace.step_id.empty() ? trace.step_id : trace.detail);
            } else if (trace.kind == common_runtime_trace_kind::completed) {
                emit_once(
                    common_agent_daemon_event_type::plan_step_completed,
                    !trace.step_id.empty() ? trace.step_id : trace.detail);
            }
        }
        if (trace.stage == common_runtime_trace_stage::observation &&
                trace.kind == common_runtime_trace_kind::recorded) {
            emit_once(
                common_agent_daemon_event_type::observation_recorded,
                !trace.observation_id.empty() ? trace.observation_id : trace.detail);
        }
    }
}

} // namespace

common_agent_daemon_command_result project_agent_daemon_command_execution(
        common_agent_daemon_command_execution execution) {
    common_agent_daemon_command_result result;
    static_cast<common_agent_daemon_command_outcome &>(result) = std::move(execution.outcome);
    result.daemon_event_count = execution.events.size();
    result.events = std::move(execution.events);
    return result;
}

void append_agent_daemon_execution_event(
        common_agent_daemon_command_execution & execution,
        std::string type,
        std::string request_id,
        std::string turn_id,
        std::string detail) {
    execution.events.push_back(common_agent_daemon_event{
        std::move(type),
        std::move(request_id),
        std::move(turn_id),
        std::move(detail),
    });
}

common_agent_daemon_service::common_agent_daemon_service(common_agent_daemon_runtime runtime)
    : runtime(std::move(runtime)) {
    if (this->runtime.host) {
        this->runtime.host->set_event_sink(
            [this](
                    common_agent_daemon_event_type type,
                    const std::string & request_id,
                    const std::string & turn_id,
                    const std::string & detail) {
                emit_internal_event(type, request_id, turn_id, detail);
            });
    }
    state_value = this->runtime.host ? common_agent_daemon_state::ready : common_agent_daemon_state::failed;
}

bool common_agent_daemon_service::execute(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error) {
    common_agent_daemon_command_execution execution;
    const bool ok = execute_outcome(command, execution.outcome, execution.events, error);
    result = project_agent_daemon_command_execution(std::move(execution));
    return ok;
}

void common_agent_daemon_service::emit_internal_event(
        common_agent_daemon_event_type type,
        const std::string & request_id,
        const std::string & turn_id,
        const std::string & detail) {
    std::lock_guard<std::mutex> lock(event_mutex);
    pending_events.push_back({
        common_agent_daemon_event_type_name(type),
        request_id,
        turn_id,
        detail,
        type,
        next_event_sequence++,
    });
}

std::vector<common_agent_daemon_event> common_agent_daemon_service::take_internal_events() {
    std::lock_guard<std::mutex> lock(event_mutex);
    auto out = std::move(pending_events);
    pending_events.clear();
    return out;
}

void common_agent_daemon_service::initialize_command_outcome(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome) const {
    auto target_request_id = std::move(outcome.target_request_id);
    auto target_turn_id = std::move(outcome.target_turn_id);
    outcome = {};
    outcome.request_id = command.request_id;
    outcome.target_request_id = std::move(target_request_id);
    outcome.target_turn_id = std::move(target_turn_id);
}

void common_agent_daemon_service::initialize_lifecycle_outcome(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome) const {
    initialize_command_outcome(command, outcome);
    outcome.response_kind = common_agent_daemon_response_kind::lifecycle;
}

void common_agent_daemon_service::initialize_turn_outcome(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome) const {
    initialize_command_outcome(command, outcome);
    outcome.response_kind = common_agent_daemon_response_kind::turn;
}

bool common_agent_daemon_service::fail_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome,
        std::vector<common_agent_daemon_event> & events,
        std::string & error,
        std::string event,
        std::string daemon_event_type) const {
    initialize_lifecycle_outcome(command, outcome);
    outcome.ok = false;
    outcome.event = std::move(event);
    outcome.error = error;
    common_agent_daemon_command_execution execution{std::move(outcome), std::move(events)};
    append_agent_daemon_execution_event(execution, std::move(daemon_event_type), command.request_id, {}, error);
    outcome = std::move(execution.outcome);
    events = std::move(execution.events);
    return false;
}

bool common_agent_daemon_service::fail_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome,
        std::vector<common_agent_daemon_event> & events,
        std::string & error,
        std::string event,
        std::string daemon_event_type) const {
    initialize_turn_outcome(command, outcome);
    outcome.ok = false;
    outcome.event = std::move(event);
    outcome.error = error;
    populate_daemon_failed_turn_result(command, outcome.turn_result, error);
    common_agent_daemon_command_execution execution{std::move(outcome), std::move(events)};
    append_agent_daemon_execution_event(
        execution,
        std::move(daemon_event_type),
        command.request_id,
        command_turn_id(command),
        error);
    outcome = std::move(execution.outcome);
    events = std::move(execution.events);
    return false;
}

bool common_agent_daemon_service::succeed_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome,
        std::vector<common_agent_daemon_event> & events,
        std::string & error,
        std::string event,
        std::string daemon_event_type,
        std::string detail) const {
    initialize_lifecycle_outcome(command, outcome);
    outcome.ok = true;
    outcome.event = std::move(event);
    common_agent_daemon_command_execution execution{std::move(outcome), std::move(events)};
    append_agent_daemon_execution_event(
        execution,
        std::move(daemon_event_type),
        command.request_id,
        {},
        std::move(detail));
    outcome = std::move(execution.outcome);
    events = std::move(execution.events);
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
    common_agent_daemon_command_execution execution;
    execution.outcome.request_id = result.request_id;
    const bool ok = populate_status_outcome(execution.outcome, execution.events, error);
    result = project_agent_daemon_command_execution(std::move(execution));
    return ok;
}

bool common_agent_daemon_service::populate_status_outcome(
        common_agent_daemon_command_outcome & outcome,
        std::vector<common_agent_daemon_event> & events,
        std::string & error) const {
    outcome.ok = runtime.host != nullptr;
    outcome.response_kind = common_agent_daemon_response_kind::status;
    outcome.event = "status";
    outcome.status.state = state_value;
    outcome.status.live = state_value != common_agent_daemon_state::stopped;
    outcome.status.ready = state_value == common_agent_daemon_state::ready;
    if (runtime.host) {
        outcome.status.sessions = runtime.host->list_sessions();
        outcome.status.session_count = outcome.status.sessions.size();
    }
    common_agent_daemon_command_execution execution{std::move(outcome), std::move(events)};
    append_agent_daemon_execution_event(
        execution,
        "status.reported",
        execution.outcome.request_id,
        {},
        common_agent_daemon_state_name(execution.outcome.status.state));
    outcome = std::move(execution.outcome);
    events = std::move(execution.events);
    error.clear();
    return outcome.ok;
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

bool common_agent_daemon_service::execute_outcome(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome,
        std::vector<common_agent_daemon_event> & events,
        std::string & error) {
    initialize_command_outcome(command, outcome);

    auto append_event = [&](
            std::string type,
            std::string request_id,
            std::string turn_id,
            std::string detail = {}) {
        events.push_back(common_agent_daemon_event{
            std::move(type),
            std::move(request_id),
            std::move(turn_id),
            std::move(detail),
        });
    };

    switch (command.type) {
        case common_agent_daemon_command_type::get_status:
            return populate_status_outcome(outcome, events, error);

        case common_agent_daemon_command_type::cancel_turn:
            error = "cancel_turn is handled by the daemon dispatcher";
            outcome.error = error;
            append_event("turn.cancel_rejected", command.request_id, {}, error);
            return false;

        case common_agent_daemon_command_type::list_sessions:
            outcome.ok = runtime.host != nullptr;
            outcome.response_kind = common_agent_daemon_response_kind::status;
            outcome.event = "sessions_listed";
            outcome.status.state = state_value;
            outcome.status.live = state_value != common_agent_daemon_state::stopped;
            outcome.status.ready = state_value == common_agent_daemon_state::ready;
            outcome.status.session_snapshot_populated = true;
            if (runtime.host) {
                outcome.status.sessions = runtime.host->list_sessions();
                outcome.status.session_count = outcome.status.sessions.size();
            } else {
                outcome.error = "daemon host is not initialized";
            }
            append_event("sessions.listed", command.request_id, {}, std::to_string(outcome.status.session_count));
            error = outcome.error;
            return outcome.ok;

        case common_agent_daemon_command_type::get_session:
            if (!command.session.has_value()) {
                error = "get_session command missing session payload";
                return fail_lifecycle_result(command, outcome, events, error, "session_lookup_failed", "session.lookup_failed");
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_lifecycle_result(command, outcome, events, error, "session_lookup_failed", "session.lookup_failed");
            }
            outcome.response_kind = common_agent_daemon_response_kind::status;
            outcome.event = "session_found";
            outcome.status.state = state_value;
            outcome.status.live = state_value != common_agent_daemon_state::stopped;
            outcome.status.ready = state_value == common_agent_daemon_state::ready;
            outcome.status.session_snapshot_populated = true;
            for (const auto & session : runtime.host->list_sessions()) {
                if (filter_session_descriptor(session, command.session->key)) {
                    outcome.status.sessions.push_back(session);
                }
            }
            outcome.status.session_count = outcome.status.sessions.size();
            if (outcome.status.sessions.empty()) {
                error = "session is not active";
                outcome.error = error;
                outcome.event = "session_not_found";
                outcome.ok = false;
                append_event("session.not_found", command.request_id, {}, error);
                return false;
            }
            outcome.ok = true;
            append_event("session.found", command.request_id, {}, outcome.status.sessions.front().key.session_id);
            error.clear();
            return true;

        case common_agent_daemon_command_type::list_resources:
            if (!command.scope.has_value()) {
                error = "list_resources command missing scope payload";
                return fail_lifecycle_result(command, outcome, events, error, "resources_list_failed", "resources.list_failed");
            }
            if (!runtime.resource_store) {
                error = "daemon resource store is not initialized";
                return fail_lifecycle_result(command, outcome, events, error, "resources_list_failed", "resources.list_failed");
            }
            outcome.response_kind = common_agent_daemon_response_kind::listing;
            outcome.event = "resources_listed";
            outcome.status.state = state_value;
            outcome.status.live = state_value != common_agent_daemon_state::stopped;
            outcome.status.ready = state_value == common_agent_daemon_state::ready;
            outcome.ok = runtime.resource_store->list(command.scope->authority, outcome.listing_result.resources, error);
            if (!outcome.ok) {
                outcome.error = error;
                append_event("resources.list_failed", command.request_id, {}, error);
                return false;
            }
            append_event("resources.listed", command.request_id, {}, std::to_string(outcome.listing_result.resources.size()));
            emit_internal_event(common_agent_daemon_event_type::resources_listed, command.request_id, {}, std::to_string(outcome.listing_result.resources.size()));
            error.clear();
            return true;

        case common_agent_daemon_command_type::list_memories:
            if (!command.scope.has_value()) {
                error = "list_memories command missing scope payload";
                return fail_lifecycle_result(command, outcome, events, error, "memories_list_failed", "memories.list_failed");
            }
            if (!runtime.memory_store) {
                error = "daemon memory store is not initialized";
                return fail_lifecycle_result(command, outcome, events, error, "memories_list_failed", "memories.list_failed");
            }
            outcome.response_kind = common_agent_daemon_response_kind::listing;
            outcome.event = "memories_listed";
            outcome.status.state = state_value;
            outcome.status.live = state_value != common_agent_daemon_state::stopped;
            outcome.status.ready = state_value == common_agent_daemon_state::ready;
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
                    outcome.error = error;
                    append_event("memories.list_failed", command.request_id, {}, error);
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
                        outcome.error = error;
                        append_event("memories.list_failed", command.request_id, {}, error);
                        return false;
                    }
                    collected.insert(collected.end(), project_memories.begin(), project_memories.end());
                }

                for (const auto & memory : collected) {
                    outcome.listing_result.memories.push_back(summarize_memory(memory));
                }
                outcome.ok = true;
            }
            append_event("memories.listed", command.request_id, {}, std::to_string(outcome.listing_result.memories.size()));
            emit_internal_event(common_agent_daemon_event_type::memories_listed, command.request_id, {}, std::to_string(outcome.listing_result.memories.size()));
            error.clear();
            return true;

        case common_agent_daemon_command_type::list_plans:
            if (!command.scope.has_value()) {
                error = "list_plans command missing scope payload";
                return fail_lifecycle_result(command, outcome, events, error, "plans_list_failed", "plans.list_failed");
            }
            if (!runtime.plan_store) {
                error = "daemon plan store is not initialized";
                return fail_lifecycle_result(command, outcome, events, error, "plans_list_failed", "plans.list_failed");
            }
            outcome.response_kind = common_agent_daemon_response_kind::listing;
            outcome.event = "plans_listed";
            outcome.status.state = state_value;
            outcome.status.live = state_value != common_agent_daemon_state::stopped;
            outcome.status.ready = state_value == common_agent_daemon_state::ready;
            {
                const auto session = find_session_descriptor(runtime.host.get(), command.scope->authority);
                const std::string project_id =
                    !command.scope->authority.project_id.empty()
                        ? command.scope->authority.project_id
                        : (session.has_value() ? session->project_id : std::string());
                auto plans = runtime.plan_store->list(error);
                if (!error.empty()) {
                    outcome.error = error;
                    append_event("plans.list_failed", command.request_id, {}, error);
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
                    outcome.listing_result.plans.push_back(summarize_plan(plan));
                }
                outcome.ok = true;
            }
            append_event("plans.listed", command.request_id, {}, std::to_string(outcome.listing_result.plans.size()));
            emit_internal_event(common_agent_daemon_event_type::plans_listed, command.request_id, {}, std::to_string(outcome.listing_result.plans.size()));
            error.clear();
            return true;

        case common_agent_daemon_command_type::reset_session:
            if (!command.session.has_value()) {
                error = "reset_session command missing session payload";
                return fail_lifecycle_result(command, outcome, events, error, "session_reset_failed", "session.reset_failed");
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_lifecycle_result(command, outcome, events, error, "session_reset_failed", "session.reset_failed");
            }
            outcome.ok = runtime.host->reset_session(command.session->key, error);
            if (!outcome.ok) {
                return fail_lifecycle_result(command, outcome, events, error, "session_reset_failed", "session.reset_failed");
            }
            return succeed_lifecycle_result(command, outcome, events, error, "session_reset", "session.reset", "session reset");

        case common_agent_daemon_command_type::close_session:
            if (!command.session.has_value()) {
                error = "close_session command missing session payload";
                return fail_lifecycle_result(command, outcome, events, error, "session_close_failed", "session.close_failed");
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_lifecycle_result(command, outcome, events, error, "session_close_failed", "session.close_failed");
            }
            outcome.ok = runtime.host->close_session(command.session->key, error);
            if (!outcome.ok) {
                return fail_lifecycle_result(command, outcome, events, error, "session_close_failed", "session.close_failed");
            }
            return succeed_lifecycle_result(command, outcome, events, error, "session_closed", "session.closed", "session closed");

        case common_agent_daemon_command_type::read_resource:
            if (!command.resource.has_value()) {
                error = "read_resource command missing resource payload";
                return fail_lifecycle_result(command, outcome, events, error, "resource_read_failed", "resource.read_failed");
            }
            if (!runtime.resource_store) {
                error = "daemon resource store is not initialized";
                return fail_lifecycle_result(command, outcome, events, error, "resource_read_failed", "resource.read_failed");
            }
            outcome.response_kind = common_agent_daemon_response_kind::resource;
            if (!runtime.resource_store->stat(command.resource->uri, command.resource->authority, outcome.resource_result.resource, error)) {
                outcome.ok = false;
                outcome.event = "resource_not_found";
                outcome.error = error;
                append_event("resource.not_found", command.request_id, {}, error);
                return false;
            }
            if (!runtime.resource_store->read_text(command.resource->uri, command.resource->authority, command.resource->max_bytes, outcome.resource_result.content, error)) {
                outcome.ok = false;
                outcome.event = "resource_read_failed";
                outcome.error = error;
                append_event("resource.read_failed", command.request_id, {}, error);
                return false;
            }
            outcome.ok = true;
            outcome.event = "resource_read";
            append_event("resource.read", command.request_id, {}, outcome.resource_result.resource.uri);
            emit_internal_event(common_agent_daemon_event_type::resource_read, command.request_id, {}, outcome.resource_result.resource.uri);
            error.clear();
            return true;

        case common_agent_daemon_command_type::drain:
            state_value = common_agent_daemon_state::draining;
            emit_internal_event(common_agent_daemon_event_type::drain_requested, command.request_id, {}, "drain requested");
            return succeed_lifecycle_result(command, outcome, events, error, "drain", "daemon.drain_requested", "drain requested");

        case common_agent_daemon_command_type::shutdown:
            shutdown_requested_flag = true;
            shutdown_mode_value = common_agent_daemon_shutdown_mode::drain;
            state_value = common_agent_daemon_state::draining;
            emit_internal_event(common_agent_daemon_event_type::shutdown_requested, command.request_id, {}, "shutdown requested");
            return succeed_lifecycle_result(command, outcome, events, error, "shutdown", "daemon.shutdown_requested", "shutdown requested");

        case common_agent_daemon_command_type::run_turn:
            if (!command.turn.has_value()) {
                error = "run_turn command missing turn payload";
                return fail_turn_result(command, outcome, events, error, "turn_failed", "turn.failed");
            }
            if (shutdown_requested_flag || state_value != common_agent_daemon_state::ready) {
                error = "daemon is not accepting new turns";
                return fail_turn_result(command, outcome, events, error, "turn_rejected", "turn.rejected");
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_turn_result(command, outcome, events, error, "turn_failed", "turn.failed");
            }

            error.clear();
            runtime.host->run_turn(command.turn->request, outcome.turn_result, error);
            outcome.response_kind = common_agent_daemon_response_kind::turn;
            outcome.ok = outcome.turn_result.ok;
            if (!error.empty() && outcome.turn_result.error.empty()) {
                outcome.turn_result.error = error;
            }
            if (!outcome.turn_result.error.empty()) {
                outcome.error = outcome.turn_result.error;
            }
            outcome.event =
                outcome.turn_result.cancelled
                    ? "turn_cancelled"
                    : (outcome.turn_result.ok ? "turn_completed" : "turn_failed");
            for (const auto & trace : outcome.turn_result.trace) {
                if (trace.stage == common_runtime_trace_stage::tool && !trace.tool_name.empty()) {
                    emit_internal_event(common_agent_daemon_event_type::tool_started, command.request_id, command_turn_id(command), trace.tool_name);
                    emit_internal_event(common_agent_daemon_event_type::tool_completed, command.request_id, command_turn_id(command), trace.tool_name + ":" + common_runtime_trace_kind_name(trace.kind));
                }
            }
            emit_plan_and_resource_events_from_turn(*this, command, outcome.turn_result);
            if (outcome.turn_result.memory_learning_related_count > 0 ||
                    (!outcome.turn_result.memory_learning_summary.empty() &&
                        outcome.turn_result.memory_learning_summary != "none")) {
                emit_internal_event(common_agent_daemon_event_type::memory_learned, command.request_id, command_turn_id(command), outcome.turn_result.memory_learning_summary);
            }
            append_event(
                outcome.turn_result.cancelled
                    ? "turn.cancelled"
                    : (outcome.turn_result.ok ? "turn.completed" : "turn.failed"),
                command.request_id,
                command_turn_id(command),
                outcome.error);
            return outcome.turn_result.ok;
    }

    error = "unsupported daemon command";
    return fail_lifecycle_result(command, outcome, events, error, {}, "command.failed");
}
