#include "agent-daemon-service.h"

namespace {

common_agent_event_context make_command_event_context(
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
    if (command.scope.has_value()) {
        context.namespace_id = command.scope->authority.namespace_id;
        context.project_id = command.scope->authority.project_id;
        context.session_id = command.scope->authority.session_id;
        context.turn_id = command.scope->authority.turn_id;
        return context;
    }
    if (command.resource.has_value()) {
        context.namespace_id = command.resource->authority.namespace_id;
        context.project_id = command.resource->authority.project_id;
        context.session_id = command.resource->authority.session_id;
        context.turn_id = command.resource->authority.turn_id;
        return context;
    }
    return context;
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

void emit_plan_and_resource_events_from_turn(
        common_agent_daemon_service & service,
        const common_agent_daemon_command & command,
        const common_agent_runtime_session_manager_turn_result & turn_result) {
    const auto command_context = make_command_event_context(command);
    std::set<std::string> emitted_keys;

    const auto emit_once = [&](
            common_agent_daemon_event_type type,
            const std::string & detail) {
        const std::string key =
            std::string(common_agent_daemon_event_type_name(type)) + "|" + detail;
        if (!emitted_keys.insert(key).second) {
            return;
        }
        auto context = command_context;
        service.emit_internal_event(make_common_agent_daemon_event(
            type,
            context.request_id,
            context.turn_id,
            detail,
            0,
            std::move(context)));
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
        common_agent_daemon_event event) {
    execution.events.push_back(std::move(event));
}

void emit_agent_daemon_execution_typed_event(
        common_agent_daemon_command_execution & execution,
        common_agent_daemon_event_type type,
        common_agent_event_context context,
        std::string detail) {
    execution.events.push_back(make_common_agent_daemon_event(
        type,
        std::move(context.request_id),
        std::move(context.turn_id),
        std::move(detail),
        0,
        std::move(context)));
}

common_agent_daemon_service::common_agent_daemon_service(
        common_agent_daemon_runtime runtime,
        std::unique_ptr<common_agent_daemon_event_collector> event_collector)
    : runtime(std::move(runtime))
    , event_collector(std::move(event_collector)) {
    if (!this->event_collector) {
        this->event_collector = std::make_unique<common_agent_daemon_event_collector>();
    }
    if (this->runtime.host) {
        this->runtime.host->set_event_sink([this](common_agent_daemon_event event) {
            emit_internal_event(std::move(event));
        });
    }
    state_value = (this->runtime.host || this->runtime.tool_executor)
        ? common_agent_daemon_state::ready
        : common_agent_daemon_state::failed;
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
        common_agent_daemon_event event) {
    if (!event_collector) {
        return;
    }
    event_collector->append(std::move(event));
}

std::vector<common_agent_daemon_event> common_agent_daemon_service::take_internal_events() {
    if (!event_collector) {
        return {};
    }
    return event_collector->take();
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
        common_agent_daemon_event_type event_type) const {
    initialize_lifecycle_outcome(command, outcome);
    outcome.ok = false;
    outcome.event = std::move(event);
    outcome.error = error;
    common_agent_daemon_command_execution execution{std::move(outcome), std::move(events)};
    auto context = make_command_event_context(command);
    context.turn_id.clear();
    emit_agent_daemon_execution_typed_event(
        execution,
        event_type,
        std::move(context),
        error);
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
        common_agent_daemon_event_type event_type) const {
    initialize_turn_outcome(command, outcome);
    outcome.ok = false;
    outcome.event = std::move(event);
    outcome.error = error;
    populate_daemon_failed_turn_result(command, outcome.turn_result, error);
    common_agent_daemon_command_execution execution{std::move(outcome), std::move(events)};
    emit_agent_daemon_execution_typed_event(
        execution,
        event_type,
        make_command_event_context(command),
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
        common_agent_daemon_event_type event_type,
        std::string detail) const {
    initialize_lifecycle_outcome(command, outcome);
    outcome.ok = true;
    outcome.event = std::move(event);
    common_agent_daemon_command_execution execution{std::move(outcome), std::move(events)};
    auto context = make_command_event_context(command);
    context.turn_id.clear();
    emit_agent_daemon_execution_typed_event(
        execution,
        event_type,
        std::move(context),
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
    emit_agent_daemon_execution_typed_event(
        execution,
        common_agent_daemon_event_type::status_reported,
        common_agent_event_context{{}, {}, {}, execution.outcome.request_id, {}, {}},
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
    const auto command_context = make_command_event_context(command);
    common_agent_event_emitter command_events(
        [&events](common_agent_daemon_event event) {
            events.push_back(std::move(event));
        },
        command_context);

    switch (command.type) {
        case common_agent_daemon_command_type::get_status:
            return populate_status_outcome(outcome, events, error);

        case common_agent_daemon_command_type::cancel_turn:
            error = "cancel_turn is handled by the daemon dispatcher";
            outcome.error = error;
            command_events.emit(common_agent_daemon_event_type::turn_cancel_rejected, error);
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
            command_events.emit(common_agent_daemon_event_type::sessions_listed, std::to_string(outcome.status.session_count));
            error = outcome.error;
            return outcome.ok;

        case common_agent_daemon_command_type::get_session:
            if (!command.session.has_value()) {
                error = "get_session command missing session payload";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "session_lookup_failed",
                    common_agent_daemon_event_type::session_lookup_failed);
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "session_lookup_failed",
                    common_agent_daemon_event_type::session_lookup_failed);
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
                command_events.emit(common_agent_daemon_event_type::session_not_found, error);
                return false;
            }
            outcome.ok = true;
            command_events.emit(common_agent_daemon_event_type::session_found, outcome.status.sessions.front().key.session_id);
            error.clear();
            return true;

        case common_agent_daemon_command_type::list_resources:
            if (!command.scope.has_value()) {
                error = "list_resources command missing scope payload";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "resources_list_failed",
                    common_agent_daemon_event_type::resources_list_failed);
            }
            if (!runtime.resource_store) {
                error = "daemon resource store is not initialized";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "resources_list_failed",
                    common_agent_daemon_event_type::resources_list_failed);
            }
            outcome.response_kind = common_agent_daemon_response_kind::listing;
            outcome.event = "resources_listed";
            outcome.status.state = state_value;
            outcome.status.live = state_value != common_agent_daemon_state::stopped;
            outcome.status.ready = state_value == common_agent_daemon_state::ready;
            outcome.ok = runtime.resource_store->list(command.scope->authority, outcome.listing_result.resources, error);
            if (!outcome.ok) {
                outcome.error = error;
                command_events.emit(common_agent_daemon_event_type::resources_list_failed, error);
                return false;
            }
            command_events.emit(common_agent_daemon_event_type::resources_listed, std::to_string(outcome.listing_result.resources.size()));
            error.clear();
            return true;

        case common_agent_daemon_command_type::list_memories:
            if (!command.scope.has_value()) {
                error = "list_memories command missing scope payload";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "memories_list_failed",
                    common_agent_daemon_event_type::memories_list_failed);
            }
            if (!runtime.memory_store) {
                error = "daemon memory store is not initialized";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "memories_list_failed",
                    common_agent_daemon_event_type::memories_list_failed);
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
                    command_events.emit(common_agent_daemon_event_type::memories_list_failed, error);
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
                        command_events.emit(common_agent_daemon_event_type::memories_list_failed, error);
                        return false;
                    }
                    collected.insert(collected.end(), project_memories.begin(), project_memories.end());
                }

                for (const auto & memory : collected) {
                    outcome.listing_result.memories.push_back(summarize_memory(memory));
                }
                outcome.ok = true;
            }
            command_events.emit(common_agent_daemon_event_type::memories_listed, std::to_string(outcome.listing_result.memories.size()));
            error.clear();
            return true;

        case common_agent_daemon_command_type::list_plans:
            if (!command.scope.has_value()) {
                error = "list_plans command missing scope payload";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "plans_list_failed",
                    common_agent_daemon_event_type::plans_list_failed);
            }
            if (!runtime.plan_store) {
                error = "daemon plan store is not initialized";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "plans_list_failed",
                    common_agent_daemon_event_type::plans_list_failed);
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
                    command_events.emit(common_agent_daemon_event_type::plans_list_failed, error);
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
            command_events.emit(common_agent_daemon_event_type::plans_listed, std::to_string(outcome.listing_result.plans.size()));
            error.clear();
            return true;

        case common_agent_daemon_command_type::reset_session:
            if (!command.session.has_value()) {
                error = "reset_session command missing session payload";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "session_reset_failed",
                    common_agent_daemon_event_type::session_reset_failed);
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "session_reset_failed",
                    common_agent_daemon_event_type::session_reset_failed);
            }
            outcome.ok = runtime.host->reset_session(command.session->key, error);
            if (!outcome.ok) {
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "session_reset_failed",
                    common_agent_daemon_event_type::session_reset_failed);
            }
            return succeed_lifecycle_result(
                command,
                outcome,
                events,
                error,
                "session_reset",
                common_agent_daemon_event_type::session_reset,
                "session reset");

        case common_agent_daemon_command_type::close_session:
            if (!command.session.has_value()) {
                error = "close_session command missing session payload";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "session_close_failed",
                    common_agent_daemon_event_type::session_close_failed);
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "session_close_failed",
                    common_agent_daemon_event_type::session_close_failed);
            }
            outcome.ok = runtime.host->close_session(command.session->key, error);
            if (!outcome.ok) {
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "session_close_failed",
                    common_agent_daemon_event_type::session_close_failed);
            }
            return succeed_lifecycle_result(
                command,
                outcome,
                events,
                error,
                "session_closed",
                common_agent_daemon_event_type::session_closed,
                "session closed");

        case common_agent_daemon_command_type::read_resource:
            if (!command.resource.has_value()) {
                error = "read_resource command missing resource payload";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "resource_read_failed",
                    common_agent_daemon_event_type::resource_read_failed);
            }
            if (!runtime.resource_store) {
                error = "daemon resource store is not initialized";
                return fail_lifecycle_result(
                    command,
                    outcome,
                    events,
                    error,
                    "resource_read_failed",
                    common_agent_daemon_event_type::resource_read_failed);
            }
            outcome.response_kind = common_agent_daemon_response_kind::resource;
            if (!runtime.resource_store->stat(command.resource->uri, command.resource->authority, outcome.resource_result.resource, error)) {
                outcome.ok = false;
                outcome.event = "resource_not_found";
                outcome.error = error;
                command_events.emit(common_agent_daemon_event_type::resource_not_found, error);
                return false;
            }
            if (!runtime.resource_store->read_text(command.resource->uri, command.resource->authority, command.resource->max_bytes, outcome.resource_result.content, error)) {
                outcome.ok = false;
                outcome.event = "resource_read_failed";
                outcome.error = error;
                command_events.emit(common_agent_daemon_event_type::resource_read_failed, error);
                return false;
            }
            outcome.ok = true;
            outcome.event = "resource_read";
            command_events.emit(common_agent_daemon_event_type::resource_read, outcome.resource_result.resource.uri);
            error.clear();
            return true;

        case common_agent_daemon_command_type::drain:
            state_value = common_agent_daemon_state::draining;
            return succeed_lifecycle_result(
                command,
                outcome,
                events,
                error,
                "drain",
                common_agent_daemon_event_type::drain_requested,
                "drain requested");

        case common_agent_daemon_command_type::shutdown:
            shutdown_requested_flag = true;
            shutdown_mode_value = common_agent_daemon_shutdown_mode::drain;
            state_value = common_agent_daemon_state::draining;
            return succeed_lifecycle_result(
                command,
                outcome,
                events,
                error,
                "shutdown",
                common_agent_daemon_event_type::shutdown_requested,
                "shutdown requested");

        case common_agent_daemon_command_type::reload_config:
            initialize_lifecycle_outcome(command, outcome);
            outcome.response_kind = common_agent_daemon_response_kind::lifecycle;
            command_events.emit(common_agent_daemon_event_type::config_reload_started, command.reload_path);
            if (!runtime.reload_config) {
                error = "config reload is not enabled for this daemon";
                outcome.ok = false;
                outcome.event = "config.reload.rejected";
                outcome.error = error;
                command_events.emit(common_agent_daemon_event_type::config_reload_rejected, error);
                return false;
            }
            if (!runtime.reload_config(command.reload_path, outcome.reload_result, error)) {
                outcome.ok = false;
                outcome.event = "config.reload.rejected";
                outcome.error = error;
                command_events.emit(
                    common_agent_daemon_event_type::config_reload_rejected,
                    outcome.reload_result.warning.empty() ? error : outcome.reload_result.warning);
                return false;
            }
            outcome.ok = outcome.reload_result.restart_required.empty();
            outcome.event = outcome.ok ? "config.reload.completed" : "config.reload.rejected";
            outcome.error = outcome.ok ? std::string() : "configuration change requires daemon restart";
            if (outcome.ok) {
                for (const auto & provider : outcome.reload_result.providers_added) {
                    command_events.emit(common_agent_daemon_event_type::mcp_provider_added, provider);
                }
                for (const auto & provider : outcome.reload_result.providers_removed) {
                    command_events.emit(common_agent_daemon_event_type::mcp_provider_removed, provider);
                }
                for (const auto & provider : outcome.reload_result.providers_replaced) {
                    command_events.emit(common_agent_daemon_event_type::mcp_provider_replaced, provider);
                }
            }
            command_events.emit(
                outcome.ok
                    ? common_agent_daemon_event_type::config_reload_completed
                    : common_agent_daemon_event_type::config_reload_rejected,
                outcome.ok ? std::to_string(outcome.reload_result.config_version) : outcome.error);
            error = outcome.error;
            return outcome.ok;

        case common_agent_daemon_command_type::execute_tool:
            if (!command.tool.has_value()) {
                error = "execute_tool command missing tool payload";
                outcome.error = error;
                outcome.event = "tool_failed";
                command_events.emit(common_agent_daemon_event_type::command_failed, error);
                return false;
            }
            if (shutdown_requested_flag || state_value != common_agent_daemon_state::ready) {
                error = "daemon is not accepting new tools";
                outcome.error = error;
                outcome.event = "tool_rejected";
                command_events.emit(common_agent_daemon_event_type::command_rejected, error);
                return false;
            }
            if (!runtime.tool_executor) {
                error = "daemon tool executor is not initialized";
                outcome.error = error;
                outcome.event = "tool_failed";
                command_events.emit(common_agent_daemon_event_type::command_failed, error);
                return false;
            }
            outcome.response_kind = common_agent_daemon_response_kind::tool;
            command_events.emit(common_agent_daemon_event_type::tool_started, command.tool->tool_name);
            outcome.ok = runtime.tool_executor(*command.tool, outcome.tool_result, error);
            outcome.event = outcome.ok ? "tool_completed" : "tool_failed";
            outcome.error = error.empty() ? outcome.tool_result.raw_diagnostic : error;
            command_events.emit(
                outcome.ok ? common_agent_daemon_event_type::tool_completed : common_agent_daemon_event_type::command_failed,
                outcome.tool_result.safe_summary);
            return outcome.ok;

        case common_agent_daemon_command_type::run_turn:
            if (!command.turn.has_value()) {
                error = "run_turn command missing turn payload";
                return fail_turn_result(
                    command,
                    outcome,
                    events,
                    error,
                    "turn_failed",
                    common_agent_daemon_event_type::turn_failed);
            }
            if (shutdown_requested_flag || state_value != common_agent_daemon_state::ready) {
                error = "daemon is not accepting new turns";
                return fail_turn_result(
                    command,
                    outcome,
                    events,
                    error,
                    "turn_rejected",
                    common_agent_daemon_event_type::turn_rejected);
            }
            if (!runtime.host) {
                error = "daemon host is not initialized";
                return fail_turn_result(
                    command,
                    outcome,
                    events,
                    error,
                    "turn_failed",
                    common_agent_daemon_event_type::turn_failed);
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
            const auto command_context = make_command_event_context(command);
            for (const auto & trace : outcome.turn_result.trace) {
                if (trace.stage == common_runtime_trace_stage::tool && !trace.tool_name.empty()) {
                    auto started_context = command_context;
                    emit_internal_event(make_common_agent_daemon_event(
                        common_agent_daemon_event_type::tool_started,
                        started_context.request_id,
                        started_context.turn_id,
                        trace.tool_name,
                        0,
                        std::move(started_context)));
                    auto completed_context = command_context;
                    emit_internal_event(make_common_agent_daemon_event(
                        common_agent_daemon_event_type::tool_completed,
                        completed_context.request_id,
                        completed_context.turn_id,
                        trace.tool_name + ":" + common_runtime_trace_kind_name(trace.kind),
                        0,
                        std::move(completed_context)));
                }
            }
            emit_plan_and_resource_events_from_turn(*this, command, outcome.turn_result);
            if (outcome.turn_result.memory_learning_related_count > 0 ||
                    (!outcome.turn_result.memory_learning_summary.empty() &&
                        outcome.turn_result.memory_learning_summary != "none")) {
                auto memory_context = command_context;
                emit_internal_event(make_common_agent_daemon_event(
                    common_agent_daemon_event_type::memory_learned,
                    memory_context.request_id,
                    memory_context.turn_id,
                    outcome.turn_result.memory_learning_summary,
                    0,
                    std::move(memory_context)));
            }
            command_events.emit(
                outcome.turn_result.cancelled
                    ? common_agent_daemon_event_type::turn_cancelled
                    : (outcome.turn_result.ok
                        ? common_agent_daemon_event_type::turn_completed
                        : common_agent_daemon_event_type::turn_failed),
                outcome.error);
            return outcome.turn_result.ok;
    }

    error = "unsupported daemon command";
    return fail_lifecycle_result(
        command,
        outcome,
        events,
        error,
        {},
        common_agent_daemon_event_type::command_failed);
}
