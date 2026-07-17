#include "agent-daemon-adapter.h"

#include "agent/tool-result-contracts.h"
#include "../cli/agent-cli-selection.h"

#include <limits>

using json = nlohmann::ordered_json;

namespace {

constexpr int agent_daemon_protocol_version = 1;

template<typename T>
bool read_optional_timeout_ms(
        const json & parsed,
        const char * field,
        T & value,
        std::string & error) {
    if (!parsed.contains(field)) {
        return true;
    }
    const auto & timeout_value = parsed.at(field);
    if (!timeout_value.is_number_integer()) {
        error = std::string(field) + " must be an integer";
        return false;
    }
    const auto parsed_value = timeout_value.get<long long>();
    if (parsed_value < 0) {
        error = std::string(field) + " must be non-negative";
        return false;
    }
    if (static_cast<unsigned long long>(parsed_value) >
            static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
        error = std::string(field) + " is too large";
        return false;
    }
    value = static_cast<T>(parsed_value);
    return true;
}

common_agent_runtime_timeout_policy make_daemon_timeout_policy(
        const daemon_options & options) {
    return {
        options.turn_timeout_ms > 0
            ? options.turn_timeout_ms
            : (options.max_turn_seconds > 0 ? options.max_turn_seconds * 1000 : size_t(0)),
        options.inference_step_timeout_ms,
        options.tool_timeout_ms,
        options.mcp_connect_timeout_ms,
        options.mcp_request_timeout_ms,
        options.mcp_shutdown_timeout_ms,
    };
}

std::string default_plan_scope_for_memory_scope(common_memory_scope memory_scope) {
    switch (memory_scope) {
        case common_memory_scope::turn:    return "turn";
        case common_memory_scope::session: return "session";
        case common_memory_scope::project: return "project";
        case common_memory_scope::global:  return "global";
    }
    return "session";
}

const char * daemon_plan_scope_name(common_plan_scope scope) {
    switch (scope) {
        case common_plan_scope::turn:    return "turn";
        case common_plan_scope::session: return "session";
        case common_plan_scope::project: return "project";
        case common_plan_scope::global:  return "global";
    }
    return "turn";
}

bool parse_agent_daemon_command_name(
        const json & parsed,
        common_agent_daemon_command & command,
        std::string & error) {
    const std::string command_name = parsed.value("command", "");
    if (command_name == "shutdown") {
        command.type = common_agent_daemon_command_type::shutdown;
        error.clear();
        return true;
    }
    if (command_name == "status") {
        command.type = common_agent_daemon_command_type::get_status;
        error.clear();
        return true;
    }
    if (command_name == "list_sessions") {
        command.type = common_agent_daemon_command_type::list_sessions;
        error.clear();
        return true;
    }
    if (command_name == "get_session") {
        command.type = common_agent_daemon_command_type::get_session;
        command.session = common_agent_daemon_session_payload{
            {
                parsed.value("namespace_id", "default-namespace"),
                parsed.value("session_id", "default-session"),
            }
        };
        error.clear();
        return true;
    }
    if (command_name == "list_resources" ||
            command_name == "list_memories" ||
            command_name == "list_plans") {
        if (command_name == "list_resources") {
            command.type = common_agent_daemon_command_type::list_resources;
        } else if (command_name == "list_memories") {
            command.type = common_agent_daemon_command_type::list_memories;
        } else {
            command.type = common_agent_daemon_command_type::list_plans;
        }
        command.scope = common_agent_daemon_scope_payload{};
        command.scope->authority.namespace_id = parsed.value("namespace_id", "default-namespace");
        command.scope->authority.session_id = parsed.value("session_id", "default-session");
        command.scope->authority.project_id = parsed.value("project_id", "");
        command.scope->authority.turn_id = parsed.value("turn_id", "");
        error.clear();
        return true;
    }
    if (command_name == "cancel_turn") {
        command.type = common_agent_daemon_command_type::cancel_turn;
        command.cancel = common_agent_daemon_cancel_payload{
            parsed.value("target_request_id", ""),
            parsed.value("target_turn_id", ""),
        };
        if (command.cancel->target_request_id.empty() && command.cancel->target_turn_id.empty()) {
            error = "cancel_turn requires target_request_id or target_turn_id";
            return false;
        }
        error.clear();
        return true;
    }
    if (command_name == "reset_session" || command_name == "close_session") {
        command.type = command_name == "reset_session"
            ? common_agent_daemon_command_type::reset_session
            : common_agent_daemon_command_type::close_session;
        command.session = common_agent_daemon_session_payload{
            {
                parsed.value("namespace_id", "default-namespace"),
                parsed.value("session_id", "default-session"),
            }
        };
        error.clear();
        return true;
    }
    if (command_name == "read_resource") {
        command.type = common_agent_daemon_command_type::read_resource;
        command.resource = common_agent_daemon_resource_payload{};
        command.resource->uri = parsed.value("uri", "");
        command.resource->authority.namespace_id = parsed.value("namespace_id", "default-namespace");
        command.resource->authority.session_id = parsed.value("session_id", "default-session");
        command.resource->authority.project_id = parsed.value("project_id", "");
        command.resource->authority.turn_id = parsed.value("turn_id", "");
        const auto max_bytes = parsed.value("max_bytes", 8192);
        if (max_bytes <= 0) {
            error = "read_resource max_bytes must be positive";
            return false;
        }
        command.resource->max_bytes = static_cast<size_t>(max_bytes);
        if (command.resource->uri.empty()) {
            error = "read_resource requires uri";
            return false;
        }
        error.clear();
        return true;
    }
    if (command_name == "drain") {
        command.type = common_agent_daemon_command_type::drain;
        error.clear();
        return true;
    }
    if (!command_name.empty() && command_name != "run_turn") {
        error = "unsupported command: " + command_name;
        return false;
    }
    command.type = common_agent_daemon_command_type::run_turn;
    error.clear();
    return true;
}

bool parse_agent_daemon_turn_request(
        const json & parsed,
        const daemon_options & options,
        common_agent_runtime_host_mode default_mode,
        common_agent_runtime_session_host_turn_request & request,
        std::string & error) {
    request = {};
    request.prompt = parsed.value("prompt", "");
    request.session_id = parsed.value("session_id", "default-session");
    request.namespace_id = parsed.value("namespace_id", "default-namespace");
    request.project_id = parsed.value("project_id", "");
    request.turn_id = parsed.value("turn_id", "");
    request.n_predict = parsed.value("n_predict", 0);
    request.mode = default_mode;
    request.memory_scope = common_memory_scope::session;
    request.plan_scope = common_plan_scope::turn;
    auto timeout_policy = make_daemon_timeout_policy(options);
    if (!read_optional_timeout_ms(parsed, "turn_timeout_ms", timeout_policy.turn_timeout_ms, error) ||
            !read_optional_timeout_ms(parsed, "inference_step_timeout_ms", timeout_policy.inference_step_timeout_ms, error) ||
            !read_optional_timeout_ms(parsed, "tool_timeout_ms", timeout_policy.tool_timeout_ms, error) ||
            !read_optional_timeout_ms(parsed, "mcp_connect_timeout_ms", timeout_policy.mcp_connect_timeout_ms, error) ||
            !read_optional_timeout_ms(parsed, "mcp_request_timeout_ms", timeout_policy.mcp_request_timeout_ms, error) ||
            !read_optional_timeout_ms(parsed, "mcp_shutdown_timeout_ms", timeout_policy.mcp_shutdown_timeout_ms, error)) {
        return false;
    }
    request.execution_control = make_common_agent_runtime_execution_control(timeout_policy);

    const std::string mode_value = parsed.value("mode", options.default_mode);
    if (!parse_mode(mode_value, request.mode)) {
        error = "unsupported mode: " + mode_value;
        return false;
    }

    const std::string memory_scope_value = parsed.value("memory_scope", "session");
    if (!common_memory_scope_parse(memory_scope_value, request.memory_scope)) {
        error = "unsupported memory_scope: " + memory_scope_value;
        return false;
    }

    const std::string plan_scope_value = parsed.value(
        "plan_scope",
        default_plan_scope_for_memory_scope(request.memory_scope));
    if (!parse_plan_scope(plan_scope_value, request.plan_scope)) {
        error = "unsupported plan_scope: " + plan_scope_value;
        return false;
    }

    error.clear();
    return true;
}

json serialize_agent_daemon_event(
        const common_agent_daemon_event & event) {
    json event_json = {
        {"type", event.type},
    };
    if (event.sequence != 0) {
        event_json["sequence"] = event.sequence;
    }
    if (event.event_type != common_agent_daemon_event_type::unknown) {
        event_json["event_type"] = common_agent_daemon_event_type_name(event.event_type);
    }
    if (!event.request_id.empty()) {
        event_json["request_id"] = event.request_id;
    }
    if (!event.turn_id.empty()) {
        event_json["turn_id"] = event.turn_id;
    }
    if (!event.namespace_id.empty()) {
        event_json["namespace_id"] = event.namespace_id;
    }
    if (!event.project_id.empty()) {
        event_json["project_id"] = event.project_id;
    }
    if (!event.session_id.empty()) {
        event_json["session_id"] = event.session_id;
    }
    if (!event.operation_id.empty()) {
        event_json["operation_id"] = event.operation_id;
    }
    if (!event.detail.empty()) {
        event_json["detail"] = event.detail;
    }
    return event_json;
}

json serialize_agent_daemon_session_status(
        const common_agent_runtime_session_descriptor & session) {
    json session_json = {
        {"namespace_id", session.key.namespace_id},
        {"session_id", session.key.session_id},
        {"project_id", session.project_id},
        {"memory_scope", common_memory_scope_name(session.memory_scope)},
        {"plan_scope", daemon_plan_scope_name(session.plan_scope)},
        {"lane_state", session.lane_state},
        {"queued_turn_count", session.queued_turn_count},
    };
    if (!session.policy_pack_id.empty()) {
        session_json["policy_pack_id"] = session.policy_pack_id;
    }
    if (session.has_active_turn) {
        session_json["active_request_id"] = session.active_request_id;
        session_json["active_turn_id"] = session.active_turn_id;
        session_json["active_turn_phase"] = session.active_turn_phase;
        session_json["active_turn_disposition"] = session.active_turn_disposition;
        if (!session.pending_operation_kind.empty()) {
            session_json["pending_operation_kind"] = session.pending_operation_kind;
        }
        if (!session.pending_operation_detail.empty()) {
            session_json["pending_operation_detail"] = session.pending_operation_detail;
        }
        if (session.active_cancel_requested) {
            session_json["active_cancel_requested"] = true;
        }
    }
    if (!session.last_turn_id.empty()) {
        session_json["last_turn_id"] = session.last_turn_id;
        session_json["last_turn_phase"] = session.last_turn_phase;
        session_json["last_turn_disposition"] = session.last_turn_disposition;
    }
    return session_json;
}

json serialize_agent_daemon_trace_entry(
        const common_runtime_trace_entry & entry) {
    json entry_json = {
        {"stage", common_runtime_trace_stage_name(entry.stage)},
        {"kind", common_runtime_trace_kind_name(entry.kind)},
    };
    if (!entry.detail.empty()) {
        entry_json["detail"] = entry.detail;
    }
    if (!entry.plan_id.empty()) {
        entry_json["plan_id"] = entry.plan_id;
    }
    if (!entry.step_id.empty()) {
        entry_json["step_id"] = entry.step_id;
    }
    if (!entry.tool_name.empty()) {
        entry_json["tool_name"] = entry.tool_name;
    }
    if (!entry.observation_id.empty()) {
        entry_json["observation_id"] = entry.observation_id;
    }
    if (!entry.related_id.empty()) {
        entry_json["related_id"] = entry.related_id;
    }
    return entry_json;
}

json make_agent_daemon_base_response(
        const common_agent_daemon_command_result & result) {
    json response = {
        {"ok", result.ok},
        {"daemon_event_count", result.daemon_event_count},
    };
    if (!result.request_id.empty()) {
        response["request_id"] = result.request_id;
    }
    if (!result.event.empty()) {
        response["event"] = result.event;
    }

    json daemon_events = json::array();
    for (const auto & event : result.events) {
        daemon_events.push_back(serialize_agent_daemon_event(event));
    }
    response["events"] = std::move(daemon_events);
    return response;
}

void append_agent_daemon_status_snapshot(
        json & response,
        const common_agent_daemon_status & status,
        bool include_sessions) {
    response["state"] = common_agent_daemon_state_name(status.state);
    response["live"] = status.live;
    response["ready"] = status.ready;
    response["worker_running"] = status.worker_running;
    response["worker_count"] = status.worker_count;
    response["workers_running"] = status.workers_running;
    response["accepting_commands"] = status.accepting_commands;
    response["shutdown_requested"] = status.shutdown_requested;
    response["sessions"] = status.session_count;
    response["queued_commands"] = status.queued_command_count;
    response["max_queue_size"] = status.max_queue_size;
    response["queue_capacity_remaining"] = status.queue_capacity_remaining;
    const auto * active_turn = status.active_turn.has_value()
        ? &*status.active_turn
        : nullptr;
    if (active_turn != nullptr) {
        response["active_request_id"] = active_turn->request_id;
        response["active_turn_id"] = active_turn->turn_id;
        response["active_turn_phase"] = active_turn->phase;
        response["active_turn_disposition"] = active_turn->disposition;
        if (!active_turn->pending_operation_kind.empty()) {
            response["active_pending_operation_kind"] = active_turn->pending_operation_kind;
        }
        if (!active_turn->pending_operation_detail.empty()) {
            response["active_pending_operation_detail"] = active_turn->pending_operation_detail;
        }
    } else if (!status.active_request_id.empty()) {
        response["active_request_id"] = status.active_request_id;
        response["active_turn_id"] = status.active_turn_id;
        response["active_turn_phase"] = status.active_turn_phase;
        response["active_turn_disposition"] = status.active_turn_disposition;
        if (!status.active_pending_operation_kind.empty()) {
            response["active_pending_operation_kind"] = status.active_pending_operation_kind;
        }
        if (!status.active_pending_operation_detail.empty()) {
            response["active_pending_operation_detail"] = status.active_pending_operation_detail;
        }
    }
    if ((active_turn != nullptr && active_turn->cancellation_requested) ||
            status.active_cancel_requested) {
        response["active_cancel_requested"] = true;
    }
    if (!include_sessions) {
        return;
    }
    json session_array = json::array();
    for (const auto & session : status.sessions) {
        session_array.push_back(serialize_agent_daemon_session_status(session));
    }
    response["session_keys"] = std::move(session_array);
}

json make_agent_daemon_status_response(
        const common_agent_daemon_command_result & result) {
    json response = make_agent_daemon_base_response(result);
    append_agent_daemon_status_snapshot(response, result.status, true);
    if (!result.error.empty()) {
        response["error"] = result.error;
    }
    return response;
}

json make_agent_daemon_lifecycle_response(
        const common_agent_daemon_command_result & result) {
    json response = make_agent_daemon_base_response(result);
    append_agent_daemon_status_snapshot(
        response,
        result.status,
        result.status.session_snapshot_populated);
    if (!result.target_request_id.empty()) {
        response["target_request_id"] = result.target_request_id;
    }
    if (!result.target_turn_id.empty()) {
        response["target_turn_id"] = result.target_turn_id;
    }
    if (!result.error.empty()) {
        response["error"] = result.error;
    }
    return response;
}

json make_agent_daemon_resource_response(
        const common_agent_daemon_command_result & result) {
    json response = make_agent_daemon_base_response(result);
    append_agent_daemon_status_snapshot(response, result.status, false);
    response["resource"] =
        common_tool_resource_descriptor_to_json(result.resource_result.resource);
    response["content"] = result.resource_result.content;
    if (!result.error.empty()) {
        response["error"] = result.error;
    }
    return response;
}

json make_agent_daemon_listing_response(
        const common_agent_daemon_command_result & result) {
    json response = make_agent_daemon_base_response(result);
    append_agent_daemon_status_snapshot(response, result.status, false);
    if (!result.listing_result.resources.empty()) {
        json resources = json::array();
        for (const auto & resource : result.listing_result.resources) {
            resources.push_back(common_tool_resource_descriptor_to_json(resource));
        }
        response["resources"] = std::move(resources);
    }
    if (!result.listing_result.memories.empty()) {
        json memories = json::array();
        for (const auto & memory : result.listing_result.memories) {
            memories.push_back({
                {"id", memory.id},
                {"kind", memory.kind},
                {"scope", memory.scope},
                {"summary", memory.summary},
                {"session_id", memory.session_id},
                {"project_id", memory.project_id},
                {"turn_id", memory.turn_id},
                {"created_at", memory.created_at},
            });
        }
        response["memories"] = std::move(memories);
    }
    if (!result.listing_result.plans.empty()) {
        json plans = json::array();
        for (const auto & plan : result.listing_result.plans) {
            plans.push_back({
                {"plan_id", plan.plan_id},
                {"purpose", plan.purpose},
                {"goal", plan.goal},
                {"status", plan.status},
                {"scope", plan.scope},
                {"session_id", plan.session_id},
                {"project_id", plan.project_id},
                {"turn_id", plan.turn_id},
                {"active_step_id", plan.active_step_id},
                {"next_action", plan.next_action},
                {"version", plan.version},
                {"step_count", plan.step_count},
                {"observation_count", plan.observation_count},
            });
        }
        response["plans"] = std::move(plans);
    }
    if (!result.error.empty()) {
        response["error"] = result.error;
    }
    return response;
}

json make_agent_daemon_turn_response(
        const common_agent_daemon_command_result & result) {
    json response = make_agent_daemon_base_response(result);
    append_agent_daemon_status_snapshot(response, result.status, false);
    const auto & turn = result.turn_result;
    response["cancelled"] = turn.cancelled;
    response["runtime_reused"] = turn.runtime_reused;
    response["limit_reached"] = turn.limit_reached;
    response["reflected"] = turn.reflected;
    response["revised"] = turn.revised;
    response["failure_class"] = common_agent_failure_class_name(turn.failure_class);
    response["response"] = turn.response;
    response["response_generation_status"] = common_agent_generation_status_name(turn.response_generation_status);
    response["response_stop_reason"] = common_agent_generation_stop_reason_name(turn.response_stop_reason);
    response["total_decoded_tokens"] = turn.total_decoded_tokens;
    response["event_count"] = turn.event_count;
    response["trace_count"] = turn.trace_count;
    response["memory_learning_related_count"] = turn.memory_learning_related_count;
    response["memory_learning_summary"] = turn.memory_learning_summary;
    json trace_entries = json::array();
    for (const auto & entry : turn.trace) {
        trace_entries.push_back(serialize_agent_daemon_trace_entry(entry));
    }
    response["trace"] = std::move(trace_entries);
    if (!turn.plan_id.empty()) {
        response["plan_id"] = turn.plan_id;
    }
    if (!result.error.empty()) {
        response["error"] = result.error;
    }
    return response;
}

} // namespace

bool parse_agent_daemon_command(
        const json & parsed,
        const daemon_options & options,
        common_agent_runtime_host_mode default_mode,
        common_agent_daemon_command & command,
        std::string & error) {
    command = {};
    command.request_id = parsed.value("request_id", "");

    if (!parse_agent_daemon_command_name(parsed, command, error)) {
        return false;
    }
    if (command.type == common_agent_daemon_command_type::run_turn) {
        command.turn.emplace();
        command.turn->request.request_id = command.request_id;
        if (!parse_agent_daemon_turn_request(parsed, options, default_mode, command.turn->request.turn, error)) {
            return false;
        }
    }
    error.clear();
    return true;
}

json make_agent_daemon_ready_response(const daemon_options & options) {
    return {
        {"ok", true},
        {"event", "ready"},
        {"default_mode", options.default_mode},
        {"protocol_version", agent_daemon_protocol_version},
        {"capabilities", json::array({
            "chat",
            "mini",
            "planning",
            "reflection",
            "memory_learning",
            "scoped_sessions",
        })},
    };
}

json make_agent_daemon_error_response(const std::string & error) {
    return {
        {"ok", false},
        {"error", error},
    };
}

json make_agent_daemon_command_response(const common_agent_daemon_command_result & result) {
    switch (result.response_kind) {
        case common_agent_daemon_response_kind::status:
            return make_agent_daemon_status_response(result);
        case common_agent_daemon_response_kind::lifecycle:
            return make_agent_daemon_lifecycle_response(result);
        case common_agent_daemon_response_kind::resource:
            return make_agent_daemon_resource_response(result);
        case common_agent_daemon_response_kind::listing:
            return make_agent_daemon_listing_response(result);
        case common_agent_daemon_response_kind::turn:
            return make_agent_daemon_turn_response(result);
    }
    return make_agent_daemon_turn_response(result);
}
