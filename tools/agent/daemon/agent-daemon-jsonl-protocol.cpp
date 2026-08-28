#include "agent-daemon-jsonl-protocol.h"

#include "../../../common/agent/protocol/agent-jsonl.h"
#include "../../../common/agent/dataset-contracts.h"
#include "base64.hpp"

using json = nlohmann::ordered_json;

namespace {

const char * session_command_name(agent_daemon_jsonl_session_command command) {
    switch (command) {
        case agent_daemon_jsonl_session_command::reset: return "reset_session";
        case agent_daemon_jsonl_session_command::close: return "close_session";
    }
    return "reset_session";
}

bool parse_string_array_field(
        const json & value,
        std::vector<std::string> & output) {
    if (!value.is_array()) {
        return false;
    }
    output.clear();
    for (const auto & item : value) {
        if (!item.is_string()) {
            return false;
        }
        output.push_back(item.get<std::string>());
    }
    return true;
}

bool parse_resource_descriptor_field(
        const json & value,
        agent_resource_descriptor & descriptor) {
    if (!value.is_object()) {
        return false;
    }
    descriptor = {};
    descriptor.uri = value.value("uri", std::string());
    descriptor.name = value.value("name", std::string());
    descriptor.description = value.value("description", std::string());
    descriptor.mime_type = value.value("mime_type", std::string());
    descriptor.size_bytes = value.value("size_bytes", size_t(0));
    descriptor.resource_id = value.value("resource_id", std::string());
    const auto scope = value.value("scope", "turn");
    if (scope == "session") {
        descriptor.scope = common_runtime_resource_scope::session;
    } else if (scope == "project") {
        descriptor.scope = common_runtime_resource_scope::project;
    } else if (scope == "turn") {
        descriptor.scope = common_runtime_resource_scope::turn;
    } else {
        return false;
    }
    if (value.contains("metadata") && value["metadata"].is_object()) {
        const auto & metadata = value["metadata"];
        descriptor.metadata.purpose = metadata.value("purpose", std::string());
        descriptor.metadata.content_summary = metadata.value("content_summary", std::string());
        descriptor.metadata.usage_hint = metadata.value("usage_hint", std::string());
        descriptor.metadata.limitations = metadata.value("limitations", std::string());
        descriptor.metadata.keywords = metadata.value("keywords", std::vector<std::string>{});
        descriptor.metadata.entities = metadata.value("entities", std::vector<std::string>{});
    }
    return !descriptor.uri.empty();
}

bool parse_event_entry_field(
        const json & value,
        agent_daemon_jsonl_event_entry & entry) {
    if (!value.is_object()) {
        return false;
    }
    entry = {};
    entry.type = value.value("type", std::string());
    entry.request_id = value.value("request_id", std::string());
    entry.turn_id = value.value("turn_id", std::string());
    entry.namespace_id = value.value("namespace_id", std::string());
    entry.project_id = value.value("project_id", std::string());
    entry.session_id = value.value("session_id", std::string());
    entry.operation_id = value.value("operation_id", std::string());
    entry.detail = value.value("detail", std::string());
    entry.event_type = value.value("event_type", std::string());
    entry.sequence = value.value("sequence", uint64_t(0));
    entry.memory_id = value.value("memory_id", std::string());
    entry.plan_id = value.value("plan_id", std::string());
    entry.step_id = value.value("step_id", std::string());
    entry.observation_id = value.value("observation_id", std::string());
    entry.tool_name = value.value("tool_name", std::string());
    entry.resource_uri = value.value("resource_uri", std::string());
    return !entry.type.empty();
}

bool parse_common_response_event_fields(
        const json & message,
        int & daemon_event_count,
        std::vector<agent_daemon_jsonl_event_entry> & events) {
    daemon_event_count = message.value("daemon_event_count", 0);
    events.clear();
    if (!message.contains("events")) {
        return true;
    }
    const auto & value = message["events"];
    if (!value.is_array()) {
        return false;
    }
    for (const auto & item : value) {
        agent_daemon_jsonl_event_entry entry;
        if (!parse_event_entry_field(item, entry)) {
            return false;
        }
        events.push_back(std::move(entry));
    }
    return true;
}

} // namespace

bool read_agent_daemon_jsonl_message(
        FILE * stream,
        json & out,
        std::string & error) {
    return common_agent_jsonl_read_message(stream, out, error);
}

bool write_agent_daemon_jsonl_message(
        FILE * stream,
        const json & message,
        std::string & error) {
    return common_agent_jsonl_write_message(stream, message, error);
}

nlohmann::ordered_json make_agent_daemon_jsonl_event_message(
        const std::string & subscription_id,
        const common_agent_event_stream_delivery & delivery) {
    nlohmann::ordered_json message = {
        {"message_type", "event"},
        {"subscription_id", subscription_id},
        {"delivery_kind", delivery.kind == common_agent_event_stream_delivery_kind::event
            ? "event"
            : delivery.kind == common_agent_event_stream_delivery_kind::heartbeat
                ? "heartbeat"
                : delivery.kind == common_agent_event_stream_delivery_kind::closed
                    ? "closed" : "overflow"},
        {"cursor", { {"after_sequence", delivery.cursor.after_sequence} }},
    };
    if (delivery.kind == common_agent_event_stream_delivery_kind::event) {
        nlohmann::ordered_json event = {
            {"type", delivery.event.type},
            {"event_type", common_agent_daemon_event_type_name(delivery.event.event_type)},
            {"event_category", common_agent_daemon_event_category_name(delivery.event.category)},
            {"sequence", delivery.event.sequence},
        };
        if (!delivery.event.request_id.empty()) event["request_id"] = delivery.event.request_id;
        if (!delivery.event.turn_id.empty()) event["turn_id"] = delivery.event.turn_id;
        if (!delivery.event.namespace_id.empty()) event["namespace_id"] = delivery.event.namespace_id;
        if (!delivery.event.project_id.empty()) event["project_id"] = delivery.event.project_id;
        if (!delivery.event.session_id.empty()) event["session_id"] = delivery.event.session_id;
        if (!delivery.event.operation_id.empty()) event["operation_id"] = delivery.event.operation_id;
        if (!delivery.event.detail.empty()) event["detail"] = delivery.event.detail;
        message["event"] = std::move(event);
    } else if (delivery.kind == common_agent_event_stream_delivery_kind::overflow) {
        message["overflow"] = {
            {"from_sequence", delivery.overflow_from_sequence},
            {"to_sequence", delivery.overflow_to_sequence},
            {"skipped_sequence_count", delivery.skipped_sequence_count},
        };
    }
    return message;
}

json make_agent_daemon_jsonl_turn_request(
        const agent_daemon_jsonl_turn_request & request) {
    return common_agent_jsonl_make_turn_request(request);
}

json make_agent_daemon_jsonl_status_request(
        const agent_daemon_jsonl_status_request &) {
    return {
        {"command", "status"},
    };
}

json make_agent_daemon_jsonl_list_sessions_request(
        const agent_daemon_jsonl_list_sessions_request &) {
    return {
        {"command", "list_sessions"},
    };
}

json make_agent_daemon_jsonl_get_session_request(
        const agent_daemon_jsonl_get_session_request & request) {
    return {
        {"command", "get_session"},
        {"session_id", request.session_id},
        {"namespace_id", request.namespace_id},
    };
}

json make_agent_daemon_jsonl_list_resources_request(
        const agent_daemon_jsonl_list_resources_request & request) {
    return {
        {"command", "list_resources"},
        {"session_id", request.session_id},
        {"namespace_id", request.namespace_id},
        {"project_id", request.project_id},
        {"turn_id", request.turn_id},
    };
}

json make_agent_daemon_jsonl_list_memories_request(
        const agent_daemon_jsonl_list_memories_request & request) {
    return {
        {"command", "list_memories"},
        {"session_id", request.session_id},
        {"namespace_id", request.namespace_id},
        {"project_id", request.project_id},
        {"turn_id", request.turn_id},
    };
}

json make_agent_daemon_jsonl_list_plans_request(
        const agent_daemon_jsonl_list_plans_request & request) {
    return {
        {"command", "list_plans"},
        {"session_id", request.session_id},
        {"namespace_id", request.namespace_id},
        {"project_id", request.project_id},
        {"turn_id", request.turn_id},
    };
}

json make_agent_daemon_jsonl_read_resource_request(
        const agent_daemon_jsonl_read_resource_request & request) {
    return {
        {"command", "read_resource"},
        {"uri", request.uri},
        {"session_id", request.session_id},
        {"namespace_id", request.namespace_id},
        {"project_id", request.project_id},
        {"turn_id", request.turn_id},
        {"max_bytes", request.max_bytes},
    };
}

json make_agent_daemon_jsonl_put_resource_request(
        const agent_daemon_jsonl_put_resource_request & request) {
    json message = {
        {"command", "put_resource"},
        {"name", request.name},
        {"description", request.description},
        {"mime_type", request.mime_type},
        {"text", request.text},
        {"scope", request.scope},
        {"namespace_id", request.namespace_id},
        {"session_id", request.session_id},
        {"project_id", request.project_id},
        {"turn_id", request.turn_id},
    };
    if (!request.bytes.empty() || request.bytes_are_authoritative) {
        message["bytes_base64"] = base64::encode(request.bytes);
        message["text"] = "";
    }
    return message;
}

json make_agent_daemon_jsonl_drain_request(
        const agent_daemon_jsonl_drain_request &) {
    return {
        {"command", "drain"},
    };
}

json make_agent_daemon_jsonl_shutdown_request(
        const agent_daemon_jsonl_shutdown_request &) {
    return {
        {"command", "shutdown"},
    };
}

json make_agent_daemon_jsonl_session_request(
        const agent_daemon_jsonl_session_request & request) {
    return {
        {"command", session_command_name(request.command)},
        {"session_id", request.session_id},
        {"namespace_id", request.namespace_id},
    };
}

json make_agent_daemon_jsonl_reset_session_request(
        const std::string & session_id,
        const std::string & namespace_id) {
    return make_agent_daemon_jsonl_session_request({
        agent_daemon_jsonl_session_command::reset,
        session_id,
        namespace_id,
    });
}

json make_agent_daemon_jsonl_close_session_request(
        const std::string & session_id,
        const std::string & namespace_id) {
    return make_agent_daemon_jsonl_session_request({
        agent_daemon_jsonl_session_command::close,
        session_id,
        namespace_id,
    });
}

json make_agent_daemon_jsonl_cancel_request(
        const agent_daemon_jsonl_cancel_request & request) {
    return {
        {"command", "cancel_turn"},
        {"target_request_id", request.target_request_id},
        {"target_turn_id", request.target_turn_id},
    };
}

bool parse_agent_daemon_jsonl_ready_response(
        const json & message,
        agent_daemon_jsonl_ready_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object() ||
            !message.value("ok", false) ||
            message.value("event", std::string()) != "ready") {
        error = "unexpected daemon ready response";
        return false;
    }
    if (!message.contains("default_mode") || !message["default_mode"].is_string()) {
        error = "daemon ready response is missing default_mode";
        return false;
    }
    if (!message.contains("protocol_version") || !message["protocol_version"].is_number_integer()) {
        error = "daemon ready response is missing protocol_version";
        return false;
    }
    if (!message.contains("capabilities") ||
            !parse_string_array_field(message["capabilities"], response.capabilities)) {
        error = "daemon ready response is missing capabilities";
        return false;
    }

    response.default_mode = message["default_mode"].get<std::string>();
    response.protocol_version = message["protocol_version"].get<int>();
    error.clear();
    return true;
}

bool parse_agent_daemon_jsonl_turn_response(
        const json & message,
        agent_daemon_jsonl_turn_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object()) {
        error = "unexpected daemon turn response";
        return false;
    }

    response.ok = message.value("ok", false);
    response.event = message.value("event", std::string());
    if (!parse_common_response_event_fields(message, response.daemon_event_count, response.events)) {
        error = "unexpected daemon turn event payload";
        return false;
    }
    response.cancelled = message.value("cancelled", false);
    response.response = message.value("response", std::string());
    response.error = message.value("error", std::string());
    response.failure_class = message.value("failure_class", std::string());
    response.response_generation_status = message.value("response_generation_status", std::string());
    response.response_stop_reason = message.value("response_stop_reason", std::string());
    response.runtime_reused = message.value("runtime_reused", false);
    response.event_count = message.value("event_count", 0);
    if (message.contains("continuation_checkpoint") &&
            message["continuation_checkpoint"].is_object()) {
        const auto & value = message["continuation_checkpoint"];
        common_agent_continuation_checkpoint checkpoint;
        checkpoint.checkpoint_id = value.value("checkpoint_id", std::string());
        checkpoint.request_id = value.value("request_id", std::string());
        checkpoint.turn_id = value.value("turn_id", std::string());
        checkpoint.plan_id = value.value("plan_id", std::string());
        checkpoint.active_step_id = value.value("active_step_id", std::string());
        checkpoint.next_action = value.value("next_action", std::string());
        checkpoint.plan_version = value.value("plan_version", uint64_t(0));
        checkpoint.sequence = value.value("sequence", size_t(0));
        const auto reason = value.value("reason", std::string());
        if (reason == "completion_limit") {
            checkpoint.reason = common_agent_continuation_reason::completion_limit;
        } else if (reason == "phase_boundary") {
            checkpoint.reason = common_agent_continuation_reason::phase_boundary;
        } else {
            checkpoint.reason = common_agent_continuation_reason::context_pressure;
        }
        checkpoint.completed_step_ids = value.value(
            "completed_step_ids", std::vector<std::string>());
        checkpoint.chunk_parent_uri = value.value("chunk_parent_uri", std::string());
        checkpoint.chunk_count = value.value("chunk_count", size_t(0));
        checkpoint.completed_chunk_indexes = value.value(
            "completed_chunk_indexes", std::vector<size_t>());
        if (value.contains("resource_refs") && value["resource_refs"].is_array()) {
            for (const auto & item : value["resource_refs"]) {
                if (!item.is_object()) {
                    continue;
                }
                common_runtime_resource_ref resource;
                resource.uri = item.value("uri", std::string());
                resource.name = item.value("name", std::string());
                resource.description = item.value("description", std::string());
                resource.mime_type = item.value("mime_type", std::string());
                resource.size_bytes = item.value("size_bytes", size_t(0));
                checkpoint.resource_refs.push_back(std::move(resource));
            }
        }
        if (value.contains("dataset_refs") && value["dataset_refs"].is_array()) {
            for (const auto & item : value["dataset_refs"]) {
                if (!item.is_object()) continue;
                common_agent_dataset_ref dataset;
                if (!common_agent_dataset_ref_from_json(item, dataset, error)) return false;
                checkpoint.dataset_refs.push_back(std::move(dataset));
            }
        }
        if (value.contains("working_state") && value["working_state"].is_object()) {
            common_agent_working_state state;
            const auto & state_value = value["working_state"];
            state.goal = state_value.value("goal", std::string());
            state.current_phase = state_value.value("current_phase", std::string());
            state.completed_steps = state_value.value("completed_steps", std::vector<std::string>());
            state.active_step = state_value.value("active_step", std::string());
            state.remaining_steps = state_value.value("remaining_steps", std::vector<std::string>());
            state.decisions = state_value.value("decisions", std::vector<std::string>());
            state.constraints = state_value.value("constraints", std::vector<std::string>());
            state.open_questions = state_value.value("open_questions", std::vector<std::string>());
            if (state_value.contains("resource_refs") && state_value["resource_refs"].is_array()) {
                for (const auto & item : state_value["resource_refs"]) {
                    if (!item.is_object()) continue;
                    common_runtime_resource_ref resource;
                    resource.uri = item.value("uri", std::string());
                    resource.name = item.value("name", std::string());
                    resource.mime_type = item.value("mime_type", std::string());
                    resource.size_bytes = item.value("size_bytes", size_t(0));
                    state.resource_refs.push_back(std::move(resource));
                }
            }
            if (state_value.contains("dataset_refs") && state_value["dataset_refs"].is_array()) {
                for (const auto & item : state_value["dataset_refs"]) {
                    if (!item.is_object()) continue;
                    common_agent_dataset_ref dataset;
                    if (!common_agent_dataset_ref_from_json(item, dataset, error)) return false;
                    state.dataset_refs.push_back(std::move(dataset));
                }
            }
            state.chunk_status = state_value.value("chunk_status", std::vector<std::string>());
            state.tool_results = state_value.value("tool_results", std::vector<std::string>());
            state.continuation_action = state_value.value("continuation_action", std::string());
            checkpoint.working_state = std::move(state);
        }
        response.continuation_checkpoint = std::move(checkpoint);
    }
    if (message.contains("turn_summary") && message["turn_summary"].is_object()) {
        const auto & value = message["turn_summary"];
        common_agent_turn_summary summary;
        summary.mode = value.value("mode", std::string());
        summary.status = value.value("status", std::string());
        summary.objective = value.value("objective", std::string());
        summary.phases = value.value("phases", std::vector<std::string>());
        summary.tools_used = value.value("tools_used", std::vector<std::string>());
        summary.plan_revisions = value.value("plan_revisions", size_t(0));
        summary.sources = value.value("sources", size_t(0));
        summary.evidence_items = value.value("evidence_items", size_t(0));
        summary.unresolved_items = value.value("unresolved_items", size_t(0));
        summary.verified = value.value("verified", false);
        summary.stop_reason = value.value("stop_reason", std::string());
        summary.unresolved = value.value("unresolved", std::vector<std::string>());
        response.turn_summary = std::move(summary);
    }

    if (response.ok) {
        error.clear();
        return true;
    }

    error = response.error.empty() ? "daemon turn failed" : response.error;
    return false;
}

bool parse_agent_daemon_jsonl_status_response(
        const json & message,
        agent_daemon_jsonl_status_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object()) {
        error = "unexpected daemon status response";
        return false;
    }

    response.ok = message.value("ok", false);
    response.event = message.value("event", std::string());
    if (!parse_common_response_event_fields(message, response.daemon_event_count, response.events)) {
        error = "unexpected daemon status event payload";
        return false;
    }
    response.state = message.value("state", std::string());
    response.live = message.value("live", false);
    response.ready = message.value("ready", false);
    response.readiness = message.value("readiness", json::object());
    response.worker_running = message.value("worker_running", false);
    response.worker_count = message.value("worker_count", 1);
    response.workers_running = message.value("workers_running", 0);
    response.accepting_commands = message.value("accepting_commands", false);
    response.shutdown_requested = message.value("shutdown_requested", false);
    response.sessions = message.value("sessions", 0);
    response.queued_commands = message.value("queued_commands", 0);
    response.max_queue_size = message.value("max_queue_size", 0);
    response.queue_capacity_remaining = message.value("queue_capacity_remaining", 0);
    if (message.contains("metrics") && message["metrics"].is_object()) {
        const auto & metrics = message["metrics"];
        response.commands_accepted = metrics.value("commands_accepted", uint64_t(0));
        response.commands_completed = metrics.value("commands_completed", uint64_t(0));
        response.commands_failed = metrics.value("commands_failed", uint64_t(0));
        response.turns_completed = metrics.value("turns_completed", uint64_t(0));
        response.tools_completed = metrics.value("tools_completed", uint64_t(0));
    }
    response.active_request_id = message.value("active_request_id", std::string());
    response.active_turn_id = message.value("active_turn_id", std::string());
    response.active_turn_phase = message.value("active_turn_phase", std::string());
    response.active_turn_disposition = message.value("active_turn_disposition", std::string());
    response.active_pending_operation_kind = message.value("active_pending_operation_kind", std::string());
    response.active_pending_operation_detail = message.value("active_pending_operation_detail", std::string());
    response.payload = message;
    response.error = message.value("error", std::string());

    if (message.contains("session_keys") && message["session_keys"].is_array()) {
        for (const auto & item : message["session_keys"]) {
            if (!item.is_object()) {
                continue;
            }
            response.session_keys.push_back({
                item.value("namespace_id", std::string()),
                item.value("session_id", std::string()),
                item.value("project_id", std::string()),
                item.value("memory_scope", std::string()),
                item.value("plan_scope", std::string()),
                item.value("policy_pack_id", std::string()),
                item.value("lane_state", std::string()),
                item.value("queued_turn_count", 0),
                item.contains("active_turn_id"),
                item.value("active_request_id", std::string()),
                item.value("active_turn_id", std::string()),
                item.value("active_turn_phase", std::string()),
                item.value("active_turn_disposition", std::string()),
                item.value("active_cancel_requested", false),
                item.value("pending_operation_kind", std::string()),
                item.value("pending_operation_detail", std::string()),
                item.value("last_turn_id", std::string()),
                item.value("last_turn_phase", std::string()),
                item.value("last_turn_disposition", std::string()),
            });
        }
    }

    if (response.ok) {
        error.clear();
        return true;
    }

    error = response.error.empty() ? "daemon status failed" : response.error;
    return false;
}

bool parse_agent_daemon_jsonl_lifecycle_response(
        const json & message,
        agent_daemon_jsonl_lifecycle_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object()) {
        error = "unexpected daemon lifecycle response";
        return false;
    }

    response.ok = message.value("ok", false);
    response.event = message.value("event", std::string());
    if (!parse_common_response_event_fields(message, response.daemon_event_count, response.events)) {
        error = "unexpected daemon lifecycle event payload";
        return false;
    }
    response.target_request_id = message.value("target_request_id", std::string());
    response.target_turn_id = message.value("target_turn_id", std::string());
    response.error = message.value("error", std::string());

    std::string status_error;
    if (!parse_agent_daemon_jsonl_status_response(message, response.status, status_error)) {
        if (response.ok) {
            error = status_error.empty() ? "daemon lifecycle response missing status snapshot" : status_error;
            return false;
        }
    }

    if (response.ok && !response.event.empty()) {
        error.clear();
        return true;
    }

    error = response.error.empty() ? "daemon lifecycle failed" : response.error;
    return false;
}

bool parse_agent_daemon_jsonl_resource_response(
        const json & message,
        agent_daemon_jsonl_resource_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object()) {
        error = "unexpected daemon resource response";
        return false;
    }

    response.ok = message.value("ok", false);
    response.event = message.value("event", std::string());
    if (!parse_common_response_event_fields(message, response.daemon_event_count, response.events)) {
        error = "unexpected daemon resource event payload";
        return false;
    }
    response.content = message.value("content", std::string());
    response.payload = message;
    response.error = message.value("error", std::string());
    if (message.contains("resource")) {
        if (!parse_resource_descriptor_field(message["resource"], response.resource)) {
            error = "daemon resource response is missing resource descriptor";
            return false;
        }
    }

    std::string status_error;
    if (!parse_agent_daemon_jsonl_status_response(message, response.status, status_error)) {
        if (response.ok) {
            error = status_error.empty() ? "daemon resource response missing status snapshot" : status_error;
            return false;
        }
    }

    if (response.ok) {
        error.clear();
        return true;
    }

    error = response.error.empty() ? "daemon resource read failed" : response.error;
    return false;
}

bool parse_agent_daemon_jsonl_listing_response(
        const json & message,
        agent_daemon_jsonl_listing_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object()) {
        error = "unexpected daemon listing response";
        return false;
    }

    response.ok = message.value("ok", false);
    response.event = message.value("event", std::string());
    if (!parse_common_response_event_fields(message, response.daemon_event_count, response.events)) {
        error = "unexpected daemon listing event payload";
        return false;
    }
    response.payload = message;
    response.error = message.value("error", std::string());

    std::string status_error;
    if (!parse_agent_daemon_jsonl_status_response(message, response.status, status_error)) {
        if (response.ok) {
            error = status_error.empty() ? "daemon listing response missing status snapshot" : status_error;
            return false;
        }
    }

    if (response.ok) {
        error.clear();
        return true;
    }

    error = response.error.empty() ? "daemon listing failed" : response.error;
    return false;
}

bool parse_agent_daemon_jsonl_event_response(
        const json & message,
        agent_daemon_jsonl_event_response & response,
        std::string & error) {
    response = {};
    if (!message.is_object()) {
        error = "unexpected daemon event response";
        return false;
    }

    response.ok = message.value("ok", false);
    response.event = message.value("event", std::string());
    if (!parse_common_response_event_fields(message, response.daemon_event_count, response.events)) {
        error = "unexpected daemon event payload";
        return false;
    }
    response.error = message.value("error", std::string());

    if (response.ok && !response.event.empty()) {
        error.clear();
        return true;
    }

    error = response.error.empty() ? "daemon event failed" : response.error;
    return false;
}

bool parse_agent_daemon_jsonl_event_response(
        const json & message,
        const std::string & expected_event,
        std::string & error) {
    agent_daemon_jsonl_event_response response;
    if (!parse_agent_daemon_jsonl_event_response(message, response, error) ||
            response.event != expected_event) {
        error = "unexpected daemon " + expected_event + " response";
        return false;
    }
    error.clear();
    return true;
}
