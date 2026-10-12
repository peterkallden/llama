#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>

enum class common_agent_daemon_event_type {
    unknown,
    command_queued,
    command_started,
    command_rejected,
    turn_accepted,
    turn_started,
    turn_resumed,
    turn_rejected,
    turn_cancel_requested,
    turn_cancel_rejected,
    turn_waiting_for_tool,
    turn_waiting_for_inference,
    inference_queued,
    inference_capacity_granted,
    inference_started,
    inference_completed,
    tool_queued,
    turn_completed,
    turn_failed,
    turn_cancelled,
    tool_started,
    tool_progress,
    tool_output,
    tool_artifact_created,
    tool_completed,
    tool_failed,
    tool_cancelled,
    tool_timed_out,
    memory_learned,
    plan_created,
    plan_updated,
    plan_step_started,
    plan_step_completed,
    observation_recorded,
    resource_chunk_planned,
    resource_chunk_processed,
    resource_created,
    resource_create_failed,
    resource_attached,
    session_reset_requested,
    session_reset,
    session_close_requested,
    session_closed,
    lane_drained,
    status_reported,
    sessions_listed,
    session_found,
    session_not_found,
    session_lookup_failed,
    resources_listed,
    resources_list_failed,
    memories_listed,
    memories_list_failed,
    plans_listed,
    plans_list_failed,
    resource_read,
    resource_not_found,
    resource_read_failed,
    session_reset_failed,
    session_close_failed,
    drain_requested,
    shutdown_requested,
    config_reload_started,
    config_reload_completed,
    config_reload_rejected,
    mcp_provider_added,
    mcp_provider_removed,
    mcp_provider_replaced,
    command_failed,
    agent_runtime_event,
};

enum class common_agent_daemon_event_category {
    unknown,
    command,
    turn,
    inference,
    tool,
    memory,
    plan,
    observation,
    resource,
    session,
    daemon,
    config,
    mcp,
    agent,
};

inline const char * common_agent_daemon_event_category_name(
        common_agent_daemon_event_category category) {
    switch (category) {
        case common_agent_daemon_event_category::unknown:      return "unknown";
        case common_agent_daemon_event_category::command:      return "command";
        case common_agent_daemon_event_category::turn:         return "turn";
        case common_agent_daemon_event_category::inference:    return "inference";
        case common_agent_daemon_event_category::tool:         return "tool";
        case common_agent_daemon_event_category::memory:       return "memory";
        case common_agent_daemon_event_category::plan:         return "plan";
        case common_agent_daemon_event_category::observation:  return "observation";
        case common_agent_daemon_event_category::resource:     return "resource";
        case common_agent_daemon_event_category::session:      return "session";
        case common_agent_daemon_event_category::daemon:       return "daemon";
        case common_agent_daemon_event_category::config:       return "config";
        case common_agent_daemon_event_category::mcp:          return "mcp";
        case common_agent_daemon_event_category::agent:        return "agent";
    }
    return "unknown";
}

inline common_agent_daemon_event_category common_agent_daemon_event_category_for_type(
        common_agent_daemon_event_type type) {
    switch (type) {
        case common_agent_daemon_event_type::command_queued:
        case common_agent_daemon_event_type::command_started:
        case common_agent_daemon_event_type::command_rejected:
        case common_agent_daemon_event_type::command_failed:
            return common_agent_daemon_event_category::command;
        case common_agent_daemon_event_type::turn_accepted:
        case common_agent_daemon_event_type::turn_started:
        case common_agent_daemon_event_type::turn_resumed:
        case common_agent_daemon_event_type::turn_rejected:
        case common_agent_daemon_event_type::turn_cancel_requested:
        case common_agent_daemon_event_type::turn_cancel_rejected:
        case common_agent_daemon_event_type::turn_waiting_for_tool:
        case common_agent_daemon_event_type::turn_waiting_for_inference:
        case common_agent_daemon_event_type::turn_completed:
        case common_agent_daemon_event_type::turn_failed:
        case common_agent_daemon_event_type::turn_cancelled:
            return common_agent_daemon_event_category::turn;
        case common_agent_daemon_event_type::inference_queued:
        case common_agent_daemon_event_type::inference_capacity_granted:
        case common_agent_daemon_event_type::inference_started:
        case common_agent_daemon_event_type::inference_completed:
            return common_agent_daemon_event_category::inference;
        case common_agent_daemon_event_type::tool_queued:
        case common_agent_daemon_event_type::tool_started:
        case common_agent_daemon_event_type::tool_progress:
        case common_agent_daemon_event_type::tool_output:
        case common_agent_daemon_event_type::tool_artifact_created:
        case common_agent_daemon_event_type::tool_completed:
        case common_agent_daemon_event_type::tool_failed:
        case common_agent_daemon_event_type::tool_cancelled:
        case common_agent_daemon_event_type::tool_timed_out:
            return common_agent_daemon_event_category::tool;
        case common_agent_daemon_event_type::memory_learned:
        case common_agent_daemon_event_type::memories_listed:
        case common_agent_daemon_event_type::memories_list_failed:
            return common_agent_daemon_event_category::memory;
        case common_agent_daemon_event_type::plan_created:
        case common_agent_daemon_event_type::plan_updated:
        case common_agent_daemon_event_type::plan_step_started:
        case common_agent_daemon_event_type::plan_step_completed:
        case common_agent_daemon_event_type::plans_listed:
        case common_agent_daemon_event_type::plans_list_failed:
            return common_agent_daemon_event_category::plan;
        case common_agent_daemon_event_type::observation_recorded:
            return common_agent_daemon_event_category::observation;
        case common_agent_daemon_event_type::resource_chunk_planned:
        case common_agent_daemon_event_type::resource_chunk_processed:
        case common_agent_daemon_event_type::resource_created:
        case common_agent_daemon_event_type::resource_create_failed:
        case common_agent_daemon_event_type::resource_attached:
        case common_agent_daemon_event_type::resources_listed:
        case common_agent_daemon_event_type::resources_list_failed:
        case common_agent_daemon_event_type::resource_read:
        case common_agent_daemon_event_type::resource_not_found:
        case common_agent_daemon_event_type::resource_read_failed:
            return common_agent_daemon_event_category::resource;
        case common_agent_daemon_event_type::session_reset_requested:
        case common_agent_daemon_event_type::session_reset:
        case common_agent_daemon_event_type::session_close_requested:
        case common_agent_daemon_event_type::session_closed:
        case common_agent_daemon_event_type::session_reset_failed:
        case common_agent_daemon_event_type::session_close_failed:
        case common_agent_daemon_event_type::sessions_listed:
        case common_agent_daemon_event_type::session_found:
        case common_agent_daemon_event_type::session_not_found:
        case common_agent_daemon_event_type::session_lookup_failed:
            return common_agent_daemon_event_category::session;
        case common_agent_daemon_event_type::drain_requested:
        case common_agent_daemon_event_type::shutdown_requested:
        case common_agent_daemon_event_type::lane_drained:
        case common_agent_daemon_event_type::status_reported:
            return common_agent_daemon_event_category::daemon;
        case common_agent_daemon_event_type::config_reload_started:
        case common_agent_daemon_event_type::config_reload_completed:
        case common_agent_daemon_event_type::config_reload_rejected:
            return common_agent_daemon_event_category::config;
        case common_agent_daemon_event_type::mcp_provider_added:
        case common_agent_daemon_event_type::mcp_provider_removed:
        case common_agent_daemon_event_type::mcp_provider_replaced:
            return common_agent_daemon_event_category::mcp;
        case common_agent_daemon_event_type::agent_runtime_event:
            return common_agent_daemon_event_category::agent;
        case common_agent_daemon_event_type::unknown:
            return common_agent_daemon_event_category::unknown;
    }
    return common_agent_daemon_event_category::unknown;
}

inline const char * common_agent_daemon_event_type_name(
        common_agent_daemon_event_type type) {
    switch (type) {
        case common_agent_daemon_event_type::unknown:                 return "unknown";
        case common_agent_daemon_event_type::command_queued:          return "command.queued";
        case common_agent_daemon_event_type::command_started:         return "command.started";
        case common_agent_daemon_event_type::command_rejected:        return "command.rejected";
        case common_agent_daemon_event_type::turn_accepted:           return "turn.accepted";
        case common_agent_daemon_event_type::turn_started:            return "turn.started";
        case common_agent_daemon_event_type::turn_resumed:            return "turn.resumed";
        case common_agent_daemon_event_type::turn_rejected:           return "turn.rejected";
        case common_agent_daemon_event_type::turn_cancel_requested:   return "turn.cancel_requested";
        case common_agent_daemon_event_type::turn_cancel_rejected:    return "turn.cancel_rejected";
        case common_agent_daemon_event_type::turn_waiting_for_tool:   return "turn.waiting_for_tool";
        case common_agent_daemon_event_type::turn_waiting_for_inference: return "turn.waiting_for_inference";
        case common_agent_daemon_event_type::inference_queued:       return "inference.queued";
        case common_agent_daemon_event_type::inference_capacity_granted: return "inference.capacity_granted";
        case common_agent_daemon_event_type::inference_started:       return "inference.started";
        case common_agent_daemon_event_type::inference_completed:     return "inference.completed";
        case common_agent_daemon_event_type::tool_queued:            return "tool.queued";
        case common_agent_daemon_event_type::turn_completed:          return "turn.completed";
        case common_agent_daemon_event_type::turn_failed:             return "turn.failed";
        case common_agent_daemon_event_type::turn_cancelled:          return "turn.cancelled";
        case common_agent_daemon_event_type::tool_started:            return "tool.started";
        case common_agent_daemon_event_type::tool_progress:           return "tool.progress";
        case common_agent_daemon_event_type::tool_output:             return "tool.output";
        case common_agent_daemon_event_type::tool_artifact_created:   return "tool.artifact_created";
        case common_agent_daemon_event_type::tool_completed:          return "tool.completed";
        case common_agent_daemon_event_type::tool_failed:             return "tool.failed";
        case common_agent_daemon_event_type::tool_cancelled:          return "tool.cancelled";
        case common_agent_daemon_event_type::tool_timed_out:          return "tool.timed_out";
        case common_agent_daemon_event_type::memory_learned:          return "memory.learned";
        case common_agent_daemon_event_type::plan_created:           return "plan.created";
        case common_agent_daemon_event_type::plan_updated:           return "plan.updated";
        case common_agent_daemon_event_type::plan_step_started:      return "plan.step_started";
        case common_agent_daemon_event_type::plan_step_completed:    return "plan.step_completed";
        case common_agent_daemon_event_type::observation_recorded:   return "observation.recorded";
        case common_agent_daemon_event_type::resource_chunk_planned: return "resource.chunk_planned";
        case common_agent_daemon_event_type::resource_chunk_processed: return "resource.chunk_processed";
        case common_agent_daemon_event_type::resource_created:       return "resource.created";
        case common_agent_daemon_event_type::resource_create_failed: return "resource.create_failed";
        case common_agent_daemon_event_type::resource_attached:      return "resource.attached";
        case common_agent_daemon_event_type::session_reset_requested: return "session.reset_requested";
        case common_agent_daemon_event_type::session_reset:           return "session.reset";
        case common_agent_daemon_event_type::session_close_requested: return "session.close_requested";
        case common_agent_daemon_event_type::session_closed:          return "session.closed";
        case common_agent_daemon_event_type::lane_drained:            return "lane.drained";
        case common_agent_daemon_event_type::status_reported:         return "status.reported";
        case common_agent_daemon_event_type::sessions_listed:         return "sessions.listed";
        case common_agent_daemon_event_type::session_found:           return "session.found";
        case common_agent_daemon_event_type::session_not_found:       return "session.not_found";
        case common_agent_daemon_event_type::session_lookup_failed:   return "session.lookup_failed";
        case common_agent_daemon_event_type::resources_listed:        return "resources.listed";
        case common_agent_daemon_event_type::resources_list_failed:   return "resources.list_failed";
        case common_agent_daemon_event_type::memories_listed:         return "memories.listed";
        case common_agent_daemon_event_type::memories_list_failed:    return "memories.list_failed";
        case common_agent_daemon_event_type::plans_listed:            return "plans.listed";
        case common_agent_daemon_event_type::plans_list_failed:       return "plans.list_failed";
        case common_agent_daemon_event_type::resource_read:           return "resource.read";
        case common_agent_daemon_event_type::resource_not_found:      return "resource.not_found";
        case common_agent_daemon_event_type::resource_read_failed:    return "resource.read_failed";
        case common_agent_daemon_event_type::session_reset_failed:    return "session.reset_failed";
        case common_agent_daemon_event_type::session_close_failed:    return "session.close_failed";
        case common_agent_daemon_event_type::drain_requested:         return "daemon.drain_requested";
        case common_agent_daemon_event_type::shutdown_requested:      return "daemon.shutdown_requested";
        case common_agent_daemon_event_type::config_reload_started:   return "config.reload.started";
        case common_agent_daemon_event_type::config_reload_completed: return "config.reload.completed";
        case common_agent_daemon_event_type::config_reload_rejected:  return "config.reload.rejected";
        case common_agent_daemon_event_type::mcp_provider_added:      return "mcp.provider.added";
        case common_agent_daemon_event_type::mcp_provider_removed:    return "mcp.provider.removed";
        case common_agent_daemon_event_type::mcp_provider_replaced:   return "mcp.provider.replaced";
        case common_agent_daemon_event_type::command_failed:          return "command.failed";
        case common_agent_daemon_event_type::agent_runtime_event:     return "agent.runtime_event";
    }
    return "unknown";
}

struct common_agent_daemon_event {
    std::string type;
    std::string request_id;
    std::string turn_id;
    std::string namespace_id;
    std::string project_id;
    std::string session_id;
    std::string operation_id;
    std::string detail;
    common_agent_daemon_event_type event_type = common_agent_daemon_event_type::unknown;
    uint64_t sequence = 0;
    common_agent_daemon_event_category category = common_agent_daemon_event_category::unknown;
};

enum class common_agent_event_stream_delivery_kind {
    event,
    heartbeat,
    closed,
    overflow,
};

enum class common_agent_event_stream_wait_status {
    delivered,
    timeout,
    closed,
    not_found,
};

struct common_agent_event_stream_cursor {
    uint64_t after_sequence = 0;
};

struct common_agent_event_stream_filter {
    std::string namespace_id;
    std::string project_id;
    std::string session_id;
    std::string request_id;
    std::string turn_id;
    std::string operation_id;

    bool matches(const common_agent_daemon_event & event) const {
        return (namespace_id.empty() || namespace_id == event.namespace_id) &&
            (project_id.empty() || project_id == event.project_id) &&
            (session_id.empty() || session_id == event.session_id) &&
            (request_id.empty() || request_id == event.request_id) &&
            (turn_id.empty() || turn_id == event.turn_id) &&
            (operation_id.empty() || operation_id == event.operation_id);
    }
};

struct common_agent_event_stream_subscription {
    std::string subscription_id;
    common_agent_event_stream_filter filter;
    common_agent_event_stream_cursor cursor;
    size_t max_pending_events = 256;
    bool include_terminal_events = true;
};

struct common_agent_event_stream_delivery {
    common_agent_event_stream_delivery_kind kind =
        common_agent_event_stream_delivery_kind::event;
    common_agent_daemon_event event;
    common_agent_event_stream_cursor cursor;
    uint64_t overflow_from_sequence = 0;
    uint64_t overflow_to_sequence = 0;
    uint64_t skipped_sequence_count = 0;
};

using common_agent_event_stream_sink =
    std::function<void(common_agent_event_stream_delivery delivery)>;

struct common_agent_event_context {
    std::string namespace_id;
    std::string project_id;
    std::string session_id;
    std::string request_id;
    std::string turn_id;
    std::string operation_id;
};

using common_agent_event_attributes = std::map<std::string, std::string>;

inline common_agent_daemon_event make_common_agent_daemon_event(
        common_agent_daemon_event_type type,
        std::string request_id,
        std::string turn_id,
        std::string detail = {},
        uint64_t sequence = 0,
        common_agent_event_context context = {}) {
    return {
        common_agent_daemon_event_type_name(type),
        std::move(request_id),
        std::move(turn_id),
        std::move(context.namespace_id),
        std::move(context.project_id),
        std::move(context.session_id),
        std::move(context.operation_id),
        std::move(detail),
        type,
        sequence,
        common_agent_daemon_event_category_for_type(type),
    };
}

using common_agent_daemon_event_sink =
    std::function<void(common_agent_daemon_event event)>;

class common_agent_event_emitter {
public:
    common_agent_event_emitter() = default;

    explicit common_agent_event_emitter(
            common_agent_daemon_event_sink sink,
            common_agent_event_context context = {})
        : sink(std::move(sink))
        , context(std::move(context)) {}

    void set_sink(common_agent_daemon_event_sink new_sink) {
        sink = std::move(new_sink);
    }

    void set_context(common_agent_event_context new_context) {
        context = std::move(new_context);
    }

    [[nodiscard]]
    common_agent_event_emitter with_context(
            common_agent_event_context child_context) const {
        return common_agent_event_emitter(
            sink,
            std::move(child_context));
    }

    [[nodiscard]]
    common_agent_event_emitter with_request(
            std::string request_id) const {
        auto child = context;
        child.request_id = std::move(request_id);
        return common_agent_event_emitter(
            sink,
            std::move(child));
    }

    [[nodiscard]]
    common_agent_event_emitter with_turn(
            std::string request_id,
            std::string turn_id) const {
        auto child = context;
        child.request_id = std::move(request_id);
        child.turn_id = std::move(turn_id);
        return common_agent_event_emitter(
            sink,
            std::move(child));
    }

    [[nodiscard]]
    common_agent_event_emitter with_operation(
            std::string operation_id) const {
        auto child = context;
        child.operation_id = std::move(operation_id);
        return common_agent_event_emitter(
            sink,
            std::move(child));
    }

    void emit(
            common_agent_daemon_event_type type,
            std::string detail = {}) const {
        if (!sink) {
            return;
        }

        auto event = common_agent_daemon_event{
            common_agent_daemon_event_type_name(type),
            context.request_id,
            context.turn_id,
            context.namespace_id,
            context.project_id,
            context.session_id,
            context.operation_id,
            std::move(detail),
            type,
            0,
            common_agent_daemon_event_category_for_type(type),
        };
        sink(std::move(event));
    }

    explicit operator bool() const {
        return static_cast<bool>(sink);
    }

private:
    common_agent_daemon_event_sink sink;
    common_agent_event_context context;
};
