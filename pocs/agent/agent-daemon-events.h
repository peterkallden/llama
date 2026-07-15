#pragma once

#include <cstdint>
#include <functional>
#include <string>

enum class common_agent_daemon_event_type {
    unknown,
    command_queued,
    command_started,
    command_rejected,
    turn_accepted,
    turn_started,
    turn_rejected,
    turn_cancel_requested,
    turn_cancel_rejected,
    turn_waiting_for_tool,
    turn_waiting_for_inference,
    turn_completed,
    turn_failed,
    turn_cancelled,
    tool_started,
    tool_completed,
    memory_learned,
    plan_created,
    plan_updated,
    plan_step_started,
    plan_step_completed,
    observation_recorded,
    resource_created,
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
    command_failed,
};

inline const char * common_agent_daemon_event_type_name(
        common_agent_daemon_event_type type) {
    switch (type) {
        case common_agent_daemon_event_type::unknown:                 return "unknown";
        case common_agent_daemon_event_type::command_queued:          return "command.queued";
        case common_agent_daemon_event_type::command_started:         return "command.started";
        case common_agent_daemon_event_type::command_rejected:        return "command.rejected";
        case common_agent_daemon_event_type::turn_accepted:           return "turn.accepted";
        case common_agent_daemon_event_type::turn_started:            return "turn.started";
        case common_agent_daemon_event_type::turn_rejected:           return "turn.rejected";
        case common_agent_daemon_event_type::turn_cancel_requested:   return "turn.cancel_requested";
        case common_agent_daemon_event_type::turn_cancel_rejected:    return "turn.cancel_rejected";
        case common_agent_daemon_event_type::turn_waiting_for_tool:   return "turn.waiting_for_tool";
        case common_agent_daemon_event_type::turn_waiting_for_inference: return "turn.waiting_for_inference";
        case common_agent_daemon_event_type::turn_completed:          return "turn.completed";
        case common_agent_daemon_event_type::turn_failed:             return "turn.failed";
        case common_agent_daemon_event_type::turn_cancelled:          return "turn.cancelled";
        case common_agent_daemon_event_type::tool_started:            return "tool.started";
        case common_agent_daemon_event_type::tool_completed:          return "tool.completed";
        case common_agent_daemon_event_type::memory_learned:          return "memory.learned";
        case common_agent_daemon_event_type::plan_created:           return "plan.created";
        case common_agent_daemon_event_type::plan_updated:           return "plan.updated";
        case common_agent_daemon_event_type::plan_step_started:      return "plan.step_started";
        case common_agent_daemon_event_type::plan_step_completed:    return "plan.step_completed";
        case common_agent_daemon_event_type::observation_recorded:   return "observation.recorded";
        case common_agent_daemon_event_type::resource_created:       return "resource.created";
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
        case common_agent_daemon_event_type::command_failed:          return "command.failed";
    }
    return "unknown";
}

struct common_agent_daemon_event {
    std::string type;
    std::string request_id;
    std::string turn_id;
    std::string detail;
    common_agent_daemon_event_type event_type = common_agent_daemon_event_type::unknown;
    uint64_t sequence = 0;
};

inline common_agent_daemon_event make_common_agent_daemon_event(
        common_agent_daemon_event_type type,
        std::string request_id,
        std::string turn_id,
        std::string detail = {},
        uint64_t sequence = 0) {
    return {
        common_agent_daemon_event_type_name(type),
        std::move(request_id),
        std::move(turn_id),
        std::move(detail),
        type,
        sequence,
    };
}

using common_agent_daemon_event_sink =
    std::function<void(
        common_agent_daemon_event_type type,
        const std::string & request_id,
        const std::string & turn_id,
        const std::string & detail)>;
