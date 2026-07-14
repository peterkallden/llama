#pragma once

#include <cstdint>
#include <functional>
#include <string>

enum class common_agent_daemon_event_type {
    unknown,
    command_queued,
    command_started,
    turn_accepted,
    turn_started,
    turn_completed,
    turn_failed,
    turn_cancelled,
    tool_started,
    tool_completed,
    memory_learned,
    session_reset_requested,
    session_reset,
    session_close_requested,
    session_closed,
    lane_drained,
    status_reported,
    resources_listed,
    memories_listed,
    plans_listed,
    resource_read,
    drain_requested,
    shutdown_requested,
};

inline const char * common_agent_daemon_event_type_name(
        common_agent_daemon_event_type type) {
    switch (type) {
        case common_agent_daemon_event_type::unknown:                 return "unknown";
        case common_agent_daemon_event_type::command_queued:          return "command.queued";
        case common_agent_daemon_event_type::command_started:         return "command.started";
        case common_agent_daemon_event_type::turn_accepted:           return "turn.accepted";
        case common_agent_daemon_event_type::turn_started:            return "turn.started";
        case common_agent_daemon_event_type::turn_completed:          return "turn.completed";
        case common_agent_daemon_event_type::turn_failed:             return "turn.failed";
        case common_agent_daemon_event_type::turn_cancelled:          return "turn.cancelled";
        case common_agent_daemon_event_type::tool_started:            return "tool.started";
        case common_agent_daemon_event_type::tool_completed:          return "tool.completed";
        case common_agent_daemon_event_type::memory_learned:          return "memory.learned";
        case common_agent_daemon_event_type::session_reset_requested: return "session.reset_requested";
        case common_agent_daemon_event_type::session_reset:           return "session.reset";
        case common_agent_daemon_event_type::session_close_requested: return "session.close_requested";
        case common_agent_daemon_event_type::session_closed:          return "session.closed";
        case common_agent_daemon_event_type::lane_drained:            return "lane.drained";
        case common_agent_daemon_event_type::status_reported:         return "status.reported";
        case common_agent_daemon_event_type::resources_listed:        return "resources.listed";
        case common_agent_daemon_event_type::memories_listed:         return "memories.listed";
        case common_agent_daemon_event_type::plans_listed:            return "plans.listed";
        case common_agent_daemon_event_type::resource_read:           return "resource.read";
        case common_agent_daemon_event_type::drain_requested:         return "daemon.drain_requested";
        case common_agent_daemon_event_type::shutdown_requested:      return "daemon.shutdown_requested";
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

using common_agent_daemon_event_sink =
    std::function<void(
        common_agent_daemon_event_type type,
        const std::string & request_id,
        const std::string & turn_id,
        const std::string & detail)>;
