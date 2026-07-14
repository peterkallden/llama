#pragma once

#include "agent-runtime-turn.h"
#include "agent-runtime-control.h"

#include <string>

enum class common_agent_runtime_turn_phase {
    queued,
    preparing,
    awaiting_inference,
    awaiting_tool,
    completing,
    completed,
    failed,
    cancelled,
};

inline const char * common_agent_runtime_turn_phase_name(
        common_agent_runtime_turn_phase phase) {
    switch (phase) {
        case common_agent_runtime_turn_phase::queued:             return "queued";
        case common_agent_runtime_turn_phase::preparing:          return "preparing";
        case common_agent_runtime_turn_phase::awaiting_inference: return "awaiting_inference";
        case common_agent_runtime_turn_phase::awaiting_tool:      return "awaiting_tool";
        case common_agent_runtime_turn_phase::completing:         return "completing";
        case common_agent_runtime_turn_phase::completed:          return "completed";
        case common_agent_runtime_turn_phase::failed:             return "failed";
        case common_agent_runtime_turn_phase::cancelled:          return "cancelled";
    }
    return "queued";
}

enum class common_agent_runtime_turn_disposition {
    continue_immediately,
    completed,
    failed,
    cancelled,
};

inline const char * common_agent_runtime_turn_disposition_name(
        common_agent_runtime_turn_disposition disposition) {
    switch (disposition) {
        case common_agent_runtime_turn_disposition::continue_immediately: return "continue_immediately";
        case common_agent_runtime_turn_disposition::completed:            return "completed";
        case common_agent_runtime_turn_disposition::failed:               return "failed";
        case common_agent_runtime_turn_disposition::cancelled:            return "cancelled";
    }
    return "continue_immediately";
}

struct common_agent_runtime_turn_execution {
    std::string request_id;
    std::string turn_id;
    common_agent_runtime_host_mode mode = common_agent_runtime_host_mode::chat;
    common_agent_runtime_turn_phase phase = common_agent_runtime_turn_phase::queued;
    common_agent_runtime_turn_disposition disposition = common_agent_runtime_turn_disposition::continue_immediately;
    bool cancellation_requested = false;
    std::shared_ptr<common_agent_runtime_cancellation_state> cancellation;
};
