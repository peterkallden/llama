#pragma once

#include "../../../common/runtime/runtime-operation.h"
#include "../runtime/agent-runtime-control.h"
#include "../runtime/agent-runtime-turn.h"
#include "agent/agent-continuation.h"

#include <chrono>
#include <optional>
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
    wait_for_inference,
    wait_for_tool,
    completed,
    failed,
    cancelled,
};

inline const char * common_agent_runtime_turn_disposition_name(
        common_agent_runtime_turn_disposition disposition) {
    switch (disposition) {
        case common_agent_runtime_turn_disposition::continue_immediately: return "continue_immediately";
        case common_agent_runtime_turn_disposition::wait_for_inference:   return "wait_for_inference";
        case common_agent_runtime_turn_disposition::wait_for_tool:        return "wait_for_tool";
        case common_agent_runtime_turn_disposition::completed:            return "completed";
        case common_agent_runtime_turn_disposition::failed:               return "failed";
        case common_agent_runtime_turn_disposition::cancelled:            return "cancelled";
    }
    return "continue_immediately";
}

using common_agent_runtime_pending_operation_kind = common_runtime_operation_kind;

inline const char * common_agent_runtime_pending_operation_kind_name(
        common_agent_runtime_pending_operation_kind kind) {
    return common_runtime_operation_kind_name(kind);
}

using common_agent_runtime_pending_operation = common_runtime_operation;

struct common_agent_runtime_turn_execution {
    std::string request_id;
    std::string turn_id;
    common_agent_runtime_host_mode mode = common_agent_runtime_host_mode::chat;
    common_agent_runtime_turn_phase phase = common_agent_runtime_turn_phase::queued;
    common_agent_runtime_turn_disposition disposition = common_agent_runtime_turn_disposition::continue_immediately;
    bool cancellation_requested = false;
    std::shared_ptr<common_agent_runtime_cancellation_state> cancellation;
    std::optional<common_agent_runtime_pending_operation> pending_operation;
    std::optional<common_agent_continuation_checkpoint> continuation_checkpoint;
    size_t continuation_count = 0;
    bool inference_capacity_acquired = false;
};

inline common_agent_runtime_turn_execution make_common_agent_runtime_turn_execution(
        const std::string & request_id,
        const std::string & turn_id,
        common_agent_runtime_host_mode mode,
        bool cancellation_requested,
        const std::shared_ptr<common_agent_runtime_cancellation_state> & cancellation) {
    common_agent_runtime_turn_execution execution;
    execution.request_id = request_id;
    execution.turn_id = turn_id;
    execution.mode = mode;
    execution.cancellation_requested = cancellation_requested;
    execution.cancellation = cancellation;
    return execution;
}
