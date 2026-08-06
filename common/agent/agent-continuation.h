#pragma once

#include "plan/plan-types.h"
#include "agent/agent-working-state.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// A continuation checkpoint is execution state for the current turn.  It is
// deliberately separate from durable memory: it tells the existing session
// lane how to resume work, rather than becoming a generally retrievable fact.
enum class common_agent_continuation_reason {
    completion_limit,
    context_pressure,
    phase_boundary,
};

inline const char * common_agent_continuation_reason_name(
        common_agent_continuation_reason reason) {
    switch (reason) {
        case common_agent_continuation_reason::completion_limit: return "completion_limit";
        case common_agent_continuation_reason::context_pressure: return "context_pressure";
        case common_agent_continuation_reason::phase_boundary:   return "phase_boundary";
    }
    return "context_pressure";
}

struct common_agent_continuation_checkpoint {
    std::string checkpoint_id;
    std::string request_id;
    std::string turn_id;
    std::string plan_id;
    std::string active_step_id;
    std::string next_action;
    uint64_t plan_version = 0;
    size_t sequence = 0;
    common_agent_continuation_reason reason = common_agent_continuation_reason::context_pressure;
    std::vector<std::string> completed_step_ids;
    std::vector<common_runtime_resource_ref> resource_refs;
    std::optional<common_agent_working_state> working_state;
    std::string chunk_parent_uri;
    size_t chunk_count = 0;
    std::vector<size_t> completed_chunk_indexes;
};

// Native validation keeps a malformed model-produced checkpoint from being
// treated as a resumable turn.  The checkpoint is still host-owned state.
inline bool common_agent_continuation_checkpoint_valid(
        const common_agent_continuation_checkpoint & checkpoint,
        std::string & error) {
    if (checkpoint.checkpoint_id.empty()) {
        error = "continuation checkpoint requires checkpoint_id";
        return false;
    }
    if (checkpoint.request_id.empty()) {
        error = "continuation checkpoint requires request_id";
        return false;
    }
    if (checkpoint.turn_id.empty()) {
        error = "continuation checkpoint requires turn_id";
        return false;
    }
    if (checkpoint.plan_id.empty()) {
        error = "continuation checkpoint requires plan_id";
        return false;
    }
    if (checkpoint.active_step_id.empty() && checkpoint.next_action.empty()) {
        error = "continuation checkpoint requires active_step_id or next_action";
        return false;
    }
    if (checkpoint.sequence == 0) {
        error = "continuation checkpoint requires a non-zero sequence";
        return false;
    }
    if (!checkpoint.chunk_parent_uri.empty()) {
        if (checkpoint.chunk_count == 0) {
            error = "continuation checkpoint chunk progress requires chunk_count";
            return false;
        }
        for (size_t index : checkpoint.completed_chunk_indexes) {
            if (index >= checkpoint.chunk_count) {
                error = "continuation checkpoint chunk index is outside chunk_count";
                return false;
            }
        }
    } else if (!checkpoint.completed_chunk_indexes.empty() || checkpoint.chunk_count != 0) {
        error = "continuation checkpoint chunk progress requires chunk_parent_uri";
        return false;
    }
    error.clear();
    return true;
}

// A checkpoint may resume only the same externally accepted turn and the
// plan revision it observed.  This is the lane's anti-forking boundary.
inline bool common_agent_continuation_checkpoint_matches(
        const common_agent_continuation_checkpoint & checkpoint,
        const std::string & request_id,
        const std::string & turn_id,
        const common_plan_state & plan,
        std::string & error) {
    if (!common_agent_continuation_checkpoint_valid(checkpoint, error)) {
        return false;
    }
    if (checkpoint.request_id != request_id || checkpoint.turn_id != turn_id) {
        error = "continuation checkpoint belongs to a different turn";
        return false;
    }
    if (checkpoint.plan_id != plan.id || checkpoint.plan_version != plan.version) {
        error = "continuation checkpoint does not match the current plan revision";
        return false;
    }
    error.clear();
    return true;
}
