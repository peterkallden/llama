#pragma once
#include "agent-failures.h"

// Technical recovery disposition. It neither selects a source nor authorizes
// execution; the host's route, retry and budget policies still apply.
enum class common_agent_recovery_action { retry, repair_arguments, inspect, blocked };

inline common_agent_recovery_action common_agent_recovery_for_failure(common_agent_failure_class kind, bool retryable) {
    switch (kind) {
        case common_agent_failure_class::validation:
        case common_agent_failure_class::limit: return common_agent_recovery_action::repair_arguments;
        case common_agent_failure_class::not_found:
        case common_agent_failure_class::plan_conflict: return common_agent_recovery_action::inspect;
        case common_agent_failure_class::policy: return common_agent_recovery_action::blocked;
        default: return retryable ? common_agent_recovery_action::retry : common_agent_recovery_action::blocked;
    }
}

inline const char * common_agent_recovery_action_name(common_agent_recovery_action action) {
    switch (action) {
        case common_agent_recovery_action::retry: return "retry";
        case common_agent_recovery_action::repair_arguments: return "repair_arguments";
        case common_agent_recovery_action::inspect: return "inspect";
        case common_agent_recovery_action::blocked: return "blocked";
    }
    return "blocked";
}
