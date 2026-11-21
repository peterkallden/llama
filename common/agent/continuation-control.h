#pragma once

#include "agent/contracts/agent-recovery.h"
#include "plan/plan-types.h"
#include "plan/plan-policy.h"
#include <cstddef>
#include <string>
#include <vector>

// Request-local host accounting. This grants no tools, route transitions or
// additional execution budget, and is never accepted from model output.
struct common_agent_continuation_budget {
    size_t executions_remaining = 0;
    size_t tool_calls_remaining = 0;
    size_t revisions_remaining = 0;
    size_t new_steps_remaining = 0;
};

bool common_agent_continuation_allows_same_call(const common_plan_state &, const std::string & step_id);
bool common_agent_continuation_can_plan(const common_agent_continuation_budget &, std::string & error);
bool common_agent_admit_continuation(const common_agent_continuation_budget &,
    const std::vector<common_plan_operation> &, std::string & error);
bool common_agent_validate_continuation_retries(const common_plan_state &,
    const std::vector<common_plan_operation> &, std::string & error);
std::string common_agent_render_continuation_context(const common_plan_state &,
    const common_agent_continuation_budget &);

// Reserve durable evidence and control slots before a workflow's side effects.
bool common_agent_workflow_storage_available(const common_plan_state &,
    const common_plan_policy_config &, std::string & error);
