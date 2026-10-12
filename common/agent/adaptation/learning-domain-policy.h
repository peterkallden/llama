#pragma once

#include "agent/contracts/agent-request.h"
#include "agent/contracts/agent-result.h"
#include "plan/plan-types.h"

#include <map>
#include <string>

// One policy covers all tool transports. A family override narrows or
// enables tool-use collection for a canonical family such as diagnostics,
// independent of whether the tool was native, MCP, or OpenAPI backed.
struct common_learning_domain_policy {
    bool configured = false;
    bool planning = false;
    bool tool_use = false;
    bool research = false;
    bool procedure_learning = false;
    std::map<std::string, bool> tool_use_families;
};

bool common_learning_domain_policy_allows(
        const common_learning_domain_policy & policy,
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result);
