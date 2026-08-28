#pragma once

#include "agent/contracts/agent-request.h"
#include "agent/contracts/agent-result.h"
#include "plan/plan-types.h"

#include <string>

// Read-only lifecycle seam between the runtime's host-owned learning signals
// and an adaptation transaction sink. Implementations must be bounded and
// must not mutate the active turn or model weights.
class common_agent_adaptation_observer {
public:
    virtual ~common_agent_adaptation_observer() = default;

    // The runtime deliberately ignores observer failures: adaptation is an
    // auxiliary audit path and must never make the user-facing turn fail.
    virtual bool observe(
            const common_agent_request & request,
            const common_plan_state & plan,
            const common_agent_result & result,
            std::string & error) = 0;
};
