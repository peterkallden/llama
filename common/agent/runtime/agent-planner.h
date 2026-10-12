#pragma once

#include "agent/agent-generation.h"
#include "agent/contracts/agent-request.h"
#include "plan/plan-types.h"

#include <optional>
#include <string>
#include <vector>

struct common_plan_proposal {
    common_plan_state plan;
    std::vector<common_plan_operation> operations;
    std::optional<common_agent_generated_text_result> generation;
};

class common_planner {
public:
    virtual ~common_planner() = default;
    virtual common_plan_proposal create_plan(
            const common_agent_request & request, std::string & error) = 0;
    virtual common_plan_proposal create_plan_result(
            const common_agent_request & request, std::string & error) {
        return create_plan(request, error);
    }
};
