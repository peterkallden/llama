#pragma once

#include "agent/agent-contract.h"
#include "agent/research/research-contract.h"

bool common_agent_research_create_workspace(
        const common_agent_request & request,
        common_agent_research_workspace & workspace,
        std::string & error);
