#pragma once

#include "agent/thinking/research/research-contract.h"

#include <nlohmann/json.hpp>
#include <string>

nlohmann::ordered_json common_agent_research_workspace_to_json(
    const common_agent_research_workspace & workspace);
bool common_agent_research_workspace_from_json(
    const nlohmann::json & value,
    common_agent_research_workspace & workspace,
    std::string & error);
