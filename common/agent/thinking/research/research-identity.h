#pragma once

#include "agent/agent-scope.h"

#include <string>

std::string common_agent_research_identity(const common_agent_scope & scope);
std::string common_agent_research_draft_identity(
    const std::string & research_id, const std::string & session_id);
