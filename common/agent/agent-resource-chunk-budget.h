#pragma once

#include "agent-context-budgets.h"
#include "agent-inference.h"
#include "resource/resource-chunker.h"

#include <string>

// Resolve the byte policy once per resource evidence session. A configured
// maximum is a ceiling; zero selects a bounded context-derived target.
bool common_agent_resolve_resource_chunk_policy(
    common_agent_inference & inference,
    const common_agent_context_budget_config & config,
    const std::string & task,
    common_runtime_resource_chunk_policy & policy,
    std::string & error);
