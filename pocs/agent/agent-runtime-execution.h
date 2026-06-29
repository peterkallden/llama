#pragma once

#include "../common/cli-config.h"

#include "agent/agent-contract.h"
#include "agent/agent-inference.h"
#include "agent/blueprint-selector.h"
#include "agent/tool-registry.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <string>
#include <vector>

struct common_agent_cli_runtime_execution {
    common_memory_store & memory_store;
    common_plan_store & plan_store;
    common_agent_inference & inference;
    args & options;
    const common_agent_scope & scope;
    const std::vector<common_blueprint_candidate> & installed_blueprint_candidates;
    const std::vector<common_memory_hit> & memories;
    common_memory_scope memory_scope = common_memory_scope::session;
    bool memory_enabled = false;
    const std::vector<common_chat_tool> & tools;
    bool profile_tools_active = false;
    const common_tool_registry * tool_registry = nullptr;
};

bool run_agent_cli_mini_runtime(
    common_agent_cli_runtime_execution & execution,
    common_agent_result & result,
    std::string & error);
