#pragma once

#include "agent/contracts/agent-request.h"
#include "agent/contracts/agent-result.h"
#include "agent/agent-inference.h"
#include "agent/tool-family-index.h"
#include "agent-runtime-control.h"
#include "../runtime/agent-runtime-tooling.h"

#include <string>
#include <vector>

struct common_agent_chat_runtime_policy {
    size_t max_tool_rounds = 16;
    size_t max_continuations = 2;
    bool enable_tool_family_routing = false;
};

struct common_agent_chat_runtime_execution {
    common_agent_inference & inference;
    common_agent_request request;
    common_agent_generation_options generation_options;
    common_agent_chat_runtime_policy policy;
    const common_agent_runtime_tooling & tooling;
    common_agent_runtime_execution_control execution_control;
};

bool run_agent_chat_runtime(
    common_agent_chat_runtime_execution & execution,
    common_agent_result & result,
    std::string & error);

// Shared intent selection; input families come exclusively from the authorized tool view.
bool common_agent_select_tool_families(
    common_agent_inference & inference,
    const common_agent_request & request,
    const common_agent_generation_options & generation_options,
    const std::vector<common_tool_family_index> & families,
    bool required_family_selection,
    common_tool_family_selection & selection,
    std::string & error);
