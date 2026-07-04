#pragma once

#include "agent/agent-contract.h"
#include "agent/agent-inference.h"

#include <functional>
#include <string>
#include <vector>

struct common_agent_chat_runtime_policy {
    size_t max_tool_rounds = 1;
};

class agent_tool_view;

using common_agent_chat_tool_handler = std::function<std::string(const common_chat_tool_call & call)>;

struct common_agent_chat_runtime_execution {
    common_agent_inference & inference;
    common_agent_request request;
    common_agent_generation_options generation_options;
    common_agent_chat_runtime_policy policy;
    const std::vector<common_chat_tool> & tools;
    bool profile_tools_active = false;
    agent_tool_view * tool_view = nullptr;
    common_agent_chat_tool_handler tool_handler;
};

bool run_agent_chat_runtime(
    common_agent_chat_runtime_execution & execution,
    common_agent_result & result,
    std::string & error);
