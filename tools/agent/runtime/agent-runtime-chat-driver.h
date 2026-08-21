#pragma once

#include "agent/contracts/agent-request.h"
#include "agent/contracts/agent-result.h"
#include "agent/agent-inference.h"
#include "../runtime/agent-runtime-tooling.h"

#include <string>
#include <vector>

struct common_agent_chat_runtime_policy {
    size_t max_tool_rounds = 16;
    size_t max_continuations = 2;
};

struct common_agent_chat_runtime_execution {
    common_agent_inference & inference;
    common_agent_request request;
    common_agent_generation_options generation_options;
    common_agent_chat_runtime_policy policy;
    const common_agent_runtime_tooling & tooling;
};

bool run_agent_chat_runtime(
    common_agent_chat_runtime_execution & execution,
    common_agent_result & result,
    std::string & error);
