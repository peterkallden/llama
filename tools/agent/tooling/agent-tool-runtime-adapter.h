#pragma once

#include "../tooling/agent-tool-provider.h"
#include "agent/agent-runtime.h"

#include <memory>

std::unique_ptr<common_agent_tool_runtime> make_provider_agent_tool_runtime(
    agent_tool_view & tool_view);
