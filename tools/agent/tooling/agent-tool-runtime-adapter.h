#pragma once

#include "../tooling/agent-tool-provider.h"
#include "agent/agent-runtime.h"
#include "../runtime/agent-route-compiler.h"

#include <memory>

std::unique_ptr<common_agent_tool_runtime> make_provider_agent_tool_runtime(
        agent_tool_view & tool_view);

// Wraps the provider runtime with the selected route's phase-aware authority
// fence.  The provider remains the implementation source; this wrapper is
// the single enforcement point used by planning, reflection, research and
// synchronous/asynchronous execution.
std::unique_ptr<common_agent_tool_runtime> make_scoped_agent_tool_runtime(
        std::unique_ptr<common_agent_tool_runtime> provider_runtime,
        common_agent_execution_envelope envelope,
        common_agent_execution_phase initial_phase = common_agent_execution_phase::normal);
