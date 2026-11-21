#pragma once

#include "tools/agent/tooling/agent-tool-provider.h"
#include "resource/resource-contract.h"

// Materialize only resource links returned by an authorized MCP tool call.
// The result is updated only after every linked resource was fully acquired.
bool agent_materialize_mcp_resources(agent_mcp_tool_client & client,
        agent_resource_store & store, const agent_tool_context & context,
        const std::string & provider_id, const std::string & source_tool,
        mcp_agent_tool_call_result & result, std::string & error);
