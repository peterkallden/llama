#pragma once

#include "agent-mcp-server-tool-registry.h"
#include "agent-mcp-auth.h"

#include <string>

struct agent_mcp_agent_tool_options {
    size_t max_task_bytes = 4096;
    size_t max_result_bytes = 16384;
    size_t max_delegation_depth = 1;
};

bool agent_mcp_register_agent_tools(
    agent_mcp_server_tool_registry & registry,
    agent_mcp_agent_tool_options options,
    std::string & error);

bool agent_mcp_is_agent_tool(const std::string & name);
