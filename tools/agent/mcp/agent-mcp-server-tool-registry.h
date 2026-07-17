#pragma once

#include "agent-mcp-server-protocol.h"

#include <functional>
#include <string>
#include <vector>

struct agent_mcp_server_tool_result {
    bool ok = true;
    agent_mcp_json structured_content = agent_mcp_json::object();
    agent_mcp_json content = agent_mcp_json::array();
    std::string failure_code;
    std::string failure_class = "execution";
    bool retryable = false;
    std::string safe_summary;
    std::string raw_diagnostic;
};

struct agent_mcp_server_tool {
    std::string name;
    std::string description;
    std::string input_schema_json = R"({"type":"object"})";

    bool read_only = true;
    bool requires_confirmation = false;
    bool uses_network = false;
    bool writes_memory = false;
    bool writes_plan = false;

    std::function<bool(
        const agent_mcp_json & arguments,
        agent_mcp_server_tool_result & result,
        std::string & error)> handler;
};

class agent_mcp_server_tool_registry {
public:
    bool register_tool(agent_mcp_server_tool tool, std::string & error);
    std::vector<agent_mcp_server_tool> list_tools() const;
    bool contains_tool(const std::string & name) const;
    bool validate_tool_arguments(
        const std::string & name,
        const agent_mcp_json & arguments,
        std::string & error) const;

    bool call_tool(
        const std::string & name,
        const agent_mcp_json & arguments,
        agent_mcp_server_tool_result & result,
        std::string & error) const;

private:
    std::vector<agent_mcp_server_tool> tools_;
};

agent_mcp_json agent_mcp_render_tools_list_result(
    const agent_mcp_server_tool_registry & registry);

agent_mcp_json agent_mcp_render_tool_call_result(
    const agent_mcp_server_tool_result & result);
