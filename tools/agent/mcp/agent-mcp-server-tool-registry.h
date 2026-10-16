#pragma once

#include "agent-mcp-server-protocol.h"
#include "chat.h"

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
    std::string output_schema_json;
    agent_mcp_json ui_metadata = agent_mcp_json::object();
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
agent_mcp_json agent_mcp_canonical_output_schema(
    const std::string & result_schema_json);

// Injects the current native model-facing tool view into the MCP registry.
// Execution remains owned by the daemon dispatcher; the registry handler is
// only a fallback for direct registry calls without that dispatcher seam.
bool agent_mcp_register_native_tool_view(
        const std::vector<common_chat_tool> & tools,
        const std::function<bool(const std::string &)> & is_read_only,
        const std::function<bool(const std::string &)> & is_policy_gated,
        agent_mcp_server_tool_registry & registry,
        std::string & error);

agent_mcp_json agent_mcp_render_tool_call_result(
        const agent_mcp_server_tool_result & result);
void agent_mcp_set_tool_call_result(
        agent_mcp_server_tool_result & result,
        bool ok,
        agent_mcp_json structured_content,
        std::string safe_summary,
        std::string failure_code,
        std::string failure_class,
        bool retryable,
        std::string raw_diagnostic,
        agent_mcp_json content = agent_mcp_json::array());

// Small first-party MCP Apps are shipped as MCP UI resources, not ordinary
// model-facing tools. The resource URI is also attached to its owning tools.
agent_mcp_json agent_mcp_first_party_app_resources();
bool agent_mcp_read_first_party_app_resource(
        const std::string & uri,
        agent_mcp_json & content,
        std::string & error);
agent_mcp_json agent_mcp_first_party_app_metadata(const std::string & tool_name);
