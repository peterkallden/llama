#include "agent-mcp-server-tool-registry.h"

namespace {

const agent_mcp_server_tool * find_tool(
        const std::vector<agent_mcp_server_tool> & tools,
        const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return &tool;
        }
    }
    return nullptr;
}

} // namespace

bool agent_mcp_server_tool_registry::register_tool(
        agent_mcp_server_tool tool,
        std::string & error) {
    if (tool.name.empty()) {
        error = "MCP server tool name is required";
        return false;
    }
    if (!tool.handler) {
        error = "MCP server tool handler is required";
        return false;
    }
    if (find_tool(tools_, tool.name) != nullptr) {
        error = "duplicate MCP server tool: " + tool.name;
        return false;
    }
    tools_.push_back(std::move(tool));
    error.clear();
    return true;
}

std::vector<agent_mcp_server_tool> agent_mcp_server_tool_registry::list_tools() const {
    return tools_;
}

bool agent_mcp_server_tool_registry::call_tool(
        const std::string & name,
        const agent_mcp_json & arguments,
        agent_mcp_server_tool_result & result,
        std::string & error) const {
    result = {};
    const auto * tool = find_tool(tools_, name);
    if (tool == nullptr) {
        error = "unknown MCP server tool: " + name;
        return false;
    }
    if (!arguments.is_object()) {
        error = "MCP tool arguments must be a JSON object";
        return false;
    }
    return tool->handler(arguments, result, error);
}

agent_mcp_json agent_mcp_render_tools_list_result(
        const agent_mcp_server_tool_registry & registry) {
    agent_mcp_json tools = agent_mcp_json::array();
    for (const auto & tool : registry.list_tools()) {
        agent_mcp_json schema = agent_mcp_json::parse(tool.input_schema_json, nullptr, false);
        if (schema.is_discarded() || !schema.is_object()) {
            schema = agent_mcp_json::object();
        }
        tools.push_back({
            {"name", tool.name},
            {"description", tool.description},
            {"inputSchema", std::move(schema)},
            {"annotations", {
                {"readOnlyHint", tool.read_only},
            }},
            {"hostPolicy", {
                {"requiresConfirmation", tool.requires_confirmation},
                {"usesNetwork", tool.uses_network},
                {"writesMemory", tool.writes_memory},
                {"writesPlan", tool.writes_plan},
            }},
        });
    }
    return {{"tools", std::move(tools)}};
}

agent_mcp_json agent_mcp_render_tool_call_result(
        const agent_mcp_server_tool_result & result) {
    agent_mcp_json rendered = {
        {"content", result.content.is_array() ? result.content : agent_mcp_json::array()},
        {"isError", !result.ok},
    };
    if (!result.structured_content.is_null() && !result.structured_content.empty()) {
        rendered["structuredContent"] = result.structured_content;
    }
    if (!result.ok) {
        rendered["errorInfo"] = {
            {"code", result.failure_code.empty() ? "mcp.tool_error" : result.failure_code},
            {"class", result.failure_class.empty() ? "execution" : result.failure_class},
            {"retryable", result.retryable},
            {"safeSummary", result.safe_summary.empty() ? "The MCP server tool failed." : result.safe_summary},
        };
    }
    return rendered;
}
