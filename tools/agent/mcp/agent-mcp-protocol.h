#pragma once

#include "../tooling/agent-tool-provider.h"

#include <nlohmann/json.hpp>

#include <string>

nlohmann::ordered_json make_mcp_jsonrpc_request(
    int id,
    const std::string & method,
    const nlohmann::ordered_json & params);

nlohmann::ordered_json make_mcp_jsonrpc_notification(
    const std::string & method,
    const nlohmann::ordered_json & params);

nlohmann::ordered_json make_mcp_initialize_params();

nlohmann::ordered_json make_mcp_tools_call_params(
    const std::string & name,
    const nlohmann::ordered_json & arguments);

nlohmann::ordered_json make_mcp_resources_read_params(
    const std::string & uri);

bool parse_mcp_tool_definition(
    const std::string & provider_id,
    const nlohmann::ordered_json & item,
    mcp_agent_tool_definition & definition,
    std::string & error);

bool parse_mcp_tool_call_result(
    const nlohmann::ordered_json & rpc_result,
    mcp_agent_tool_call_result & result,
    std::string & error);

bool parse_mcp_resources_list_result(
    const nlohmann::ordered_json & rpc_result,
    std::vector<mcp_agent_resource_definition> & resources,
    std::string & error);

bool parse_mcp_resource_read_result(
    const nlohmann::ordered_json & rpc_result,
    mcp_agent_resource_read_result & result,
    std::string & error);
