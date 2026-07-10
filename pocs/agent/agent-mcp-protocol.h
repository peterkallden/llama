#pragma once

#include "agent-tool-provider.h"

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

bool parse_mcp_tool_definition(
    const std::string & provider_id,
    const nlohmann::ordered_json & item,
    mcp_agent_tool_definition & definition,
    std::string & error);

bool parse_mcp_tool_call_result(
    const nlohmann::ordered_json & rpc_result,
    mcp_agent_tool_call_result & result,
    std::string & error);
