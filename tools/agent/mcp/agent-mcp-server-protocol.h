#pragma once

#include <cstdio>
#include <nlohmann/json.hpp>
#include <string>

using agent_mcp_json = nlohmann::ordered_json;

bool agent_mcp_read_json_rpc_message(
    FILE * stream,
    agent_mcp_json & message,
    std::string & error);

bool agent_mcp_write_json_rpc_message(
    FILE * stream,
    const agent_mcp_json & message,
    std::string & error);

bool agent_mcp_write_malformed_json_rpc_result(
    FILE * stream,
    const agent_mcp_json & id,
    std::string & error);

agent_mcp_json agent_mcp_make_json_rpc_result(
    const agent_mcp_json & id,
    agent_mcp_json result);

agent_mcp_json agent_mcp_make_json_rpc_error(
    const agent_mcp_json & id,
    int code,
    std::string message);
