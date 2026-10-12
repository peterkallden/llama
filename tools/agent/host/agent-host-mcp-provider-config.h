#pragma once

#include <string>
#include <vector>

#include "agent-host-provider-auth-config.h"

struct agent_host_mcp_provider_config {
    std::string type = "mcp";
    std::string id;
    bool enabled = true;
    bool required = false;
    std::string transport = "stdio";
    std::vector<std::string> command;
    std::string url;
    agent_host_provider_auth_config auth;
    std::vector<std::string> allowed_tools;
    uint32_t connect_timeout_ms = 0;
    uint32_t request_timeout_ms = 0;
    uint32_t shutdown_timeout_ms = 0;
    size_t max_result_bytes = 0;
    std::string prefix;
    std::string server_name;
};

struct agent_host_mcp_inbound_token_config {
    std::string id;
    std::string token_env;
    std::string audience = "llama-agent";
    std::string namespace_id = "local";
    std::string project_id;
    std::string tool_profile;
    std::vector<std::string> allowed_tools;
    bool allow_writes = false;
    bool allow_admin = false;
};
