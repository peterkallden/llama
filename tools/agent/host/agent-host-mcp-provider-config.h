#pragma once

#include <string>
#include <vector>

struct agent_host_mcp_provider_config {
    std::string type = "mcp";
    std::string id;
    bool enabled = true;
    std::string transport = "stdio";
    std::vector<std::string> command;
    std::string url;
    std::string token_env;
    std::vector<std::string> allowed_tools;
    uint32_t connect_timeout_ms = 0;
    uint32_t request_timeout_ms = 0;
    uint32_t shutdown_timeout_ms = 0;
    size_t max_result_bytes = 0;
    std::string prefix;
    std::string server_name;
};
