#pragma once

#include "tools/agent/tooling/agent-tool-provider.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct agent_mcp_client_factory_request {
    std::string server_name = "mcp";
    std::string transport = "stdio";
    std::vector<std::string> command_line;
    std::string url;
    std::string bearer_token;
    std::vector<std::string> allowed_tools;
    uint32_t connect_timeout_ms = 5000;
    uint32_t request_timeout_ms = 30000;
    uint32_t shutdown_timeout_ms = 2000;
    size_t max_result_bytes = 1024 * 1024;
    std::string credential_ref;
    std::shared_ptr<common_agent_credential_provider> credential_provider;
};

std::unique_ptr<agent_mcp_tool_client> make_agent_mcp_client(
        const agent_mcp_client_factory_request & request,
        std::string & error);
