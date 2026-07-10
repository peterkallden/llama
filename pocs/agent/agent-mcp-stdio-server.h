#pragma once

#include "agent-mcp-server-tool-registry.h"

#include <cstdio>
#include <string>

struct agent_mcp_stdio_server_options {
    std::string server_name = "llama-agent-mcp";
    std::string server_version = "0.1";
    std::string protocol_version = "2024-11-05";
    bool emit_malformed_tools_list = false;
    bool exit_after_initialize = false;
};

class agent_mcp_stdio_server {
public:
    explicit agent_mcp_stdio_server(
        agent_mcp_server_tool_registry registry,
        agent_mcp_stdio_server_options options = {});

    int run(FILE * input, FILE * output, FILE * diagnostics);

private:
    agent_mcp_server_tool_registry registry_;
    agent_mcp_stdio_server_options options_;
};
