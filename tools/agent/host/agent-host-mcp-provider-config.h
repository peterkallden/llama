#pragma once

#include <string>
#include <vector>

struct agent_host_mcp_provider_config {
    std::string type = "mcp";
    std::string id;
    bool enabled = true;
    std::string transport = "stdio";
    std::vector<std::string> command;
    std::string prefix;
    std::string server_name;
};
