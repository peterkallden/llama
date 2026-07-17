#pragma once

#include "agent-mcp-server-tool-registry.h"

#include <cpp-httplib/httplib.h>

#include <functional>
#include <mutex>
#include <string>
#include <unordered_set>

struct agent_mcp_http_server_options {
    std::string listen_address = "127.0.0.1";
    int port = 0;
    std::string path = "/mcp";
    std::string allowed_origin;
    std::string bearer_token;
    size_t max_body_bytes = 1024 * 1024;
    size_t max_result_bytes = 1024 * 1024;
    std::string server_name = "llama-agent-mcp";
    std::string server_version = "0.1";
    std::string protocol_version = "2024-11-05";
    std::function<bool(agent_mcp_json & result, std::string & error)> list_resources;
    std::function<bool(const agent_mcp_json & params, agent_mcp_json & result, std::string & error)> read_resource;
};

class agent_mcp_http_server {
public:
    agent_mcp_http_server(
        agent_mcp_server_tool_registry registry,
        agent_mcp_http_server_options options = {});

    bool bind(std::string & error);
    bool listen(std::string & error);
    void stop();
    int port() const { return port_; }

private:
    bool authorize(const httplib::Request & request, httplib::Response & response) const;
    bool validate_session(const httplib::Request & request, httplib::Response & response) const;
    bool handle_message(const agent_mcp_json & message, agent_mcp_json & response, std::string & error);
    std::string make_session_id();
    void install_routes();

    agent_mcp_server_tool_registry registry_;
    agent_mcp_http_server_options options_;
    httplib::Server server_;
    int port_ = 0;
    mutable std::mutex session_mutex_;
    std::unordered_set<std::string> sessions_;
    uint64_t next_session_id_ = 1;
};
