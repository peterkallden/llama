#pragma once

#include "../daemon/agent-daemon-events.h"
#include "agent-mcp-server-tool-registry.h"
#include "agent-mcp-auth.h"
#include "agent-mcp-task.h"

#include <cpp-httplib/httplib.h>

#include <functional>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <memory>

struct agent_mcp_http_server_options {
    std::string listen_address = "127.0.0.1";
    int port = 0;
    std::string path = "/mcp";
    std::string allowed_origin;
    std::string bearer_token;
    std::shared_ptr<const agent_mcp_authenticator> authenticator;
    agent_mcp_caller_policy default_policy;
    size_t max_body_bytes = 1024 * 1024;
    size_t max_result_bytes = 1024 * 1024;
    std::string server_name = "llama-agent-mcp";
    std::string server_version = "0.1";
    std::string protocol_version = "2025-06-18";
    std::function<bool(agent_mcp_json & result, std::string & error)> list_resources;
    std::function<bool(const agent_mcp_json & params, agent_mcp_json & result, std::string & error)> read_resource;
    std::function<bool(
        const agent_mcp_caller_policy & policy,
        const std::string & tool_name,
        const agent_mcp_json & arguments,
        agent_mcp_server_tool_result & result,
        std::string & error)> execute_tool;
    bool agent_tools_enabled = false;
    size_t max_delegation_depth = 1;
    std::function<bool(
        const agent_mcp_caller_policy & policy,
        const std::string & operation,
        const agent_mcp_json & arguments,
        agent_mcp_server_tool_result & result,
        std::string & error)> execute_agent_tool;
    std::function<std::string(common_agent_event_stream_subscription)> subscribe_events;
    std::function<void(const std::string &)> unsubscribe_events;
    std::function<common_agent_event_stream_wait_status(
        const std::string &,
        common_agent_event_stream_delivery &,
        std::chrono::milliseconds)> wait_for_event;
    bool tasks_enabled = false;
    uint64_t task_ttl_ms = 60000;
    uint64_t task_poll_interval_ms = 250;
};

class agent_mcp_http_server {
public:
    agent_mcp_http_server(
        agent_mcp_server_tool_registry registry,
        agent_mcp_http_server_options options = {});

    bool bind(std::string & error);
    bool listen(std::string & error);
    bool replace_registry(agent_mcp_server_tool_registry registry, std::string & error);
    void replace_authenticator(std::shared_ptr<const agent_mcp_authenticator> authenticator);
    void replace_default_policy(agent_mcp_caller_policy policy);
    void stop();
    int port() const { return port_; }

private:
    bool authorize(const httplib::Request & request, httplib::Response & response, agent_mcp_caller_policy & policy) const;
    bool validate_session(const httplib::Request & request, httplib::Response & response, agent_mcp_caller_policy & policy) const;
    bool handle_message(const agent_mcp_json & message, const agent_mcp_caller_policy & policy, agent_mcp_json & response, std::string & error);
    std::string make_session_id();
    void install_routes();

    agent_mcp_server_tool_registry registry_;
    mutable std::mutex registry_mutex_;
    mutable std::mutex policy_mutex_;
    agent_mcp_http_server_options options_;
    httplib::Server server_;
    std::unique_ptr<agent_mcp_task_store> tasks_;
    int port_ = 0;
    mutable std::mutex session_mutex_;
    struct session_state {
        agent_mcp_caller_policy policy;
        std::string protocol_version;
    };
    std::unordered_map<std::string, session_state> sessions_;
    uint64_t next_session_id_ = 1;
};
