#include "agent-mcp-auth.h"

#include <algorithm>

bool agent_mcp_opaque_token_authenticator::register_token(
        std::string token,
        agent_mcp_caller_policy policy,
        std::string & error) {
    if (token.empty()) {
        error = "MCP auth token must not be empty";
        return false;
    }
    if (policy.caller_id.empty() || policy.audience.empty() ||
            policy.namespace_id.empty() || policy.tool_profile.empty()) {
        error = "MCP caller policy requires caller_id, audience, namespace_id and tool_profile";
        return false;
    }
    for (const auto & entry : entries_) {
        if (entry.token == token || entry.policy.caller_id == policy.caller_id) {
            error = "duplicate MCP auth token or caller_id";
            return false;
        }
    }
    entries_.push_back({std::move(token), std::move(policy)});
    error.clear();
    return true;
}

bool agent_mcp_opaque_token_authenticator::authenticate(
        const agent_mcp_authentication_request & request,
        agent_mcp_caller_policy & policy,
        std::string & error) const {
    constexpr const char * prefix = "Bearer ";
    if (request.authorization.rfind(prefix, 0) != 0 ||
            request.authorization.size() <= std::char_traits<char>::length(prefix)) {
        error = "Bearer authentication is required";
        return false;
    }
    const std::string token = request.authorization.substr(std::char_traits<char>::length(prefix));
    for (const auto & entry : entries_) {
        if (entry.token == token) {
            policy = entry.policy;
            error.clear();
            return true;
        }
    }
    error = "Bearer token is not authorized";
    return false;
}

bool agent_mcp_policy_allows_tool(
        const agent_mcp_caller_policy & policy,
        const std::string & tool_name) {
    return policy.allowed_tools.empty() ||
        std::find(policy.allowed_tools.begin(), policy.allowed_tools.end(), tool_name) != policy.allowed_tools.end();
}

bool agent_mcp_policy_allows_tool(
        const agent_mcp_caller_policy & policy,
        const std::string & tool_name,
        bool read_only,
        bool requires_confirmation) {
    if (!agent_mcp_policy_allows_tool(policy, tool_name)) {
        return false;
    }
    if (policy.allow_writes) {
        return true;
    }
    // A caller without write authority may only see and invoke read-only
    // tools. Confirmation metadata does not grant authority by itself.
    return read_only && !requires_confirmation;
}
