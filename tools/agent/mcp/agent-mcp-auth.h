#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct agent_mcp_caller_policy {
    std::string caller_id;
    std::string audience;
    std::string namespace_id;
    std::string project_id;
    std::string tool_profile;
    std::vector<std::string> allowed_tools;
    bool allow_writes = false;
};

struct agent_mcp_authentication_request {
    std::string authorization;
};

class agent_mcp_authenticator {
public:
    virtual ~agent_mcp_authenticator() = default;

    virtual bool authenticate(
        const agent_mcp_authentication_request & request,
        agent_mcp_caller_policy & policy,
        std::string & error) const = 0;
};

class agent_mcp_opaque_token_authenticator final : public agent_mcp_authenticator {
public:
    bool register_token(
        std::string token,
        agent_mcp_caller_policy policy,
        std::string & error);

    bool authenticate(
        const agent_mcp_authentication_request & request,
        agent_mcp_caller_policy & policy,
        std::string & error) const override;

private:
    struct token_entry {
        std::string token;
        agent_mcp_caller_policy policy;
    };
    std::vector<token_entry> entries_;
};

bool agent_mcp_policy_allows_tool(
    const agent_mcp_caller_policy & policy,
    const std::string & tool_name);
