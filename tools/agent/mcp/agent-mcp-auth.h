#pragma once

#include <string>
#include <chrono>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
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
    bool allow_admin = false;
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

// Projects caller policy onto the MCP-visible tool metadata. The native
// runtime remains the final execution authority; this helper keeps listing
// and inbound dispatch aligned with that authority.
bool agent_mcp_policy_allows_tool(
    const agent_mcp_caller_policy & policy,
    const std::string & tool_name,
    bool read_only,
    bool requires_confirmation);

struct agent_mcp_jwt_authenticator_options {
    std::string issuer;
    std::string audience;
    std::string jwks_uri;
    std::vector<std::string> allowed_algorithms = {"RS256"};
    std::vector<std::string> required_scopes;
    agent_mcp_caller_policy policy_template;
    std::string subject_claim = "sub";
    uint32_t clock_skew_seconds = 30;
    uint32_t jwks_cache_seconds = 300;
};

class agent_mcp_jwt_authenticator final : public agent_mcp_authenticator {
public:
    explicit agent_mcp_jwt_authenticator(agent_mcp_jwt_authenticator_options options);

    bool authenticate(
        const agent_mcp_authentication_request & request,
        agent_mcp_caller_policy & policy,
        std::string & error) const override;

private:
    bool refresh_jwks(std::string & error) const;
    bool verify_signature(
        const nlohmann::ordered_json & header,
        const std::string & signing_input,
        const std::string & signature,
        std::string & error) const;

    agent_mcp_jwt_authenticator_options options_;
    mutable std::mutex mutex_;
    mutable nlohmann::ordered_json jwks_;
    mutable std::chrono::steady_clock::time_point jwks_loaded_at_{};
};
