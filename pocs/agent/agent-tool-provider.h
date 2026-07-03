#pragma once

#include "agent/agent-scope.h"
#include "agent/tool-adapters.h"
#include "agent/tool-catalog.h"
#include "agent/tool-chat-bridge.h"
#include "chat.h"
#include "plan/plan-types.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct agent_tool_context {
    std::string request_id;
    std::string turn_id;

    common_agent_scope scope;
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;

    std::string profile_id = "minimal";
    std::string repository_root;

    bool allow_network = false;
    bool allow_policy_gated_writes = false;
    bool allow_memory_proposals = false;
    bool allow_plan_proposals = false;

    size_t max_calls = 4;
    uint32_t default_timeout_ms = 1000;
    size_t default_max_result_bytes = 16 * 1024;
};

struct agent_tool_call {
    std::string id;
    std::string name;
    std::string arguments_json = "{}";
};

struct agent_tool_result {
    bool ok = false;

    std::string tool_call_id;
    std::string tool_name;
    std::string content_json;

    std::string failure_code;
    common_tool_failure_class failure_class = common_tool_failure_class::execution;
    bool retryable = false;
    std::string safe_summary;
    std::string raw_diagnostic;
};

class agent_tool_view {
public:
    virtual ~agent_tool_view() = default;

    virtual const std::vector<common_chat_tool> & chat_tools() const = 0;

    virtual agent_tool_result call(
        const agent_tool_call & call,
        std::string & error) = 0;
};

class agent_tool_provider {
public:
    virtual ~agent_tool_provider() = default;

    virtual std::unique_ptr<agent_tool_view> resolve_tools(
        const agent_tool_context & context,
        std::string & error) = 0;
};

class native_agent_tool_provider : public agent_tool_provider {
public:
    using binding_factory = std::function<bool(
        const agent_tool_context & context,
        common_native_tool_bindings & bindings,
        std::string & error)>;

    native_agent_tool_provider(
        const common_tool_catalog & catalog,
        binding_factory make_bindings);

    std::unique_ptr<agent_tool_view> resolve_tools(
        const agent_tool_context & context,
        std::string & error) override;

private:
    const common_tool_catalog & catalog;
    binding_factory make_bindings;
};

bool agent_dispatch_chat_tool_calls(
    common_chat_msg & assistant_message,
    agent_tool_view & tool_view,
    size_t max_calls,
    common_tool_chat_dispatch_result & result,
    std::string & error);
