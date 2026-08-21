#pragma once

#include "agent/agent-scope.h"
#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/bridge/tool-chat-bridge.h"
#include "../../../common/runtime/runtime-operation.h"
#include "../runtime/agent-runtime-control.h"
#include "chat.h"
#include "plan/plan-types.h"
#include "runtime-resource.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct common_plan_tool_dataflow_contract;
struct common_agent_tool_repair_context;

struct agent_tool_context {
    std::string request_id;
    std::string turn_id;

    common_agent_scope scope;
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;

    std::string profile_id = "minimal";
    std::shared_ptr<const common_tool_profile_snapshot> profile_snapshot;
    std::string repository_root;
    // Model-visible tool names allowed for this resolved runtime view. Native
    // tools use their definition name directly; MCP tools use their exposed
    // name after any provider prefixing.
    std::vector<std::string> allowed_exposed_tool_names;
    std::vector<std::string> async_exposed_tool_names;

    bool allow_network = false;
    bool allow_policy_gated_writes = false;
    bool allow_memory_proposals = false;
    bool allow_plan_proposals = false;

    size_t max_calls = 4;
    uint32_t default_timeout_ms = 1000;
    size_t default_max_result_bytes = 16 * 1024;
    common_agent_runtime_execution_control execution_control;
};

class agent_embedding_provider {
public:
    virtual ~agent_embedding_provider() = default;

    virtual bool embed(
        const std::string & purpose,
        const std::string & text,
        std::vector<float> & embedding,
        std::string & error) = 0;
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
    std::string content_summary;
    std::vector<common_runtime_resource_ref> resource_refs;

    std::string failure_code;
    common_tool_failure_class failure_class = common_tool_failure_class::execution;
    bool retryable = false;
    std::string safe_summary;
    std::string raw_diagnostic;
};

using agent_tool_pending_call = common_runtime_operation_ref;

class agent_tool_view {
public:
    virtual ~agent_tool_view() = default;

    virtual const std::vector<common_chat_tool> & chat_tools() const = 0;
    virtual bool describe_tool_dataflow(
        const std::string &, common_plan_tool_dataflow_contract &, std::string &) const { return false; }
    virtual common_agent_tool_repair_context make_repair_context(
        const std::string &, const std::string &, const std::string &) const;
    virtual bool exposes_tool(const std::string & name) const = 0;
    virtual bool is_read_only(const std::string & name) const = 0;
    virtual bool is_policy_gated(const std::string & name) const = 0;
    virtual bool validate(const agent_tool_call & call, std::string & error) const = 0;

    virtual agent_tool_result call(
        const agent_tool_call & call,
        std::string & error) = 0;

    virtual bool supports_async_call(const std::string & name) const = 0;

    virtual bool begin_call_async(
        const agent_tool_call & call,
        agent_tool_pending_call & pending,
        std::string & error) = 0;

    virtual bool poll_call_async(
        const agent_tool_pending_call & pending,
        bool & ready,
        agent_tool_result & result,
        std::string & error) = 0;

    virtual bool cancel_call_async(
        const agent_tool_pending_call & pending,
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

struct mcp_agent_tool_definition {
    std::string provider_id;
    std::string name;
    std::string description;
    std::string input_schema_json = R"({"type":"object"})";
    bool read_only = true;
    bool requires_confirmation = false;
    bool uses_network = false;
    bool writes_memory = false;
    bool writes_plan = false;
    size_t max_result_bytes = 16 * 1024;
};

struct mcp_agent_tool_call_result {
    bool ok = false;
    std::string structured_content_json;
    std::string text_content;
    std::vector<common_runtime_resource_ref> resource_refs;
    std::string failure_code;
    common_tool_failure_class failure_class = common_tool_failure_class::execution;
    bool retryable = false;
    std::string safe_summary;
    std::string raw_diagnostic;
};

struct mcp_agent_resource_definition {
    common_runtime_resource_ref resource;
};

struct mcp_agent_resource_read_result {
    common_runtime_resource_ref resource;
    std::string text_content;
};

class agent_mcp_tool_client {
public:
    virtual ~agent_mcp_tool_client() = default;

    virtual bool list_tools(
        const agent_tool_context & context,
        std::vector<mcp_agent_tool_definition> & tools,
        std::string & error) = 0;

    virtual bool call_tool(
        const agent_tool_context & context,
        const mcp_agent_tool_definition & tool,
        const std::string & arguments_json,
        mcp_agent_tool_call_result & result,
        std::string & error) = 0;
};

class mcp_agent_tool_provider : public agent_tool_provider {
public:
    mcp_agent_tool_provider(
        std::string provider_id,
        agent_mcp_tool_client & client,
        std::string exposed_name_prefix = {});

    std::unique_ptr<agent_tool_view> resolve_tools(
        const agent_tool_context & context,
        std::string & error) override;

private:
    std::string provider_id;
    agent_mcp_tool_client & client;
    std::string exposed_name_prefix;
};

class composite_agent_tool_provider : public agent_tool_provider {
public:
    composite_agent_tool_provider() = default;

    void add_provider(agent_tool_provider & provider);

    std::unique_ptr<agent_tool_view> resolve_tools(
        const agent_tool_context & context,
        std::string & error) override;

private:
    std::vector<agent_tool_provider *> providers;
};

struct agent_mcp_stdio_client_config {
    std::string server_name;
    std::vector<std::string> command_line;
    std::map<std::string, std::string> environment;
    uint32_t request_timeout_ms = 30000;
    uint32_t shutdown_timeout_ms = 1000;
};

class agent_mcp_stdio_client : public agent_mcp_tool_client {
public:
    explicit agent_mcp_stdio_client(agent_mcp_stdio_client_config config);
    ~agent_mcp_stdio_client() override;

    bool list_tools(
        const agent_tool_context & context,
        std::vector<mcp_agent_tool_definition> & tools,
        std::string & error) override;

    bool call_tool(
        const agent_tool_context & context,
        const mcp_agent_tool_definition & tool,
        const std::string & arguments_json,
        mcp_agent_tool_call_result & result,
        std::string & error) override;

    bool list_resources(
        std::vector<mcp_agent_resource_definition> & resources,
        std::string & error);

    bool read_resource(
        const std::string & uri,
        mcp_agent_resource_read_result & result,
        std::string & error);

private:
    bool ensure_started(
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline = {});
    bool send_notification(const std::string & method, const nlohmann::ordered_json & params, std::string & error);
    bool send_request(
        const std::string & method,
        const nlohmann::ordered_json & params,
        nlohmann::ordered_json & response,
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline = {});
    void collect_stderr_tail();
    void capture_exit_if_needed();
    std::string with_transport_context(const std::string & base_error) const;
    std::optional<std::chrono::steady_clock::time_point> effective_deadline(
        std::optional<std::chrono::steady_clock::time_point> deadline) const;
    bool terminate_process_until(std::chrono::steady_clock::time_point deadline);
    void shutdown_process();

    agent_mcp_stdio_client_config config;
    struct impl;
    std::unique_ptr<impl> state;
};

struct agent_mcp_http_client_config {
    std::string server_name;
    std::string url;
    std::string bearer_token;
    std::vector<std::string> allowed_tools;
    uint32_t connect_timeout_ms = 5000;
    uint32_t request_timeout_ms = 30000;
    uint32_t shutdown_timeout_ms = 2000;
    size_t max_result_bytes = 1024 * 1024;
};

class agent_mcp_http_client : public agent_mcp_tool_client {
public:
    explicit agent_mcp_http_client(agent_mcp_http_client_config config);
    ~agent_mcp_http_client() override;

    bool list_tools(
        const agent_tool_context & context,
        std::vector<mcp_agent_tool_definition> & tools,
        std::string & error) override;

    bool call_tool(
        const agent_tool_context & context,
        const mcp_agent_tool_definition & tool,
        const std::string & arguments_json,
        mcp_agent_tool_call_result & result,
        std::string & error) override;

    bool list_resources(
        std::vector<mcp_agent_resource_definition> & resources,
        std::string & error);

    bool read_resource(
        const std::string & uri,
        mcp_agent_resource_read_result & result,
        std::string & error);

private:
    bool initialize(
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline = {});
    bool send_notification(
        const std::string & method,
        const nlohmann::ordered_json & params,
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline = {});
    bool send_request(
        const std::string & method,
        const nlohmann::ordered_json & params,
        nlohmann::ordered_json & response,
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline = {});

    agent_mcp_http_client_config config;
    struct impl;
    std::unique_ptr<impl> state;
};

bool agent_dispatch_chat_tool_calls(
    common_chat_msg & assistant_message,
    agent_tool_view & tool_view,
    size_t max_calls,
    common_tool_chat_dispatch_result & result,
    std::string & error);
