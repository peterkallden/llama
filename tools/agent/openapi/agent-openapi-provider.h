#pragma once

#include "agent-openapi-catalog.h"
#include "../tooling/agent-tool-provider.h"

#include <functional>
#include <memory>

struct agent_openapi_execution_result {
    bool ok = false;
    int http_status = 0;
    std::string mime_type = "application/json";
    std::string structured_content_json;
    std::string text_content;
    std::vector<common_runtime_resource_ref> resource_refs;
    std::string failure_code;
    common_tool_failure_class failure_class = common_tool_failure_class::execution;
    bool retryable = false;
    std::string safe_summary;
    std::string raw_diagnostic;
};

using agent_openapi_executor = std::function<bool(
    const agent_tool_context & context,
    const agent_openapi_operation & operation,
    const std::string & arguments_json,
    agent_openapi_execution_result & result,
    std::string & error)>;

// Optional host-owned post-processor. It may attach turn/session-scoped
// resource or dataset references using the existing stores. It must not issue
// a second model step or follow an unvalidated URL.
using agent_openapi_result_materializer = std::function<bool(
    const agent_tool_context & context,
    const agent_openapi_operation & operation,
    agent_openapi_execution_result & result,
    std::string & error)>;

// Exposes a filtered OpenAPI catalog through the same agent_tool_view contract
// as MCP. The executor is host-owned; this class never lets the model choose a
// URL, credentials, or operation outside the already filtered catalog.
class agent_openapi_tool_provider : public agent_tool_provider {
public:
    agent_openapi_tool_provider(
        agent_openapi_catalog catalog,
        agent_openapi_executor executor,
        agent_openapi_result_materializer materializer = {});
    ~agent_openapi_tool_provider() override;
    std::unique_ptr<agent_tool_view> resolve_tools(
        const agent_tool_context & context, std::string & error) override;

private:
    class client;
    agent_openapi_catalog catalog;
    agent_openapi_executor executor;
    agent_openapi_result_materializer materializer;
    std::unique_ptr<client> client_impl;
    std::unique_ptr<mcp_agent_tool_provider> delegate;
};
