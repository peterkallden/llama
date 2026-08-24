#include "agent-mcp-client-factory.h"

#include <utility>

std::unique_ptr<agent_mcp_tool_client> make_agent_mcp_client(
        const agent_mcp_client_factory_request & request,
        std::string & error) {
    if (request.transport == "stdio") {
        if (request.command_line.empty()) {
            error = "MCP provider command line must not be empty";
            return nullptr;
        }
        error.clear();
        return std::make_unique<agent_mcp_stdio_client>(agent_mcp_stdio_client_config{
            request.server_name,
            request.command_line,
            {},
            request.request_timeout_ms,
            request.shutdown_timeout_ms,
        });
    }

    if (request.url.empty()) {
        error = "HTTP MCP provider url must not be empty";
        return nullptr;
    }
    error.clear();
    return std::make_unique<agent_mcp_http_client>(agent_mcp_http_client_config{
        request.server_name,
        request.url,
        request.bearer_token,
        request.allowed_tools,
        request.connect_timeout_ms,
        request.request_timeout_ms,
        request.shutdown_timeout_ms,
        request.max_result_bytes,
        {},
        request.credential_ref,
        request.credential_provider,
    });
}
