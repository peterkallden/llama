#include "agent-mcp-stdio-server.h"

#include "agent-mcp-server-protocol.h"

#include <utility>
#include <thread>
#include <chrono>

agent_mcp_stdio_server::agent_mcp_stdio_server(
        agent_mcp_server_tool_registry registry,
        agent_mcp_stdio_server_options options)
    : registry_(std::move(registry))
    , options_(std::move(options)) {}

int agent_mcp_stdio_server::run(FILE * input, FILE * output, FILE * diagnostics) {
    bool shutdown_requested = false;
    for (;;) {
        agent_mcp_json message;
        std::string error;
        if (!agent_mcp_read_json_rpc_message(input, message, error)) {
            return 0;
        }

        const std::string method = message.value("method", "");
        if (method == "exit") {
            return shutdown_requested ? 0 : 1;
        }
        if (!message.contains("id")) {
            continue;
        }

        const auto & id = message["id"];
        agent_mcp_json response;

        if (method == "initialize") {
            response = agent_mcp_make_json_rpc_result(id, {
                {"protocolVersion", options_.protocol_version},
                {"capabilities", {
                    {"tools", agent_mcp_json::object()},
                    {"resources", options_.list_resources ? agent_mcp_json::object() : agent_mcp_json()},
                }},
                {"serverInfo", {
                    {"name", options_.server_name},
                    {"version", options_.server_version},
                }},
            });
            if (!agent_mcp_write_json_rpc_message(output, response, error)) {
                return 1;
            }
            if (options_.exit_after_initialize) {
                if (diagnostics != nullptr) {
                    std::fprintf(diagnostics, "%s: exiting after initialize\n", options_.server_name.c_str());
                    std::fflush(diagnostics);
                }
                return 9;
            }
            continue;
        }

        if (method == "shutdown") {
            shutdown_requested = true;
            response = agent_mcp_make_json_rpc_result(id, agent_mcp_json::object());
        } else if (method == "tools/list") {
            if (options_.hang_on_tools_list) {
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            if (options_.emit_malformed_tools_list) {
                if (diagnostics != nullptr) {
                    std::fprintf(diagnostics, "%s: emitting malformed tools/list payload\n", options_.server_name.c_str());
                    std::fflush(diagnostics);
                }
                if (!agent_mcp_write_malformed_json_rpc_result(output, id, error)) {
                    return 1;
                }
                return 11;
            }
            response = agent_mcp_make_json_rpc_result(id, agent_mcp_render_tools_list_result(registry_));
        } else if (method == "resources/list") {
            if (!options_.list_resources) {
                response = agent_mcp_make_json_rpc_error(id, -32601, "method not found");
            } else {
                agent_mcp_json result;
                if (!options_.list_resources(result, error)) {
                    response = agent_mcp_make_json_rpc_error(id, -32000, error.empty() ? "resources/list failed" : error);
                } else {
                    response = agent_mcp_make_json_rpc_result(id, std::move(result));
                }
            }
        } else if (method == "resources/read") {
            if (!options_.read_resource) {
                response = agent_mcp_make_json_rpc_error(id, -32601, "method not found");
            } else {
                const auto params = message.value("params", agent_mcp_json::object());
                agent_mcp_json result;
                if (!options_.read_resource(params, result, error)) {
                    response = agent_mcp_make_json_rpc_error(id, -32000, error.empty() ? "resources/read failed" : error);
                } else {
                    response = agent_mcp_make_json_rpc_result(id, std::move(result));
                }
            }
        } else if (method == "tools/call") {
            const auto params = message.value("params", agent_mcp_json::object());
            const std::string name = params.value("name", "");
            const auto arguments = params.value("arguments", agent_mcp_json::object());
            agent_mcp_server_tool_result tool_result;
            if (!registry_.call_tool(name, arguments, tool_result, error)) {
                tool_result.ok = false;
                const bool not_found = error.rfind("unknown MCP server tool:", 0) == 0;
                tool_result.failure_code = not_found ? "tool.not_found" : "tool.invalid_arguments";
                tool_result.failure_class = not_found ? "not_found" : "validation";
                tool_result.retryable = false;
                tool_result.safe_summary = error.empty() ? "MCP tool call is invalid." : error;
                tool_result.raw_diagnostic = error;
                tool_result.content = agent_mcp_json::array({
                    {
                        {"type", "text"},
                        {"text", tool_result.safe_summary},
                    },
                });
            }
            response = agent_mcp_make_json_rpc_result(id, agent_mcp_render_tool_call_result(tool_result));
        } else {
            response = agent_mcp_make_json_rpc_error(id, -32601, "method not found");
        }

        if (!agent_mcp_write_json_rpc_message(output, response, error)) {
            return 1;
        }
    }
}
