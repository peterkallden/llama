#include "tools/agent/tooling/agent-tool-provider.h"

#include <cpp-httplib/httplib.h>

#include <cstdio>
#include <string>
#include <thread>

using json = nlohmann::ordered_json;

int main() {
    httplib::Server server;
    bool auth_seen = false;
    server.Post("/mcp", [&auth_seen](const httplib::Request & request, httplib::Response & response) {
        if (!request.has_header("Authorization") ||
                request.get_header_value("Authorization") != "Bearer smoke-token") {
            response.status = 401;
            response.set_content(R"({"error":"unauthorized"})", "application/json");
            return;
        }
        auth_seen = true;
        const auto message = json::parse(request.body, nullptr, false);
        if (message.is_discarded() || !message.is_object()) {
            response.status = 400;
            return;
        }
        const std::string method = message.value("method", "");
        if (method.rfind("notifications/", 0) == 0) {
            response.status = 204;
            return;
        }

        json result;
        if (method == "initialize") {
            result = {
                {"protocolVersion", "2025-06-18"},
                {"capabilities", json::object()},
                {"serverInfo", {{"name", "http-smoke"}, {"version", "1"}}},
            };
            response.set_header("Mcp-Session-Id", "http-smoke-session");
        } else if (method == "tools/list") {
            result = {{"tools", json::array({json{
                {"name", "search"},
                {"description", "HTTP smoke search"},
                {"inputSchema", {{"type", "object"}}},
            }})}};
        } else if (method == "tools/call") {
            result = {
                {"content", json::array({json{{"type", "text"}, {"text", "http smoke result"}}})},
                {"isError", false},
            };
        } else if (method == "resources/list") {
            result = {{"resources", json::array({json{
                {"uri", "mcp-resource://http-smoke/result"},
                {"name", "result.json"},
                {"mimeType", "application/json"},
            }})}};
        } else if (method == "resources/read") {
            result = {{"contents", json::array({json{
                {"uri", "mcp-resource://http-smoke/result"},
                {"mimeType", "application/json"},
                {"text", R"({"ok":true})"},
            }})}};
        } else {
            response.status = 404;
            return;
        }

        response.set_content(json{
            {"jsonrpc", "2.0"},
            {"id", message.value("id", 0)},
            {"result", std::move(result)},
        }.dump(), "application/json");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        std::fprintf(stderr, "HTTP smoke server failed to bind\n");
        return 1;
    }
    std::thread server_thread([&server]() { server.listen_after_bind(); });

    agent_mcp_http_client client({
        "http-smoke",
        "http://127.0.0.1:" + std::to_string(port) + "/mcp",
        "smoke-token",
        {},
        2000,
        5000,
        1000,
        4096,
    });
    agent_tool_context context;
    context.allow_network = true;
    std::vector<mcp_agent_tool_definition> tools;
    std::string error;
    const bool listed = client.list_tools(context, tools, error);
    mcp_agent_tool_call_result call_result;
    const bool called = listed && client.call_tool(
        context, tools.front(), R"({"query":"smoke"})", call_result, error);
    std::vector<mcp_agent_resource_definition> resources;
    const bool resources_listed = called && client.list_resources(resources, error);
    mcp_agent_resource_read_result resource;
    const bool resource_read = resources_listed && client.read_resource(resources.front().resource.uri, resource, error);

    server.stop();
    if (server_thread.joinable()) server_thread.join();

    if (!listed || tools.size() != 1 || !called || !call_result.ok ||
            call_result.text_content != "http smoke result" || !resources_listed ||
            resources.size() != 1 || !resource_read || resource.text_content != R"({"ok":true})" ||
            !auth_seen) {
        std::fprintf(stderr, "HTTP MCP client smoke failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("http_mcp_tools=%zu\n", tools.size());
    std::printf("http_mcp_session=%s\n", "http-smoke-session");
    std::printf("http_mcp_resource=%s\n", resource.text_content.c_str());
    return 0;
}
