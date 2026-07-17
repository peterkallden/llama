#include "tools/agent/mcp/agent-mcp-http-server.h"

#include <cpp-httplib/httplib.h>

#include <cstdio>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

using json = nlohmann::ordered_json;

int main() {
    agent_mcp_server_tool_registry registry;
    std::string error;
    if (!registry.register_tool({
            "echo",
            "HTTP inbound smoke echo",
            R"({"type":"object","additionalProperties":false,"required":["text"],"properties":{"text":{"type":"string","minLength":1}}})",
            true,
            false,
            false,
            false,
            false,
            [](const agent_mcp_json & arguments, agent_mcp_server_tool_result & result, std::string &) {
                result.structured_content = {{"text", arguments.at("text")}};
                result.content = {{{"type", "text"}, {"text", arguments.at("text")}}};
                result.safe_summary = arguments.at("text").get<std::string>();
                return true;
            },
        }, error)) {
        std::fprintf(stderr, "failed to register HTTP smoke tool: %s\n", error.c_str());
        return 1;
    }

    agent_mcp_http_server server(std::move(registry), {
        "127.0.0.1", 0, "/mcp", "", "http-smoke-token", 4096, 4096,
        "http-inbound-smoke", "1", "2025-06-18", {}, {},
    });
    if (!server.bind(error)) {
        std::fprintf(stderr, "failed to bind HTTP server smoke: %s\n", error.c_str());
        return 1;
    }
    std::thread server_thread([&server, &error]() { server.listen(error); });
    httplib::Client client("127.0.0.1", server.port());

    const auto unauthorized = client.Post("/mcp", "{}", "application/json");
    if (!unauthorized || unauthorized->status != 401) {
        std::fprintf(stderr, "HTTP server did not reject missing bearer token\n");
        server.stop(); server_thread.join(); return 1;
    }

    httplib::Headers headers = {{"Authorization", "Bearer http-smoke-token"}};
    const auto initialize = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})", "application/json");
    if (!initialize || initialize->status != 200 || !initialize->has_header("Mcp-Session-Id")) {
        std::fprintf(stderr, "HTTP server initialize failed\n");
        server.stop(); server_thread.join(); return 1;
    }
    headers.emplace("Mcp-Session-Id", initialize->get_header_value("Mcp-Session-Id"));

    const auto listed = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})", "application/json");
    const auto called = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hello"}}})", "application/json");
    const auto invalid = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"echo","arguments":{}}})", "application/json");
    const auto cleanup = client.Delete("/mcp", headers);

    const auto listed_json = listed ? json::parse(listed->body, nullptr, false) : json();
    const auto called_json = called ? json::parse(called->body, nullptr, false) : json();
    const auto invalid_json = invalid ? json::parse(invalid->body, nullptr, false) : json();
    const bool ok = listed && listed->status == 200 && listed_json["result"]["tools"].size() == 1 &&
        called && called->status == 200 && called_json["result"]["structuredContent"]["text"] == "hello" &&
        invalid && invalid->status == 200 && invalid_json["result"]["isError"] == true &&
        cleanup && cleanup->status == 204;

    server.stop();
    server_thread.join();
    if (!ok) {
        std::fprintf(stderr, "HTTP inbound server smoke failed\n");
        return 1;
    }
    const auto session_header = headers.find("Mcp-Session-Id");
    std::printf("http_inbound_session=%s\n", session_header == headers.end() ? "" : session_header->second.c_str());
    std::printf("http_inbound_tools=1\n");
    std::printf("http_inbound_cleanup=ok\n");
    return 0;
}
