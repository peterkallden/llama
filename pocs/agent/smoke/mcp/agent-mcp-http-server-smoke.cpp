#include "tools/agent/mcp/agent-mcp-http-server.h"
#include "tools/agent/daemon/agent-daemon-dispatcher.h"

#include <cpp-httplib/httplib.h>

#include <cstdio>
#include <memory>
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

    auto authenticator = std::make_shared<agent_mcp_opaque_token_authenticator>();
    if (!authenticator->register_token("http-smoke-token-a", {
            "caller-a", "llama-agent", "namespace-a", "project-a", "minimal", {"echo"}, false,
        }, error) ||
            !authenticator->register_token("http-smoke-token-b", {
                "caller-b", "llama-agent", "namespace-b", "project-b", "restricted", {"not-echo"}, false,
            }, error)) {
        std::fprintf(stderr, "failed to configure HTTP smoke auth: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_runtime daemon_runtime;
    daemon_runtime.tool_executor = [](
            const common_agent_daemon_tool_payload & payload,
            agent_tool_result & result,
            std::string & callback_error) {
        const auto arguments = json::parse(payload.arguments_json, nullptr, false);
        if (arguments.is_discarded() || !arguments.is_object() ||
                payload.tool_name != "echo" || arguments.value("text", "").empty()) {
            callback_error = "dispatcher tool executor rejected arguments";
            result.raw_diagnostic = callback_error;
            return false;
        }
        result.ok = true;
        result.content_json = json{{"text", arguments.at("text")}}.dump();
        result.content_summary = arguments.at("text").get<std::string>();
        callback_error.clear();
        return true;
    };
    common_agent_daemon_dispatcher dispatcher(std::move(daemon_runtime), 8, 1);
    const auto execute_through_dispatcher = [&dispatcher](
            const agent_mcp_caller_policy & policy,
            const std::string & tool_name,
            const agent_mcp_json & arguments,
            agent_mcp_server_tool_result & result,
            std::string & callback_error) {
        common_agent_daemon_command command;
        command.request_id = "http-dispatcher-tool";
        command.type = common_agent_daemon_command_type::execute_tool;
        command.tool = common_agent_daemon_tool_payload{
            {policy.namespace_id, policy.caller_id + "-session"},
            policy.project_id,
            policy.tool_profile,
            tool_name,
            arguments.dump(),
        };
        common_agent_daemon_command_result command_result;
        if (!dispatcher.execute(command, command_result, callback_error)) {
            result.ok = false;
            result.safe_summary = callback_error;
            result.content = {{{"type", "text"}, {"text", callback_error}}};
            return false;
        }
        result.ok = command_result.tool_result.ok;
        result.structured_content = json::parse(command_result.tool_result.content_json, nullptr, false);
        if (result.structured_content.is_discarded()) result.structured_content = json::object();
        result.safe_summary = command_result.tool_result.content_summary;
        result.content = {{{"type", "text"}, {"text", result.safe_summary}}};
        callback_error = command_result.error;
        return result.ok;
    };

    agent_mcp_http_server server(std::move(registry), {
        "127.0.0.1", 0, "/mcp", "", "", authenticator, {}, 4096, 4096,
        "http-inbound-smoke", "1", "2025-06-18", {}, {}, execute_through_dispatcher,
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

    httplib::Headers headers = {{"Authorization", "Bearer http-smoke-token-a"}};
    const auto initialize = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})", "application/json");
    if (!initialize || initialize->status != 200 || !initialize->has_header("Mcp-Session-Id")) {
        std::fprintf(stderr, "HTTP server initialize failed\n");
        server.stop(); server_thread.join(); return 1;
    }
    headers.emplace("Mcp-Session-Id", initialize->get_header_value("Mcp-Session-Id"));

    const auto listed = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})", "application/json");
    agent_mcp_server_tool_registry replacement_registry;
    if (!replacement_registry.register_tool({
            "echo",
            "HTTP inbound smoke echo after catalog reload",
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
        }, error) ||
            !server.replace_registry(std::move(replacement_registry), error)) {
        std::fprintf(stderr, "HTTP server catalog replacement failed: %s\n", error.c_str());
        server.stop(); server_thread.join(); return 1;
    }
    const auto reloaded_listed = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":9,"method":"tools/list","params":{}})", "application/json");
    const auto called = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hello"}}})", "application/json");
    const auto invalid = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"echo","arguments":{}}})", "application/json");
    const auto unknown = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"missing","arguments":{}}})", "application/json");

    httplib::Headers other_headers = {{"Authorization", "Bearer http-smoke-token-b"}};
    const auto other_initialize = client.Post("/mcp", other_headers,
        R"({"jsonrpc":"2.0","id":5,"method":"initialize","params":{}})", "application/json");
    if (other_initialize && other_initialize->has_header("Mcp-Session-Id")) {
        other_headers.emplace("Mcp-Session-Id", other_initialize->get_header_value("Mcp-Session-Id"));
    }
    const auto other_listed = client.Post("/mcp", other_headers,
        R"({"jsonrpc":"2.0","id":6,"method":"tools/list","params":{}})", "application/json");
    const auto other_called = client.Post("/mcp", other_headers,
        R"({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"echo","arguments":{"text":"blocked"}}})", "application/json");
    httplib::Headers cross_headers = {
        {"Authorization", "Bearer http-smoke-token-b"},
        {"Mcp-Session-Id", headers.find("Mcp-Session-Id")->second},
    };
    const auto cross_cleanup = client.Delete("/mcp", cross_headers);
    const auto cleanup = client.Delete("/mcp", headers);
    const auto other_cleanup = client.Delete("/mcp", other_headers);

    const auto listed_json = listed ? json::parse(listed->body, nullptr, false) : json();
    const auto reloaded_listed_json = reloaded_listed ? json::parse(reloaded_listed->body, nullptr, false) : json();
    const auto called_json = called ? json::parse(called->body, nullptr, false) : json();
    const auto invalid_json = invalid ? json::parse(invalid->body, nullptr, false) : json();
    const auto unknown_json = unknown ? json::parse(unknown->body, nullptr, false) : json();
    const auto other_listed_json = other_listed ? json::parse(other_listed->body, nullptr, false) : json();
    const auto other_called_json = other_called ? json::parse(other_called->body, nullptr, false) : json();
    const bool ok = listed && listed->status == 200 && listed_json["result"]["tools"].size() == 1 &&
        reloaded_listed && reloaded_listed->status == 200 &&
        reloaded_listed_json["result"]["tools"][0]["description"] == "HTTP inbound smoke echo after catalog reload" &&
        called && called->status == 200 && called_json["result"]["structuredContent"]["text"] == "hello" &&
        invalid && invalid->status == 200 && invalid_json["result"]["isError"] == true &&
        unknown && unknown->status == 200 && unknown_json["result"]["isError"] == true &&
        other_initialize && other_initialize->status == 200 &&
        other_listed && other_listed->status == 200 && other_listed_json["result"]["tools"].empty() &&
        other_called && other_called->status == 200 && other_called_json["result"]["isError"] == true &&
        cross_cleanup && cross_cleanup->status == 403 &&
        cleanup && cleanup->status == 204 && other_cleanup && other_cleanup->status == 204;

    server.stop();
    server_thread.join();
    if (!ok) {
        std::fprintf(stderr, "HTTP inbound server smoke failed\n");
        return 1;
    }
    const auto session_header = headers.find("Mcp-Session-Id");
    std::printf("http_inbound_session=%s\n", session_header == headers.end() ? "" : session_header->second.c_str());
    std::printf("http_inbound_tools=1\n");
    std::printf("http_inbound_callers=2\n");
    std::printf("http_inbound_cleanup=ok\n");
    return 0;
}
