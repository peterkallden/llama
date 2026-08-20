#include "tools/agent/mcp/agent-mcp-http-server.h"
#include "tools/agent/daemon/agent-daemon-dispatcher.h"
#include "tools/agent/daemon/agent-daemon-event-collector.h"

#include <cpp-httplib/httplib.h>

#include <cstdio>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <atomic>
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
    if (!registry.register_tool({
            "write_probe",
            "HTTP inbound smoke write probe",
            R"({"type":"object"})",
            false,
            true,
            false,
            false,
            false,
            [](const agent_mcp_json &, agent_mcp_server_tool_result &, std::string &) {
                return true;
            },
        }, error)) {
        std::fprintf(stderr, "failed to register HTTP smoke write probe: %s\n", error.c_str());
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
            policy.allow_writes,
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

    common_agent_daemon_event_collector event_collector;
    std::atomic<bool> close_stream_after_event = false;
    agent_mcp_http_server_options server_options;
    server_options.listen_address = "127.0.0.1";
    server_options.port = 0;
    server_options.path = "/mcp";
    server_options.authenticator = authenticator;
    server_options.max_body_bytes = 4096;
    server_options.max_result_bytes = 4096;
    server_options.server_name = "http-inbound-smoke";
    server_options.server_version = "1";
    server_options.protocol_version = "2025-11-25";
    server_options.tasks_enabled = true;
    server_options.execute_tool = execute_through_dispatcher;
    server_options.subscribe_events = [&event_collector](common_agent_event_stream_subscription subscription) {
        return event_collector.subscribe(std::move(subscription));
    };
    server_options.unsubscribe_events = [&event_collector](const std::string & subscription_id) {
        event_collector.unsubscribe(subscription_id);
    };
    server_options.wait_for_event = [&event_collector, &close_stream_after_event](
            const std::string & subscription_id,
            common_agent_event_stream_delivery & delivery,
            std::chrono::milliseconds timeout) {
        if (close_stream_after_event) {
            return common_agent_event_stream_wait_status::closed;
        }
        const auto status = event_collector.wait_next(subscription_id, delivery, timeout);
        if (status == common_agent_event_stream_wait_status::delivered &&
                delivery.kind == common_agent_event_stream_delivery_kind::event) {
            close_stream_after_event = true;
        }
        return status;
    };
    agent_mcp_http_server server(std::move(registry), std::move(server_options));
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
    headers.emplace("MCP-Protocol-Version", initialize->get_header_value("MCP-Protocol-Version"));
    auto missing_protocol_headers = headers;
    missing_protocol_headers.erase("MCP-Protocol-Version");
    const auto missing_protocol = client.Post(
        "/mcp", missing_protocol_headers,
        R"({"jsonrpc":"2.0","id":98,"method":"tools/list","params":{}})",
        "application/json");
    const auto invalid_cursor = client.Get(
        "/mcp",
        httplib::Headers{
            {"Authorization", "Bearer http-smoke-token-a"},
            {"Mcp-Session-Id", headers.find("Mcp-Session-Id")->second},
            {"MCP-Protocol-Version", headers.find("MCP-Protocol-Version")->second},
            {"Accept", "text/event-stream"},
            {"Last-Event-ID", "not-a-sequence"},
        });
    const auto unsupported_initialize = client.Post(
        "/mcp",
        httplib::Headers{{"Authorization", "Bearer http-smoke-token-a"}},
        R"({"jsonrpc":"2.0","id":99,"method":"initialize","params":{"protocolVersion":"2099-01-01"}})",
        "application/json");

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
            !replacement_registry.register_tool({
                "write_probe",
                "HTTP inbound smoke write probe after catalog reload",
                R"({"type":"object"})",
                false,
                true,
                false,
                false,
                false,
                [](const agent_mcp_json &, agent_mcp_server_tool_result &, std::string &) { return true; },
            }, error) ||
            !server.replace_registry(std::move(replacement_registry), error)) {
        std::fprintf(stderr, "HTTP server catalog replacement failed: %s\n", error.c_str());
        server.stop(); server_thread.join(); return 1;
    }
    const auto reloaded_listed = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":9,"method":"tools/list","params":{}})", "application/json");
    const auto called = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hello"}}})", "application/json");
    const auto task_called = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":30,"method":"tools/call","params":{"name":"echo","arguments":{"text":"task hello"},"task":{"ttl":10000}}})", "application/json");
    const auto task_called_json = task_called ? json::parse(task_called->body, nullptr, false) : json();
    const std::string task_id = task_called_json.value("result", json::object())
        .value("task", json::object()).value("taskId", "");
    const auto task_get = task_id.empty() ? httplib::Result{} : client.Post(
        "/mcp", headers,
        json{{"jsonrpc", "2.0"}, {"id", 31}, {"method", "tasks/get"}, {"params", {{"taskId", task_id}}}}.dump(),
        "application/json");
    const auto task_result = task_id.empty() ? httplib::Result{} : client.Post(
        "/mcp", headers,
        json{{"jsonrpc", "2.0"}, {"id", 32}, {"method", "tasks/result"}, {"params", {{"taskId", task_id}}}}.dump(),
        "application/json");
    const auto invalid = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"echo","arguments":{}}})", "application/json");
    const auto unknown = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"missing","arguments":{}}})", "application/json");
    const auto write_blocked = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"write_probe","arguments":{}}})", "application/json");

    httplib::Headers other_headers = {{"Authorization", "Bearer http-smoke-token-b"}};
    const auto other_initialize = client.Post("/mcp", other_headers,
        R"({"jsonrpc":"2.0","id":5,"method":"initialize","params":{}})", "application/json");
    if (other_initialize && other_initialize->has_header("Mcp-Session-Id")) {
        other_headers.emplace("Mcp-Session-Id", other_initialize->get_header_value("Mcp-Session-Id"));
        other_headers.emplace("MCP-Protocol-Version", other_initialize->get_header_value("MCP-Protocol-Version"));
    }
    const auto other_listed = client.Post("/mcp", other_headers,
        R"({"jsonrpc":"2.0","id":6,"method":"tools/list","params":{}})", "application/json");
    const auto other_called = client.Post("/mcp", other_headers,
        R"({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"echo","arguments":{"text":"blocked"}}})", "application/json");
    std::atomic<bool> stream_headers_seen = false;
    std::string stream_body;
    std::mutex stream_body_mutex;
    std::thread event_stream_thread([&]() {
        const auto stream_response = client.Get(
            "/mcp",
            httplib::Headers{
                {"Authorization", "Bearer http-smoke-token-a"},
                {"Mcp-Session-Id", headers.find("Mcp-Session-Id")->second},
                {"MCP-Protocol-Version", headers.find("MCP-Protocol-Version")->second},
                {"Accept", "text/event-stream"},
                {"Last-Event-ID", std::to_string(event_collector.latest_sequence())},
            },
            [&stream_headers_seen](const httplib::Response &) {
                stream_headers_seen = true;
                return true;
            },
            [&stream_body, &stream_body_mutex](const char * data, size_t length) {
                std::lock_guard<std::mutex> lock(stream_body_mutex);
                stream_body.append(data, length);
                return true;
            });
        if (!stream_response) stream_headers_seen = true;
    });
    for (int attempt = 0; attempt < 100 && !stream_headers_seen; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    event_collector.append(make_common_agent_daemon_event(
        common_agent_daemon_event_type::status_reported,
        "http-stream-smoke",
        "",
        "stream event",
        0,
        {"namespace-a", "project-a", "http-stream-session", "http-stream-smoke", "", ""}));
    for (int attempt = 0; attempt < 100; ++attempt) {
        bool received = false;
        {
            std::lock_guard<std::mutex> lock(stream_body_mutex);
            received = stream_body.find("notifications/message") != std::string::npos;
        }
        if (received) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    httplib::Headers cross_headers = {
        {"Authorization", "Bearer http-smoke-token-b"},
        {"Mcp-Session-Id", headers.find("Mcp-Session-Id")->second},
        {"MCP-Protocol-Version", headers.find("MCP-Protocol-Version")->second},
    };
    const auto cross_cleanup = client.Delete("/mcp", cross_headers);
    const auto cleanup = client.Delete("/mcp", headers);
    const auto other_cleanup = client.Delete("/mcp", other_headers);
    auto reloaded_authenticator = std::make_shared<agent_mcp_opaque_token_authenticator>();
    const bool reloaded_auth_configured = reloaded_authenticator->register_token(
        "http-smoke-token-reloaded",
        {"caller-reloaded", "llama-agent", "namespace-reloaded", "project-reloaded", "minimal", {"echo"}, false},
        error);
    if (reloaded_auth_configured) {
        server.replace_authenticator(std::move(reloaded_authenticator));
    }
    const auto revoked_old_token = client.Post(
        "/mcp",
        httplib::Headers{{"Authorization", "Bearer http-smoke-token-a"}},
        R"({"jsonrpc":"2.0","id":14,"method":"initialize","params":{}})",
        "application/json");
    httplib::Headers reloaded_headers = {{"Authorization", "Bearer http-smoke-token-reloaded"}};
    const auto reloaded_auth_initialize = client.Post(
        "/mcp",
        reloaded_headers,
        R"({"jsonrpc":"2.0","id":15,"method":"initialize","params":{}})",
        "application/json");
    bool reloaded_auth_cleanup_ok = false;
    if (reloaded_auth_initialize && reloaded_auth_initialize->has_header("Mcp-Session-Id")) {
        const auto cleanup_result = client.Delete("/mcp", httplib::Headers{
            {"Authorization", "Bearer http-smoke-token-reloaded"},
            {"Mcp-Session-Id", reloaded_auth_initialize->get_header_value("Mcp-Session-Id")},
            {"MCP-Protocol-Version", reloaded_auth_initialize->get_header_value("MCP-Protocol-Version")},
        });
        reloaded_auth_cleanup_ok = cleanup_result && cleanup_result->status == 204;
    }

    const auto listed_json = listed ? json::parse(listed->body, nullptr, false) : json();
    const auto initialize_json = json::parse(initialize->body, nullptr, false);
    const auto unsupported_initialize_json = unsupported_initialize
        ? json::parse(unsupported_initialize->body, nullptr, false) : json();
    const auto reloaded_listed_json = reloaded_listed ? json::parse(reloaded_listed->body, nullptr, false) : json();
    const auto called_json = called ? json::parse(called->body, nullptr, false) : json();
    const auto task_get_json = task_get ? json::parse(task_get->body, nullptr, false) : json();
    const auto task_result_json = task_result ? json::parse(task_result->body, nullptr, false) : json();
    const auto invalid_json = invalid ? json::parse(invalid->body, nullptr, false) : json();
    const auto unknown_json = unknown ? json::parse(unknown->body, nullptr, false) : json();
    const auto other_listed_json = other_listed ? json::parse(other_listed->body, nullptr, false) : json();
    const auto other_called_json = other_called ? json::parse(other_called->body, nullptr, false) : json();
    const auto write_blocked_json = write_blocked ? json::parse(write_blocked->body, nullptr, false) : json();
    const bool ok = listed && listed->status == 200 && listed_json["result"]["tools"].size() == 1 &&
        missing_protocol && missing_protocol->status == 400 &&
        invalid_cursor && invalid_cursor->status == 400 &&
        initialize_json["result"]["protocolVersion"] == "2025-11-25" &&
        initialize_json["result"]["capabilities"].contains("tasks") &&
        unsupported_initialize && unsupported_initialize->status == 200 &&
        unsupported_initialize_json["error"]["code"] == -32602 &&
        reloaded_listed && reloaded_listed->status == 200 &&
        reloaded_listed_json["result"]["tools"][0]["description"] == "HTTP inbound smoke echo after catalog reload" &&
        called && called->status == 200 && called_json["result"]["structuredContent"]["text"] == "hello" &&
        task_called && task_called->status == 200 && !task_id.empty() &&
        task_get && task_get->status == 200 && task_get_json["result"]["taskId"] == task_id &&
        task_result && task_result->status == 200 && task_result_json["result"]["structuredContent"]["text"] == "task hello" &&
        invalid && invalid->status == 200 && invalid_json["result"]["isError"] == true &&
        unknown && unknown->status == 200 && unknown_json["result"]["isError"] == true &&
        write_blocked && write_blocked->status == 200 && write_blocked_json["result"]["isError"] == true &&
        other_initialize && other_initialize->status == 200 &&
        other_listed && other_listed->status == 200 && other_listed_json["result"]["tools"].empty() &&
        other_called && other_called->status == 200 && other_called_json["result"]["isError"] == true &&
        cross_cleanup && cross_cleanup->status == 403 &&
        cleanup && cleanup->status == 204 && other_cleanup && other_cleanup->status == 204 &&
        reloaded_auth_configured && revoked_old_token && revoked_old_token->status == 401 &&
        reloaded_auth_initialize && reloaded_auth_initialize->status == 200 &&
        reloaded_auth_cleanup_ok;

    server.stop();
    event_stream_thread.join();
    const std::string captured_stream = [&]() {
        std::lock_guard<std::mutex> lock(stream_body_mutex);
        return stream_body;
    }();
    const bool stream_ok = captured_stream.find("event: message") != std::string::npos &&
        captured_stream.find("notifications/message") != std::string::npos &&
        captured_stream.find("stream event") != std::string::npos;

    server_thread.join();
    if (!ok) {
        std::fprintf(stderr, "HTTP inbound server smoke failed init=%d listed=%d task_called=%d task_get=%d task_result=%d cleanup=%d other_cleanup=%d\n",
            initialize_json["result"]["protocolVersion"] == "2025-11-25",
            listed ? listed->status : 0,
            task_called ? task_called->status : 0,
            task_get ? task_get->status : 0,
            task_result ? task_result->status : 0,
            cleanup ? cleanup->status : 0,
            other_cleanup ? other_cleanup->status : 0);
        if (task_called) std::fprintf(stderr, "task_called=%s\n", task_called->body.c_str());
        if (task_get) std::fprintf(stderr, "task_get=%s\n", task_get->body.c_str());
        if (task_result) std::fprintf(stderr, "task_result=%s\n", task_result->body.c_str());
        return 1;
    }
    if (!stream_ok) {
        std::fprintf(stderr, "HTTP inbound SSE stream smoke failed: %s\n", captured_stream.c_str());
        return 1;
    }
    const auto session_header = headers.find("Mcp-Session-Id");
    std::printf("http_inbound_session=%s\n", session_header == headers.end() ? "" : session_header->second.c_str());
    std::printf("http_inbound_tools=1\n");
    std::printf("http_inbound_callers=2\n");
    std::printf("http_inbound_cleanup=ok\n");
    std::printf("http_inbound_sse=ok\n");
    return 0;
}
