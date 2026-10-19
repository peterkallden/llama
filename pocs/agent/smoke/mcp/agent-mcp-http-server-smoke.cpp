#include "tools/agent/mcp/agent-mcp-http-server.h"
#include "tools/agent/daemon/agent-daemon-dispatcher.h"
#include "tools/agent/daemon/agent-daemon-event-collector.h"
#include "tools/agent/tooling/agent-tool-provider.h"
#include "agent/tooling/catalog/model-projection.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/tooling/contracts/tool-result-schema.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

static bool validate_catalog_result_sample(
        const common_tool_catalog & catalog,
        const std::string & tool_name,
        const std::string & sample_json) {
    const auto * definition = catalog.find_definition(tool_name);
    if (definition == nullptr || definition->result_schema_json.empty()) return false;
    std::string normalized;
    std::string error;
    const bool valid = common_schema_normalize_and_validate_object(
        sample_json, definition->result_schema_json, normalized, error);
    if (!valid) std::fprintf(stderr, "%s result fixture failed schema: %s\n", tool_name.c_str(), error.c_str());
    return valid;
}

int main() {
    agent_mcp_server_tool_registry registry;
    std::string error;
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("research", bootstrap, error)) {
        std::fprintf(stderr, "failed to bootstrap catalog for MCP exposure smoke: %s\n", error.c_str());
        return 1;
    }
    const bool catalog_result_samples_valid =
        validate_catalog_result_sample(catalog, "dataset.list",
            R"({"datasets":["dataset://sales"],"names":["sales"],"truncated":false})") &&
        validate_catalog_result_sample(catalog, "dataset.select",
            R"({"dataset":"dataset://sales","name":"sales"})") &&
        validate_catalog_result_sample(catalog, "dataset.inspect",
            R"({"dataset":"dataset://sales","name":"sales","rows":2,"columns":1})") &&
        validate_catalog_result_sample(catalog, "dataset.validate",
            R"({"valid":false,"violations":[{"type":"not_null","column":"region","row":2}],"rows_checked":2})") &&
        validate_catalog_result_sample(catalog, "repository.search",
            R"({"matches":[{"path":"src/main.cpp","line":4,"text":"query"}]})") &&
        validate_catalog_result_sample(catalog, "resource.inspect",
            R"({"resource":"agent-resource://turn/source","mime_type":"text/csv","size_bytes":42})") &&
        validate_catalog_result_sample(catalog, "resource.read",
            R"({"uri":"agent-resource://turn/source","text":"region,amount"})");
    if (!catalog_result_samples_valid) {
        std::fprintf(stderr, "representative native results failed their catalog output schemas\n");
        return 1;
    }

    const std::vector<std::string> catalog_tools = {
        "math.calculate",
    };
    const auto profile_tools = catalog.load_profile("all-configured", error);
    for (const auto & name : catalog_tools) {
        if (std::find_if(profile_tools.begin(), profile_tools.end(), [&name](const auto & definition) {
                return definition.name == name;
            }) == profile_tools.end()) {
            std::fprintf(stderr, "catalog tool is missing from all-configured MCP profile: %s\n", name.c_str());
            return 1;
        }
    }
    native_agent_tool_provider catalog_provider(
        catalog,
        [](const agent_tool_context &, common_native_tool_bindings & bindings, std::string &) {
            bindings.repository_root = std::filesystem::current_path().string();
            return true;
        });
    agent_tool_context catalog_context;
    catalog_context.profile_id = "minimal";
    catalog_context.allow_network = true;
    catalog_context.allow_policy_gated_writes = true;
    std::unique_ptr<agent_tool_view> catalog_view = catalog_provider.resolve_tools(catalog_context, error);
    if (!catalog_view) {
        std::fprintf(stderr, "failed to resolve production native tool view: %s\n", error.c_str());
        return 1;
    }
    std::vector<common_chat_tool> catalog_chat_tools;
    for (const auto & tool : catalog_view->chat_tools()) {
        if (std::find(catalog_tools.begin(), catalog_tools.end(), tool.name) != catalog_tools.end()) {
            catalog_chat_tools.push_back(tool);
        }
    }
    if (catalog_chat_tools.size() != catalog_tools.size()) {
        std::fprintf(stderr, "native provider did not resolve the expected catalog tools (resolved=%zu): %s\n",
            catalog_chat_tools.size(), error.c_str());
        return 1;
    }
    if (!agent_mcp_register_native_tool_view(
            catalog_chat_tools,
            [](const std::string &) { return true; },
            [](const std::string &) { return false; },
            registry,
            error)) {
        std::fprintf(stderr, "failed to inject catalog tool view into MCP registry: %s\n", error.c_str());
        return 1;
    }
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
            R"({"type":"object","additionalProperties":false,"required":["text"],"properties":{"text":{"type":"string"}}})",
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
            "caller-a", "llama-agent", "namespace-a", "project-a", "minimal",
            {"echo", "math.calculate", "diagnostics.native_crash", "dataset.select", "data.aggregate", "document.table", "artifact.export"}, false,
        }, error) ||
            !authenticator->register_token("http-smoke-token-b", {
                "caller-b", "llama-agent", "namespace-b", "project-b", "restricted", {"not-echo"}, false,
            }, error)) {
        std::fprintf(stderr, "failed to configure HTTP smoke auth: %s\n", error.c_str());
        return 1;
    }

    common_agent_daemon_runtime daemon_runtime;
    daemon_runtime.tool_executor = [native_view = catalog_view.get()](
            const common_agent_daemon_tool_payload & payload,
            agent_tool_result & result,
            std::string & callback_error) {
        const auto arguments = json::parse(payload.arguments_json, nullptr, false);
        if (payload.tool_name == "math.calculate") {
            result = native_view->call({"http-native-calculator", payload.tool_name, payload.arguments_json}, callback_error);
            return result.ok;
        }
        if (payload.tool_name == "echo" && arguments.is_object() &&
                arguments.value("text", "") == "throw") {
            throw std::runtime_error("dispatcher smoke exception");
        }
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
        if (tool_name == "echo" && arguments.is_object() &&
                arguments.value("text", "") == "return-false") {
            callback_error = "private callback diagnostic";
            return false;
        }
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
            const auto & tool_result = command_result.tool_result;
            agent_mcp_set_tool_call_result(
                result,
                false,
                json::parse(tool_result.content_json, nullptr, false),
                tool_result.safe_summary.empty()
                    ? "The tool execution failed." : tool_result.safe_summary,
                tool_result.failure_code.empty()
                    ? "agent.tool_execution_failed" : tool_result.failure_code,
                "execution",
                tool_result.retryable,
                tool_result.raw_diagnostic.empty() ? callback_error : tool_result.raw_diagnostic);
            return false;
        }
        const auto & tool_result = command_result.tool_result;
        agent_mcp_set_tool_call_result(
            result,
            tool_result.ok,
            json::parse(tool_result.content_json, nullptr, false),
            tool_result.content_summary,
            tool_result.failure_code,
            "execution",
            tool_result.retryable,
            tool_result.raw_diagnostic);
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
    server_options.max_result_bytes = 16 * 1024;
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
    const auto native_called = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"math.calculate","arguments":{"expression":"6*7"}}})", "application/json");
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
            R"({"type":"object","additionalProperties":false,"required":["text"],"properties":{"text":{"type":"string"}}})",
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
    const auto throw_called = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":33,"method":"tools/call","params":{"name":"echo","arguments":{"text":"throw"}}})", "application/json");
    const auto after_throw_called = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":34,"method":"tools/call","params":{"name":"echo","arguments":{"text":"after throw"}}})", "application/json");
    const auto false_callback_called = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":35,"method":"tools/call","params":{"name":"echo","arguments":{"text":"return-false"}}})", "application/json");
    const auto after_false_callback_called = client.Post("/mcp", headers,
        R"({"jsonrpc":"2.0","id":36,"method":"tools/call","params":{"name":"echo","arguments":{"text":"after false"}}})", "application/json");
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
    const auto native_called_json = native_called ? json::parse(native_called->body, nullptr, false) : json();
    const auto throw_called_json = throw_called ? json::parse(throw_called->body, nullptr, false) : json();
    const auto after_throw_called_json = after_throw_called ? json::parse(after_throw_called->body, nullptr, false) : json();
    const auto false_callback_json = false_callback_called ? json::parse(false_callback_called->body, nullptr, false) : json();
    const auto after_false_callback_json = after_false_callback_called ? json::parse(after_false_callback_called->body, nullptr, false) : json();
    const auto task_get_json = task_get ? json::parse(task_get->body, nullptr, false) : json();
    const auto task_result_json = task_result ? json::parse(task_result->body, nullptr, false) : json();
    const auto invalid_json = invalid ? json::parse(invalid->body, nullptr, false) : json();
    const auto unknown_json = unknown ? json::parse(unknown->body, nullptr, false) : json();
    const auto other_listed_json = other_listed ? json::parse(other_listed->body, nullptr, false) : json();
    const auto other_called_json = other_called ? json::parse(other_called->body, nullptr, false) : json();
    const auto write_blocked_json = write_blocked ? json::parse(write_blocked->body, nullptr, false) : json();
    const auto has_listed_tool = [&listed_json](const std::string & name) {
        if (!listed_json.is_object() || !listed_json.contains("result") ||
                !listed_json["result"].contains("tools") || !listed_json["result"]["tools"].is_array()) return false;
        for (const auto & tool : listed_json["result"]["tools"]) {
            if (tool.value("name", "") == name) return true;
        }
        return false;
    };
    const bool catalog_tools_exposed = std::all_of(
        catalog_tools.begin(), catalog_tools.end(), has_listed_tool);
    bool catalog_output_schema_matches = false;
    bool native_payload_matches_schema = false;
    if (listed_json.is_object() && listed_json.contains("result") &&
            listed_json["result"].contains("tools")) {
        for (const auto & tool : listed_json["result"]["tools"]) {
            if (tool.value("name", "") != "math.calculate") continue;
            const auto * definition = catalog.find_definition("math.calculate");
            catalog_output_schema_matches = definition != nullptr &&
                tool.value("outputSchema", json::object()) ==
                common_tool_success_payload_schema(definition->result_schema_json);
            std::string normalized, validation_error;
            if (native_called_json.is_object() && native_called_json.contains("result") &&
                    native_called_json["result"].contains("structuredContent")) {
                native_payload_matches_schema = common_schema_normalize_and_validate_object(
                    native_called_json["result"]["structuredContent"].dump(),
                    tool.value("outputSchema", json::object()).dump(), normalized, validation_error);
            }
            break;
        }
    }
    const bool ok = listed && listed->status == 200 && listed_json["result"]["tools"].size() == 2 &&
        catalog_tools_exposed && catalog_output_schema_matches && native_payload_matches_schema &&
        native_called && native_called->status == 200 &&
        native_called_json["result"]["isError"] == false &&
        native_called_json["result"]["structuredContent"]["ok"] == true &&
        native_called_json["result"]["structuredContent"]["result"]["value"] == 42 &&
        missing_protocol && missing_protocol->status == 400 &&
        invalid_cursor && invalid_cursor->status == 400 &&
        initialize_json["result"]["protocolVersion"] == "2025-11-25" &&
        initialize_json["result"]["capabilities"].contains("tasks") &&
        unsupported_initialize && unsupported_initialize->status == 200 &&
        unsupported_initialize_json["error"]["code"] == -32602 &&
        reloaded_listed && reloaded_listed->status == 200 &&
        reloaded_listed_json["result"]["tools"][0]["description"] == "HTTP inbound smoke echo after catalog reload" &&
        reloaded_listed_json["result"]["tools"][0]["outputSchema"] ==
            json::parse(R"({"type":"object","additionalProperties":false,"required":["text"],"properties":{"text":{"type":"string"}}})") &&
        called && called->status == 200 && called_json["result"]["structuredContent"]["text"] == "hello" &&
        throw_called && throw_called->status == 200 && throw_called_json["result"]["isError"] == true &&
        throw_called_json["result"]["errorInfo"]["code"] == "tool.executor_exception" &&
        throw_called_json["result"]["errorInfo"]["class"] == "execution" &&
        throw_called_json["result"]["errorInfo"]["retryable"] == false &&
        throw_called->body.find("dispatcher smoke exception") == std::string::npos &&
        after_throw_called && after_throw_called->status == 200 &&
        after_throw_called_json["result"]["structuredContent"]["text"] == "after throw" &&
        false_callback_called && false_callback_called->status == 200 &&
        false_callback_json["result"]["isError"] == true &&
        false_callback_json["result"]["errorInfo"]["code"] == "tool.execution_failed" &&
        false_callback_called->body.find("private callback diagnostic") == std::string::npos &&
        after_false_callback_called && after_false_callback_called->status == 200 &&
        after_false_callback_json["result"]["structuredContent"]["text"] == "after false" &&
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
        std::fprintf(stderr, "HTTP inbound server smoke failed init=%d listed=%d count=%zu catalog=%d native_schema=%d reloaded_schema=%d called=%d throw=%d after_throw=%d task_called=%d task_get=%d task_result=%d cleanup=%d other_cleanup=%d\n",
            initialize_json["result"]["protocolVersion"] == "2025-11-25",
            listed ? listed->status : 0,
            listed_json["result"]["tools"].size(),
            catalog_tools_exposed,
            catalog_output_schema_matches,
            reloaded_listed_json["result"]["tools"][0].value("outputSchema", json::object()) ==
                json::parse(R"({"type":"object","additionalProperties":false,"required":["text"],"properties":{"text":{"type":"string"}}})"),
            called ? called->status : 0,
            throw_called ? throw_called->status : 0,
            after_throw_called ? after_throw_called->status : 0,
            task_called ? task_called->status : 0,
            task_get ? task_get->status : 0,
            task_result ? task_result->status : 0,
            cleanup ? cleanup->status : 0,
            other_cleanup ? other_cleanup->status : 0);
        std::fprintf(stderr, "listed_tools=%s\n", listed_json.dump().c_str());
        if (task_called) std::fprintf(stderr, "task_called=%s\n", task_called->body.c_str());
        if (task_get) std::fprintf(stderr, "task_get=%s\n", task_get->body.c_str());
        if (task_result) std::fprintf(stderr, "task_result=%s\n", task_result->body.c_str());
        if (throw_called) std::fprintf(stderr, "throw_called=%s\n", throw_called->body.c_str());
        if (native_called) std::fprintf(stderr, "native_called=%s\n", native_called->body.c_str());
        return 1;
    }
    if (!stream_ok) {
        std::fprintf(stderr, "HTTP inbound SSE stream smoke failed: %s\n", captured_stream.c_str());
        return 1;
    }
    const auto session_header = headers.find("Mcp-Session-Id");
    std::printf("http_inbound_session=%s\n", session_header == headers.end() ? "" : session_header->second.c_str());
    std::printf("http_inbound_tools=%zu\n", listed_json["result"]["tools"].size());
    std::printf("http_inbound_callers=2\n");
    std::printf("http_inbound_cleanup=ok\n");
    std::printf("http_inbound_sse=ok\n");
    return 0;
}
