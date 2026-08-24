#include "../tooling/agent-tool-provider.h"
#include "agent-mcp-protocol.h"

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <mutex>
#include <nlohmann/json.hpp>
#include <utility>

using json = nlohmann::ordered_json;

namespace {

uint32_t bounded_timeout_ms(
        uint32_t configured_timeout_ms,
        const std::optional<std::chrono::steady_clock::time_point> & deadline) {
    if (!deadline.has_value()) return configured_timeout_ms;
    const auto now = std::chrono::steady_clock::now();
    if (now >= *deadline) return 1;
    const auto remaining = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        *deadline - now).count());
    const uint64_t bounded = configured_timeout_ms == 0
        ? remaining
        : std::min<uint64_t>(configured_timeout_ms, remaining);
    return static_cast<uint32_t>(std::max<uint64_t>(1, bounded));
}

} // namespace

struct agent_mcp_http_client::impl {
    std::mutex request_mutex;
    int next_request_id = 1;
    bool initialized = false;
    std::string session_id;
    std::string protocol_version = common_mcp_default_protocol_version();
    std::shared_ptr<agent_mcp_http_transport> transport;
};

agent_mcp_http_client::agent_mcp_http_client(agent_mcp_http_client_config config)
    : config(std::move(config)), state(std::make_unique<impl>()) {
    state->transport = this->config.transport
        ? this->config.transport
        : make_agent_mcp_httplib_transport();
}

agent_mcp_http_client::~agent_mcp_http_client() = default;

bool agent_mcp_http_client::initialize(
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline) {
    if (state->initialized) {
        return true;
    }
    json response;
    if (!send_request("initialize", make_mcp_initialize_params(), response, error, deadline)) {
        return false;
    }
    if (!response.contains("result") || !response["result"].is_object()) {
        error = "MCP HTTP initialize response did not contain a result object";
        return false;
    }
    state->protocol_version = response["result"].value("protocolVersion", "");
    if (!common_mcp_is_supported_protocol_version(state->protocol_version)) {
        error = "MCP HTTP initialize response negotiated an unsupported protocol version";
        return false;
    }
    if (!send_notification("notifications/initialized", json::object(), error, deadline)) {
        return false;
    }
    state->initialized = true;
    return true;
}

bool agent_mcp_http_client::send_notification(
        const std::string & method,
        const json & params,
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline) {
    json ignored;
    return send_request(method, params, ignored, error, deadline);
}

bool agent_mcp_http_client::send_request(
        const std::string & method,
        const json & params,
        json & response,
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline) {
    std::lock_guard<std::mutex> lock(state->request_mutex);
    const int request_id = state->next_request_id++;
    const bool is_notification = method.rfind("notifications/", 0) == 0;
    const json request = is_notification
        ? make_mcp_jsonrpc_notification(method, params)
        : make_mcp_jsonrpc_request(request_id, method, params);
    const std::string body = request.dump();
    std::map<std::string, std::string> headers = {
        {"Accept", "application/json, text/event-stream"},
        {"MCP-Protocol-Version", state->protocol_version},
    };
    if (!config.bearer_token.empty()) {
        headers.emplace("Authorization", "Bearer " + config.bearer_token);
    }
    if (!state->session_id.empty()) {
        headers.emplace("Mcp-Session-Id", state->session_id);
    }

    agent_mcp_http_response result;
    if (!state->transport->post({
            config.url,
            std::move(headers),
            body,
            bounded_timeout_ms(config.connect_timeout_ms, deadline),
            bounded_timeout_ms(config.request_timeout_ms, deadline),
            config.max_result_bytes,
            deadline}, result, error)) {
        return false;
    }
    if (result.status < 200 || result.status >= 300) {
        error = "MCP HTTP server returned status " + std::to_string(result.status);
        return false;
    }
    if (result.body.size() > config.max_result_bytes) {
        error = "MCP HTTP response exceeded max_result_bytes";
        return false;
    }
    for (const auto & header : result.headers) {
        std::string name = header.first;
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (name == "mcp-session-id") {
            state->session_id = header.second;
            break;
        }
    }
    if (is_notification) {
        response = json::object();
        return true;
    }
    response = json::parse(result.body, nullptr, false);
    if (response.is_discarded() || !response.is_object()) {
        error = "MCP HTTP server returned invalid JSON-RPC payload";
        return false;
    }
    if (response.contains("error")) {
        error = "MCP HTTP JSON-RPC request failed: " + response["error"].dump();
        return false;
    }
    return true;
}

bool agent_mcp_http_client::list_tools(
        const agent_tool_context &, std::vector<mcp_agent_tool_definition> & tools, std::string & error) {
    if (!initialize(error)) return false;
    json response;
    if (!send_request("tools/list", json::object(), response, error) ||
            !response.contains("result") || !response["result"].is_object() ||
            !response["result"].contains("tools") || !response["result"]["tools"].is_array()) {
        if (error.empty()) error = "MCP HTTP tools/list response was invalid";
        return false;
    }
    tools.clear();
    for (const auto & item : response["result"]["tools"]) {
        mcp_agent_tool_definition definition;
        if (!parse_mcp_tool_definition(config.server_name, item, definition, error)) return false;
        if (!config.allowed_tools.empty() &&
                std::find(config.allowed_tools.begin(), config.allowed_tools.end(), definition.name) == config.allowed_tools.end()) {
            continue;
        }
        tools.push_back(std::move(definition));
    }
    return true;
}

bool agent_mcp_http_client::call_tool(
        const agent_tool_context & context,
        const mcp_agent_tool_definition & tool,
        const std::string & arguments_json,
        mcp_agent_tool_call_result & result,
        std::string & error) {
    if (!initialize(error, context.execution_control.deadline)) return false;
    const json arguments = json::parse(arguments_json, nullptr, false);
    if (arguments.is_discarded() || !arguments.is_object()) {
        error = "MCP HTTP tool arguments must be a JSON object";
        return false;
    }
    json response;
    if (!send_request("tools/call", make_mcp_tools_call_params(tool.name, arguments), response, error,
            context.execution_control.deadline) ||
            !response.contains("result")) return false;
    return parse_mcp_tool_call_result(response["result"], result, error);
}

bool agent_mcp_http_client::list_resources(
        std::vector<mcp_agent_resource_definition> & resources, std::string & error) {
    if (!initialize(error)) return false;
    json response;
    if (!send_request("resources/list", json::object(), response, error) ||
            !response.contains("result")) return false;
    return parse_mcp_resources_list_result(response["result"], resources, error);
}

bool agent_mcp_http_client::read_resource(
        const std::string & uri, mcp_agent_resource_read_result & result, std::string & error) {
    if (!initialize(error)) return false;
    json response;
    if (!send_request("resources/read", make_mcp_resources_read_params(uri), response, error) ||
            !response.contains("result")) return false;
    return parse_mcp_resource_read_result(response["result"], result, error);
}
