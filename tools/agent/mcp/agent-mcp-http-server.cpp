#include "agent-mcp-http-server.h"
#include "agent-mcp-agent-tools.h"

#include <algorithm>

namespace {

agent_mcp_json json_rpc_error(const agent_mcp_json & id, int code, const std::string & message) {
    return agent_mcp_make_json_rpc_error(id, code, message);
}

} // namespace

agent_mcp_http_server::agent_mcp_http_server(
        agent_mcp_server_tool_registry registry,
        agent_mcp_http_server_options options)
    : registry_(std::move(registry)), options_(std::move(options)) {
    if (options_.agent_tools_enabled) {
        std::string error;
        agent_mcp_register_agent_tools(registry_, {
            options_.max_body_bytes,
            options_.max_result_bytes,
            options_.max_delegation_depth,
        }, error);
    }
    install_routes();
}

bool agent_mcp_http_server::replace_registry(
        agent_mcp_server_tool_registry registry,
        std::string & error) {
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        registry_ = std::move(registry);
    }
    error.clear();
    return true;
}

void agent_mcp_http_server::replace_default_policy(
        agent_mcp_caller_policy policy) {
    std::lock_guard<std::mutex> lock(policy_mutex_);
    options_.default_policy = std::move(policy);
}

void agent_mcp_http_server::replace_authenticator(
        std::shared_ptr<const agent_mcp_authenticator> authenticator) {
    std::lock_guard<std::mutex> lock(policy_mutex_);
    options_.authenticator = std::move(authenticator);
}

bool agent_mcp_http_server::authorize(
        const httplib::Request & request,
        httplib::Response & response,
        agent_mcp_caller_policy & policy) const {
    const std::string origin = request.get_header_value("Origin");
    if (!origin.empty() && (options_.allowed_origin.empty() || origin != options_.allowed_origin)) {
        response.status = 403;
        response.set_content(R"({"error":"origin is not allowed"})", "application/json");
        return false;
    }
    std::string error;
    std::shared_ptr<const agent_mcp_authenticator> authenticator;
    {
        std::lock_guard<std::mutex> lock(policy_mutex_);
        authenticator = options_.authenticator;
    }
    if (authenticator) {
        if (!authenticator->authenticate(
                {request.get_header_value("Authorization")}, policy, error)) {
            response.status = 401;
            response.set_header("WWW-Authenticate", "Bearer");
            response.set_content(R"({"error":"unauthorized"})", "application/json");
            return false;
        }
    } else if (options_.bearer_token.empty() ||
            request.get_header_value("Authorization") != "Bearer " + options_.bearer_token) {
        response.status = 401;
        response.set_header("WWW-Authenticate", "Bearer");
        response.set_content(R"({"error":"unauthorized"})", "application/json");
        return false;
    } else {
        std::lock_guard<std::mutex> lock(policy_mutex_);
        policy = options_.default_policy;
    }
    return true;
}

bool agent_mcp_http_server::validate_session(
        const httplib::Request & request,
        httplib::Response & response,
        agent_mcp_caller_policy & policy) const {
    const std::string session_id = request.get_header_value("Mcp-Session-Id");
    if (session_id.empty()) {
        response.status = 400;
        response.set_content(R"({"error":"missing MCP session"})", "application/json");
        return false;
    }
    std::lock_guard<std::mutex> lock(session_mutex_);
    const auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        response.status = 404;
        response.set_content(R"({"error":"unknown MCP session"})", "application/json");
        return false;
    }
    if (!policy.caller_id.empty() && policy.caller_id != it->second.caller_id) {
        response.status = 403;
        response.set_content(R"({"error":"MCP session belongs to another caller"})", "application/json");
        return false;
    }
    policy = it->second;
    return true;
}

std::string agent_mcp_http_server::make_session_id() {
    std::lock_guard<std::mutex> lock(session_mutex_);
    const std::string id = options_.server_name + "-session-" + std::to_string(next_session_id_++);
    sessions_.emplace(id, agent_mcp_caller_policy{});
    return id;
}

bool agent_mcp_http_server::handle_message(
        const agent_mcp_json & message,
        const agent_mcp_caller_policy & policy,
        agent_mcp_json & response,
        std::string & error) {
    if (!message.is_object() || message.value("jsonrpc", "") != "2.0") {
        error = "invalid JSON-RPC message";
        return false;
    }
    const auto id = message.value("id", agent_mcp_json());
    const std::string method = message.value("method", "");
    if (method.empty()) {
        response = json_rpc_error(id, -32600, "method is required");
        return true;
    }
    if (method.rfind("notifications/", 0) == 0) {
        response = agent_mcp_json();
        return true;
    }

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
        return true;
    }
    if (method == "tools/list") {
        agent_mcp_server_tool_registry registry;
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            registry = registry_;
        }
        agent_mcp_json tools = agent_mcp_json::array();
        for (const auto & tool : registry.list_tools()) {
            if (!agent_mcp_policy_allows_tool(
                    policy, tool.name, tool.read_only, tool.requires_confirmation)) continue;
            const auto schema = agent_mcp_json::parse(tool.input_schema_json, nullptr, false);
            tools.push_back({
                {"name", tool.name},
                {"description", tool.description},
                {"inputSchema", schema.is_discarded() ? agent_mcp_json::object() : schema},
            });
        }
        response = agent_mcp_make_json_rpc_result(id, {{"tools", std::move(tools)}});
        return true;
    }
    if (method == "resources/list") {
        if (!options_.list_resources) {
            response = json_rpc_error(id, -32601, "method not found");
            return true;
        }
        agent_mcp_json result;
        if (!options_.list_resources(result, error)) {
            response = json_rpc_error(id, -32000, error.empty() ? "resources/list failed" : error);
        } else {
            response = agent_mcp_make_json_rpc_result(id, std::move(result));
        }
        return true;
    }
    if (method == "resources/read") {
        if (!options_.read_resource) {
            response = json_rpc_error(id, -32601, "method not found");
            return true;
        }
        agent_mcp_json result;
        const auto params = message.value("params", agent_mcp_json::object());
        if (!options_.read_resource(params, result, error)) {
            response = json_rpc_error(id, -32000, error.empty() ? "resources/read failed" : error);
        } else {
            response = agent_mcp_make_json_rpc_result(id, std::move(result));
        }
        return true;
    }
    if (method == "tools/call") {
        agent_mcp_server_tool_registry registry;
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            registry = registry_;
        }
        const auto params = message.value("params", agent_mcp_json::object());
        const std::string name = params.value("name", "");
        const auto arguments = params.value("arguments", agent_mcp_json::object());
        agent_mcp_server_tool_result result;
        const auto listed_tools = registry.list_tools();
        const auto listed_tool = std::find_if(
            listed_tools.begin(), listed_tools.end(),
            [&name](const agent_mcp_server_tool & tool) { return tool.name == name; });
        const bool caller_allowed = listed_tool != listed_tools.end() &&
            agent_mcp_policy_allows_tool(
                policy,
                name,
                listed_tool->read_only,
                listed_tool->requires_confirmation);
        if (!caller_allowed) {
            error = "MCP tool is not allowed for caller policy: " + name;
            result.ok = false;
            result.failure_code = "tool.forbidden";
            result.failure_class = "policy";
            result.safe_summary = error;
            result.content = agent_mcp_json::array({{{"type", "text"}, {"text", result.safe_summary}}});
        } else if (!registry.contains_tool(name)) {
            error = "unknown MCP server tool: " + name;
            result.ok = false;
            result.failure_code = "tool.not_found";
            result.failure_class = "not_found";
            result.safe_summary = error;
            result.content = agent_mcp_json::array({{{"type", "text"}, {"text", result.safe_summary}}});
        } else if (!registry.validate_tool_arguments(name, arguments, error)) {
            result.ok = false;
            result.failure_code = "tool.invalid_arguments";
            result.failure_class = "validation";
            result.safe_summary = error;
            result.content = agent_mcp_json::array({{{"type", "text"}, {"text", result.safe_summary}}});
        } else if (agent_mcp_is_agent_tool(name) && options_.execute_agent_tool) {
            const auto depth = arguments.value("delegation_depth", 0U);
            if (depth >= options_.max_delegation_depth) {
                error = "agent delegation depth limit exceeded";
                result.ok = false;
                result.failure_code = "agent.delegation_depth_exceeded";
                result.failure_class = "policy";
                result.safe_summary = error;
                result.content = agent_mcp_json::array({{{"type", "text"}, {"text", error}}});
            } else {
                if (!options_.execute_agent_tool(policy, name, arguments, result, error)) {
                    result.ok = false;
                    result.failure_code = result.failure_code.empty() ? "agent.delegation_failed" : result.failure_code;
                    result.failure_class = result.failure_class.empty() ? "execution" : result.failure_class;
                    result.safe_summary = result.safe_summary.empty() ? error : result.safe_summary;
                    result.content = agent_mcp_json::array({{{"type", "text"}, {"text", result.safe_summary}}});
                }
            }
        } else if (options_.execute_tool) {
            options_.execute_tool(policy, name, arguments, result, error);
        } else if (!registry.call_tool(name, arguments, result, error)) {
            result.ok = false;
            result.failure_code = error.rfind("unknown MCP server tool:", 0) == 0
                ? "tool.not_found" : "tool.invalid_arguments";
            result.failure_class = result.failure_code == "tool.not_found" ? "not_found" : "validation";
            result.safe_summary = error.empty() ? "MCP tool call is invalid." : error;
            result.content = agent_mcp_json::array({{{"type", "text"}, {"text", result.safe_summary}}});
        }
        response = agent_mcp_make_json_rpc_result(id, agent_mcp_render_tool_call_result(result));
        return true;
    }

    response = json_rpc_error(id, -32601, "method not found");
    return true;
}

void agent_mcp_http_server::install_routes() {
    server_.set_payload_max_length(options_.max_body_bytes);
    server_.Post(options_.path, [this](const httplib::Request & request, httplib::Response & response) {
        agent_mcp_caller_policy policy;
        if (!authorize(request, response, policy)) return;
        const auto message = agent_mcp_json::parse(request.body, nullptr, false);
        if (message.is_discarded() || !message.is_object()) {
            response.status = 400;
            response.set_content(R"({"error":"invalid JSON body"})", "application/json");
            return;
        }
        const bool is_initialize = message.value("method", "") == "initialize";
        if (!is_initialize && !validate_session(request, response, policy)) return;
        agent_mcp_json result;
        std::string error;
        if (!handle_message(message, policy, result, error)) {
            response.status = 400;
            response.set_content(agent_mcp_json{{"error", error}}.dump(), "application/json");
            return;
        }
        if (result.is_null()) {
            response.status = 204;
            return;
        }
        const std::string body = result.dump();
        if (body.size() > options_.max_result_bytes) {
            response.status = 413;
            response.set_content(R"({"error":"MCP result exceeds configured limit"})", "application/json");
            return;
        }
        if (message.value("method", "") == "initialize") {
            const std::string session_id = make_session_id();
            {
                std::lock_guard<std::mutex> lock(session_mutex_);
                sessions_[session_id] = policy;
            }
            response.set_header("Mcp-Session-Id", session_id);
        }
        response.set_content(body, "application/json");
    });

    server_.Delete(options_.path, [this](const httplib::Request & request, httplib::Response & response) {
        agent_mcp_caller_policy policy;
        if (!authorize(request, response, policy)) return;
        const std::string session_id = request.get_header_value("Mcp-Session-Id");
        std::lock_guard<std::mutex> lock(session_mutex_);
        const auto it = sessions_.find(session_id);
        if (session_id.empty() || it == sessions_.end()) {
            response.status = 404;
            return;
        }
        if (!policy.caller_id.empty() && policy.caller_id != it->second.caller_id) {
            response.status = 403;
            return;
        }
        sessions_.erase(it);
        response.status = 204;
    });
}

bool agent_mcp_http_server::bind(std::string & error) {
    if (options_.bearer_token.empty() && !options_.authenticator) {
        error = "inbound MCP HTTP server requires a bearer token";
        return false;
    }
    const int bound_port = options_.port == 0
        ? server_.bind_to_any_port(options_.listen_address)
        : (server_.bind_to_port(options_.listen_address, options_.port) ? options_.port : -1);
    if (bound_port <= 0) {
        error = "failed to bind inbound MCP HTTP server";
        return false;
    }
    port_ = bound_port;
    error.clear();
    return true;
}

bool agent_mcp_http_server::listen(std::string & error) {
    if (port_ <= 0 && !bind(error)) return false;
    if (!server_.listen_after_bind()) {
        error = "inbound MCP HTTP server stopped unexpectedly";
        return false;
    }
    error.clear();
    return true;
}

void agent_mcp_http_server::stop() {
    server_.stop();
}
