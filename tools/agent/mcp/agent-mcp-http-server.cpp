#include "agent-mcp-http-server.h"
#include "agent-mcp-protocol.h"
#include "agent-mcp-agent-tools.h"

#include <algorithm>
#include <charconv>

namespace {

agent_mcp_json json_rpc_error(const agent_mcp_json & id, int code, const std::string & message) {
    return agent_mcp_make_json_rpc_error(id, code, message);
}

agent_mcp_json make_event_notification(
        const common_agent_event_stream_delivery & delivery) {
    agent_mcp_json data = {
        {"delivery_kind", delivery.kind == common_agent_event_stream_delivery_kind::event
            ? "event"
            : delivery.kind == common_agent_event_stream_delivery_kind::heartbeat
                ? "heartbeat"
                : delivery.kind == common_agent_event_stream_delivery_kind::closed
                    ? "closed" : "overflow"},
        {"sequence", delivery.cursor.after_sequence},
    };
    if (delivery.kind == common_agent_event_stream_delivery_kind::event) {
        data["event_type"] = common_agent_daemon_event_type_name(delivery.event.event_type);
        data["event_category"] = common_agent_daemon_event_category_name(delivery.event.category);
        data["request_id"] = delivery.event.request_id;
        data["turn_id"] = delivery.event.turn_id;
        data["session_id"] = delivery.event.session_id;
        data["detail"] = delivery.event.detail;
    } else if (delivery.kind == common_agent_event_stream_delivery_kind::overflow) {
        data["overflow"] = {
            {"from_sequence", delivery.overflow_from_sequence},
            {"to_sequence", delivery.overflow_to_sequence},
            {"skipped_sequence_count", delivery.skipped_sequence_count},
        };
    }
    return {
        {"jsonrpc", "2.0"},
        {"method", "notifications/message"},
        {"params", {
            {"level", "info"},
            {"logger", "llama-agent"},
            {"data", std::move(data)},
        }},
    };
}

bool parse_event_cursor(
        const httplib::Request & request,
        uint64_t & cursor) {
    if (!request.has_header("Last-Event-ID")) {
        cursor = 0;
        return true;
    }
    const std::string value = request.get_header_value("Last-Event-ID");
    if (value.empty()) return false;
    const auto * begin = value.data();
    const auto * end = begin + value.size();
    const auto parsed = std::from_chars(begin, end, cursor);
    return parsed.ec == std::errc() && parsed.ptr == end;
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
    if (options_.tasks_enabled) {
        tasks_ = std::make_unique<agent_mcp_task_store>(
            options_.task_ttl_ms,
            options_.task_poll_interval_ms);
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
    if (!policy.caller_id.empty() && policy.caller_id != it->second.policy.caller_id) {
        response.status = 403;
        response.set_content(R"({"error":"MCP session belongs to another caller"})", "application/json");
        return false;
    }
    const std::string protocol_version = request.get_header_value("MCP-Protocol-Version");
    if (!common_mcp_is_supported_protocol_version(protocol_version) ||
            protocol_version != it->second.protocol_version) {
        response.status = 400;
        response.set_content(R"({"error":"invalid MCP protocol version for session"})", "application/json");
        return false;
    }
    policy = it->second.policy;
    return true;
}

std::string agent_mcp_http_server::make_session_id() {
    std::lock_guard<std::mutex> lock(session_mutex_);
    const std::string id = options_.server_name + "-session-" + std::to_string(next_session_id_++);
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
        const std::string configured_version = common_mcp_negotiate_protocol_version(options_.protocol_version);
        if (configured_version.empty()) {
            response = json_rpc_error(id, -32602, "server MCP protocol version is unsupported");
            return true;
        }
        const std::string requested_version = message.value("params", agent_mcp_json::object())
            .value("protocolVersion", "");
        const std::string negotiated_version = requested_version.empty()
            ? configured_version
            : common_mcp_negotiate_protocol_version(requested_version);
        if (negotiated_version.empty()) {
            response = json_rpc_error(id, -32602, "unsupported MCP protocol version");
            return true;
        }
        response = agent_mcp_make_json_rpc_result(id, {
            {"protocolVersion", negotiated_version},
            {"capabilities", {
                {"tools", agent_mcp_json::object()},
                {"resources", options_.list_resources ? agent_mcp_json::object() : agent_mcp_json()},
                {"tasks", options_.tasks_enabled && negotiated_version == "2025-11-25"
                    ? agent_mcp_json{{"requests", {{"tools", {{"call", agent_mcp_json::object()}}}}}}
                    : agent_mcp_json()},
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
        const auto params = message.value("params", agent_mcp_json::object());
        if (tasks_ && params.contains("task") && params["task"].is_object()) {
            const uint64_t requested_ttl = params["task"].value("ttl", options_.task_ttl_ms);
            agent_mcp_json task_message = message;
            task_message["params"].erase("task");
            const std::string task_id = tasks_->create(
                [this, task_message, policy]() mutable {
                    agent_mcp_json execution;
                    std::string task_error;
                    if (!handle_message(task_message, policy, execution, task_error)) {
                        return agent_mcp_json{{"isError", true}, {"error", task_error}};
                    }
                    if (execution.contains("result")) return execution["result"];
                    return agent_mcp_json{{"isError", true}, {"error", execution.value("error", "task failed")}};
                },
                requested_ttl);
            agent_mcp_task_snapshot snapshot;
            tasks_->get(task_id, snapshot);
            response = agent_mcp_make_json_rpc_result(id, {
                {"task", agent_mcp_render_task_snapshot(snapshot)},
            });
            return true;
        }
        agent_mcp_server_tool_registry registry;
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            registry = registry_;
        }
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
    if (method == "tasks/get" || method == "tasks/result" || method == "tasks/cancel" || method == "tasks/list") {
        if (!tasks_) {
            response = json_rpc_error(id, -32601, "MCP Tasks are not enabled");
            return true;
        }
        const auto params = message.value("params", agent_mcp_json::object());
        if (method == "tasks/list") {
            agent_mcp_json items = agent_mcp_json::array();
            for (const auto & task : tasks_->list()) items.push_back(agent_mcp_render_task_snapshot(task));
            response = agent_mcp_make_json_rpc_result(id, {{"tasks", std::move(items)}});
            return true;
        }
        const std::string task_id = params.value("taskId", "");
        if (task_id.empty()) {
            response = json_rpc_error(id, -32602, "taskId is required");
            return true;
        }
        agent_mcp_task_snapshot snapshot;
        if (method == "tasks/cancel") {
            if (!tasks_->cancel(task_id) || !tasks_->get(task_id, snapshot)) {
                response = json_rpc_error(id, -32001, "task not found");
                return true;
            }
            response = agent_mcp_make_json_rpc_result(id, agent_mcp_render_task_snapshot(snapshot));
            return true;
        }
        if (method == "tasks/result") {
            if (!tasks_->wait_result(task_id, snapshot)) {
                response = json_rpc_error(id, -32001, "task not found");
                return true;
            }
            if (snapshot.status == agent_mcp_task_status::cancelled) {
                response = json_rpc_error(id, -32800, "task cancelled");
            } else {
                response = agent_mcp_make_json_rpc_result(id, snapshot.result);
            }
            return true;
        }
        if (!tasks_->get(task_id, snapshot)) {
            response = json_rpc_error(id, -32001, "task not found");
            return true;
        }
        response = agent_mcp_make_json_rpc_result(id, agent_mcp_render_task_snapshot(snapshot));
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
            const std::string protocol_version = result.value("result", agent_mcp_json::object())
                .value("protocolVersion", "");
            {
                std::lock_guard<std::mutex> lock(session_mutex_);
                sessions_[session_id] = {policy, protocol_version};
            }
            response.set_header("Mcp-Session-Id", session_id);
            response.set_header("MCP-Protocol-Version", protocol_version);
        }
        response.set_content(body, "application/json");
    });

    server_.Get(options_.path, [this](const httplib::Request & request, httplib::Response & response) {
        agent_mcp_caller_policy policy;
        if (!authorize(request, response, policy)) return;
        if (!validate_session(request, response, policy)) return;
        if (!request.has_header("Accept") ||
                request.get_header_value("Accept").find("text/event-stream") == std::string::npos) {
            response.status = 406;
            response.set_content(R"({"error":"SSE requires Accept: text/event-stream"})", "application/json");
            return;
        }
        if (!options_.subscribe_events || !options_.unsubscribe_events || !options_.wait_for_event) {
            response.status = 405;
            response.set_header("Allow", "POST, GET, DELETE");
            return;
        }

        common_agent_event_stream_subscription subscription;
        uint64_t after_sequence = 0;
        if (!parse_event_cursor(request, after_sequence)) {
            response.status = 400;
            response.set_content(R"({"error":"invalid Last-Event-ID"})", "application/json");
            return;
        }
        subscription.filter.namespace_id = policy.namespace_id;
        subscription.filter.project_id = policy.project_id;
        subscription.cursor.after_sequence = after_sequence;
        subscription.max_pending_events = 256;
        const std::string subscription_id = options_.subscribe_events(std::move(subscription));
        if (subscription_id.empty()) {
            response.status = 503;
            response.set_content(R"({"error":"event stream is unavailable"})", "application/json");
            return;
        }
        response.set_header("Cache-Control", "no-cache");
        response.set_header("Connection", "keep-alive");
        response.set_chunked_content_provider(
            "text/event-stream",
            [this, subscription_id](size_t, httplib::DataSink & sink) {
                common_agent_event_stream_delivery delivery;
                const auto status = options_.wait_for_event(
                    subscription_id,
                    delivery,
                    std::chrono::milliseconds(15000));
                if (status == common_agent_event_stream_wait_status::timeout) {
                    return sink.write(": heartbeat\n\n", 13);
                }
                if (status == common_agent_event_stream_wait_status::not_found ||
                        status == common_agent_event_stream_wait_status::closed) {
                    return false;
                }
                const std::string payload = make_event_notification(delivery).dump();
                const std::string message =
                    "id: " + std::to_string(delivery.cursor.after_sequence) +
                    "\nevent: message\ndata: " + payload + "\n\n";
                return sink.write(message.data(), message.size());
            },
            [this, subscription_id](bool) {
                options_.unsubscribe_events(subscription_id);
            });
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
        if (!policy.caller_id.empty() && policy.caller_id != it->second.policy.caller_id) {
            response.status = 403;
            return;
        }
        const std::string protocol_version = request.get_header_value("MCP-Protocol-Version");
        if (!common_mcp_is_supported_protocol_version(protocol_version) ||
                protocol_version != it->second.protocol_version) {
            response.status = 400;
            response.set_content(R"({"error":"invalid MCP protocol version for session"})", "application/json");
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
