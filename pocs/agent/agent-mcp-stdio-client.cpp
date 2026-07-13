#include "agent-mcp-protocol.h"
#include "agent-tool-provider.h"

#include <sheredom/subprocess.h>

#include <cstdio>
#include <cstring>
#include <chrono>
#include <sstream>
#include <thread>
#include <utility>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using json = nlohmann::ordered_json;

namespace {

std::vector<char *> to_cstr_vec(const std::vector<std::string> & values) {
    std::vector<char *> result;
    result.reserve(values.size() + 1);
    for (const auto & value : values) {
        result.push_back(const_cast<char *>(value.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

bool read_header_line(FILE * stream, std::string & line) {
    line.clear();
    for (;;) {
        const int ch = std::fgetc(stream);
        if (ch == EOF) {
            return !line.empty();
        }
        if (ch == '\n') {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            return true;
        }
        line.push_back(static_cast<char>(ch));
    }
}

bool read_json_rpc_message(FILE * stream, json & message, std::string & error) {
    message = json();
    error.clear();

    size_t content_length = 0;
    bool have_content_length = false;
    std::string line;
    while (read_header_line(stream, line)) {
        if (line.empty()) {
            break;
        }

        constexpr const char * prefix = "Content-Length:";
        if (line.rfind(prefix, 0) == 0) {
            std::string value = line.substr(std::strlen(prefix));
            while (!value.empty() && value.front() == ' ') {
                value.erase(value.begin());
            }
            content_length = static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
            have_content_length = true;
        }
    }

    if (!have_content_length) {
        error = "MCP server response missing Content-Length header";
        return false;
    }

    std::string body(content_length, '\0');
    const size_t read_count = std::fread(body.data(), 1, content_length, stream);
    if (read_count != content_length) {
        error = "failed to read complete MCP server response body";
        return false;
    }

    const auto parsed = json::parse(body, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        error = "MCP server returned invalid JSON-RPC payload";
        return false;
    }

    message = parsed;
    return true;
}

bool write_json_rpc_message(FILE * stream, const json & message, std::string & error) {
    error.clear();
    const std::string body = message.dump();
    const std::string framed =
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;

    if (std::fwrite(framed.data(), 1, framed.size(), stream) != framed.size()) {
        error = "failed to write MCP request";
        return false;
    }
    if (std::fflush(stream) != 0) {
        error = "failed to flush MCP request";
        return false;
    }
    return true;
}

void append_tail(std::string & tail, const char * data, size_t size, size_t max_size) {
    if (data == nullptr || size == 0 || max_size == 0) {
        return;
    }

    tail.append(data, size);
    if (tail.size() > max_size) {
        tail.erase(0, tail.size() - max_size);
    }
}

} // namespace

struct agent_mcp_stdio_client::impl {
    subprocess_s proc{};
    FILE * in = nullptr;
    FILE * out = nullptr;
    FILE * err = nullptr;
    int next_request_id = 1;
    int exit_code = 1;
    bool running = false;
    bool initialized = false;
    bool joined = false;
    std::string stderr_tail;
    static constexpr size_t max_stderr_tail_bytes = 4096;
};

agent_mcp_stdio_client::agent_mcp_stdio_client(agent_mcp_stdio_client_config config)
    : config(std::move(config))
    , state(std::make_unique<impl>()) {}

agent_mcp_stdio_client::~agent_mcp_stdio_client() {
    shutdown_process();
}

bool agent_mcp_stdio_client::ensure_started(std::string & error) {
    error.clear();
    if (state->initialized) {
        return true;
    }

    if (!state->running) {
        if (config.command_line.empty()) {
            error = "MCP stdio client requires a server command line";
            return false;
        }

        auto argv = to_cstr_vec(config.command_line);
        const int options =
            subprocess_option_no_window |
            subprocess_option_enable_async |
            subprocess_option_inherit_environment;

        if (subprocess_create(argv.data(), options, &state->proc) != 0) {
            error = "failed to spawn MCP stdio server process";
            return false;
        }

        state->in = subprocess_stdin(&state->proc);
        state->out = subprocess_stdout(&state->proc);
        state->err = subprocess_stderr(&state->proc);
        if (!state->in || !state->out || !state->err) {
            error = "failed to acquire MCP stdio pipes";
            shutdown_process();
            return false;
        }
#ifdef _WIN32
        _setmode(_fileno(state->in), _O_BINARY);
        _setmode(_fileno(state->out), _O_BINARY);
#endif
        state->running = true;
        state->joined = false;
        state->stderr_tail.clear();
    }

    json response;
    if (!send_request("initialize", make_mcp_initialize_params(), response, error)) {
        return false;
    }

    if (!response.contains("result") || !response["result"].is_object()) {
        error = "MCP initialize response did not contain a result object";
        return false;
    }

    if (!send_notification("notifications/initialized", json::object(), error)) {
        return false;
    }

    state->initialized = true;
    return true;
}

bool agent_mcp_stdio_client::send_notification(
        const std::string & method,
        const json & params,
        std::string & error) {
    return write_json_rpc_message(
        state->in,
        make_mcp_jsonrpc_notification(method, params),
        error);
}

bool agent_mcp_stdio_client::send_request(
        const std::string & method,
        const json & params,
        json & response,
        std::string & error) {
    response = json();
    if (!state->running || !state->in || !state->out) {
        error = "MCP stdio client is not running";
        return false;
    }

    const int request_id = state->next_request_id++;
    if (!write_json_rpc_message(
            state->in,
            make_mcp_jsonrpc_request(request_id, method, params),
            error)) {
        return false;
    }

    for (;;) {
        json message;
        if (!read_json_rpc_message(state->out, message, error)) {
            collect_stderr_tail();
            capture_exit_if_needed();
            error = with_transport_context(error);
            return false;
        }

        if (message.contains("id") && message["id"] == request_id) {
            if (message.contains("error")) {
                collect_stderr_tail();
                capture_exit_if_needed();
                error = with_transport_context(
                    "MCP server returned JSON-RPC error: " + message["error"].dump());
                return false;
            }
            response = std::move(message);
            return true;
        }
    }
}

void agent_mcp_stdio_client::shutdown_process() {
    if (!state) {
        return;
    }

    if (state->running) {
        std::string ignored_error;
        json ignored_response;
        if (state->initialized) {
            send_request("shutdown", json::object(), ignored_response, ignored_error);
            send_notification("exit", json::object(), ignored_error);
        }
        if (state->in != nullptr) {
            std::fclose(state->in);
            state->in = nullptr;
        }
        collect_stderr_tail();
        if (!state->joined) {
            if (subprocess_alive(&state->proc)) {
                subprocess_terminate(&state->proc);
            }
            subprocess_join(&state->proc, &state->exit_code);
            state->joined = true;
        }
        collect_stderr_tail();
        subprocess_destroy(&state->proc);
    }

    state->running = false;
    state->initialized = false;
    state->out = nullptr;
    state->err = nullptr;
}

void agent_mcp_stdio_client::collect_stderr_tail() {
    if (!state || !state->running) {
        return;
    }

    char buffer[512];
    for (;;) {
        const unsigned count = subprocess_read_stderr(&state->proc, buffer, sizeof(buffer));
        if (count == 0) {
            break;
        }
        append_tail(state->stderr_tail, buffer, count, impl::max_stderr_tail_bytes);
        if (count < sizeof(buffer)) {
            break;
        }
    }
}

void agent_mcp_stdio_client::capture_exit_if_needed() {
    if (!state || !state->running || state->joined) {
        return;
    }

    if (subprocess_alive(&state->proc)) {
        for (int attempt = 0; attempt < 5; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            collect_stderr_tail();
            if (!subprocess_alive(&state->proc)) {
                break;
            }
        }
        if (subprocess_alive(&state->proc)) {
            return;
        }
    }

    collect_stderr_tail();
    subprocess_join(&state->proc, &state->exit_code);
    state->joined = true;
}

std::string agent_mcp_stdio_client::with_transport_context(const std::string & base_error) const {
    std::ostringstream oss;
    oss << base_error;

    if (state) {
        if (state->joined) {
            oss << " (exit_code=" << state->exit_code << ")";
        } else if (state->running && !subprocess_alive(&state->proc)) {
            oss << " (server exited)";
        }

        if (!state->stderr_tail.empty()) {
            oss << " stderr_tail=" << json(state->stderr_tail).dump();
        }
    }

    return oss.str();
}

bool agent_mcp_stdio_client::list_tools(
        const agent_tool_context & context,
        std::vector<mcp_agent_tool_definition> & tools,
        std::string & error) {
    tools.clear();
    if (context.execution_control.should_stop()) {
        error = context.execution_control.stop_reason();
        return false;
    }
    if (!ensure_started(error)) {
        return false;
    }

    json response;
    if (!send_request("tools/list", json::object(), response, error)) {
        return false;
    }

    const auto & result = response["result"];
    if (!result.is_object() || !result.contains("tools") || !result["tools"].is_array()) {
        collect_stderr_tail();
        capture_exit_if_needed();
        error = with_transport_context("MCP tools/list response did not contain a tools array");
        return false;
    }

    for (const auto & item : result["tools"]) {
        mcp_agent_tool_definition definition;
        std::string definition_error;
        if (!parse_mcp_tool_definition(config.server_name, item, definition, definition_error)) {
            continue;
        }
        tools.push_back(std::move(definition));
    }

    error.clear();
    return true;
}

bool agent_mcp_stdio_client::call_tool(
        const agent_tool_context & context,
        const mcp_agent_tool_definition & tool,
        const std::string & arguments_json,
        mcp_agent_tool_call_result & result,
        std::string & error) {
    result = {};
    if (context.execution_control.should_stop()) {
        error = context.execution_control.stop_reason();
        return false;
    }
    if (!ensure_started(error)) {
        return false;
    }

    json arguments = json::object();
    if (!arguments_json.empty()) {
        arguments = json::parse(arguments_json, nullptr, false);
        if (arguments.is_discarded() || !arguments.is_object()) {
            error = "MCP tool arguments must be a JSON object";
            return false;
        }
    }

    json response;
    if (!send_request(
            "tools/call",
            make_mcp_tools_call_params(tool.name, arguments),
            response,
            error)) {
        return false;
    }

    const auto & rpc_result = response["result"];
    if (!parse_mcp_tool_call_result(rpc_result, result, error)) {
        collect_stderr_tail();
        capture_exit_if_needed();
        error = with_transport_context(error);
        return false;
    }

    error.clear();
    return true;
}

bool agent_mcp_stdio_client::list_resources(
        std::vector<mcp_agent_resource_definition> & resources,
        std::string & error) {
    resources.clear();
    if (!ensure_started(error)) {
        return false;
    }

    json response;
    if (!send_request("resources/list", json::object(), response, error)) {
        return false;
    }

    const auto & rpc_result = response["result"];
    if (!parse_mcp_resources_list_result(rpc_result, resources, error)) {
        collect_stderr_tail();
        capture_exit_if_needed();
        error = with_transport_context(error);
        return false;
    }

    error.clear();
    return true;
}

bool agent_mcp_stdio_client::read_resource(
        const std::string & uri,
        mcp_agent_resource_read_result & result,
        std::string & error) {
    result = {};
    if (!ensure_started(error)) {
        return false;
    }

    json response;
    if (!send_request(
            "resources/read",
            make_mcp_resources_read_params(uri),
            response,
            error)) {
        return false;
    }

    const auto & rpc_result = response["result"];
    if (!parse_mcp_resource_read_result(rpc_result, result, error)) {
        collect_stderr_tail();
        capture_exit_if_needed();
        error = with_transport_context(error);
        return false;
    }

    error.clear();
    return true;
}
