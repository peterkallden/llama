#include "agent-mcp-protocol.h"
#include "../tooling/agent-tool-provider.h"

#include <sheredom/subprocess.h>

#include <cstdio>
#include <cstring>
#include <chrono>
#include <mutex>
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

bool try_parse_buffered_json_rpc_message(
        std::string & buffer,
        json & message,
        std::string & error) {
    message = json();
    error.clear();

    const size_t header_end = buffer.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        return false;
    }

    size_t content_length = 0;
    bool have_content_length = false;
    size_t line_begin = 0;
    while (line_begin < header_end) {
        size_t line_end = buffer.find("\r\n", line_begin);
        if (line_end == std::string::npos || line_end > header_end) {
            line_end = header_end;
        }
        std::string line = buffer.substr(line_begin, line_end - line_begin);
        constexpr const char * prefix = "Content-Length:";
        if (line.rfind(prefix, 0) == 0) {
            std::string value = line.substr(std::strlen(prefix));
            while (!value.empty() && value.front() == ' ') {
                value.erase(value.begin());
            }
            content_length = static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
            have_content_length = true;
        }
        line_begin = line_end + 2;
    }

    if (!have_content_length) {
        error = "MCP server response missing Content-Length header";
        return true;
    }

    const size_t body_begin = header_end + 4;
    const size_t frame_end = body_begin + content_length;
    if (buffer.size() < frame_end) {
        return false;
    }

    const std::string body = buffer.substr(body_begin, content_length);
    const auto parsed = json::parse(body, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        error = "MCP server returned invalid JSON-RPC payload";
        buffer.erase(0, frame_end);
        return true;
    }

    message = parsed;
    buffer.erase(0, frame_end);
    return true;
}

bool read_json_rpc_message_from_subprocess(
        subprocess_s & process,
        std::string & buffer,
        json & message,
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline) {
    for (;;) {
        if (try_parse_buffered_json_rpc_message(buffer, message, error)) {
            return error.empty();
        }
        if (deadline.has_value() && std::chrono::steady_clock::now() >= *deadline) {
            error = "MCP request timeout";
            return false;
        }

        char chunk[4096];
        const unsigned count = subprocess_read_stdout(&process, chunk, sizeof(chunk));
        if (count > 0) {
            buffer.append(chunk, count);
            continue;
        }
        if (!subprocess_alive(&process)) {
            if (buffer.empty()) {
                error = "MCP server response missing Content-Length header";
            } else {
                error = "failed to read complete MCP server response body";
            }
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
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

std::optional<std::chrono::steady_clock::time_point> mcp_request_deadline(
        const agent_tool_context & context,
        uint32_t configured_timeout_ms) {
    std::optional<std::chrono::steady_clock::time_point> deadline =
        context.execution_control.deadline;
    if (configured_timeout_ms > 0) {
        const auto configured = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(configured_timeout_ms);
        if (!deadline.has_value() || configured < *deadline) {
            deadline = configured;
        }
    }
    return deadline;
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
    std::string stdout_buffer;
    std::string stderr_tail;
    std::mutex request_mutex;
    static constexpr size_t max_stderr_tail_bytes = 4096;
};

agent_mcp_stdio_client::agent_mcp_stdio_client(agent_mcp_stdio_client_config config)
    : config(std::move(config))
    , state(std::make_unique<impl>()) {}

agent_mcp_stdio_client::~agent_mcp_stdio_client() {
    shutdown_process();
}

bool agent_mcp_stdio_client::ensure_started(
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline) {
    error.clear();
    deadline = effective_deadline(deadline);
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
            subprocess_option_enable_async_no_wait |
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
        state->stdout_buffer.clear();
        state->stderr_tail.clear();
    }

    json response;
    if (!send_request("initialize", make_mcp_initialize_params(), response, error, deadline)) {
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
    if (!state->running || !state->in) {
        error = "MCP stdio client is not writable";
        return false;
    }
    return write_json_rpc_message(
        state->in,
        make_mcp_jsonrpc_notification(method, params),
        error);
}

bool agent_mcp_stdio_client::send_request(
        const std::string & method,
        const json & params,
        json & response,
        std::string & error,
        std::optional<std::chrono::steady_clock::time_point> deadline) {
    std::lock_guard<std::mutex> request_lock(state->request_mutex);
    response = json();
    deadline = effective_deadline(deadline);
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
        bool read_ok = false;
        std::string read_error;
        read_ok = read_json_rpc_message_from_subprocess(
            state->proc,
            state->stdout_buffer,
            message,
            read_error,
            deadline);
        if (!read_ok && read_error == "MCP request timeout") {
            subprocess_terminate(&state->proc);
            state->initialized = false;
            collect_stderr_tail();
            const auto terminate_deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(config.shutdown_timeout_ms > 0 ? config.shutdown_timeout_ms : 100);
            if (terminate_process_until(terminate_deadline)) {
                state->joined = true;
            }
            subprocess_destroy(&state->proc);
            state->running = false;
            state->in = nullptr;
            state->out = nullptr;
            state->err = nullptr;
            error = "MCP request timeout; server process terminated";
            return false;
        }
        if (!read_ok) {
            error = read_error;
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

std::optional<std::chrono::steady_clock::time_point> agent_mcp_stdio_client::effective_deadline(
        std::optional<std::chrono::steady_clock::time_point> deadline) const {
    if (deadline.has_value() || config.request_timeout_ms == 0) {
        return deadline;
    }
    return std::chrono::steady_clock::now() + std::chrono::milliseconds(config.request_timeout_ms);
}

bool agent_mcp_stdio_client::terminate_process_until(std::chrono::steady_clock::time_point deadline) {
    if (!state || !state->running || state->joined) {
        return true;
    }

    while (std::chrono::steady_clock::now() < deadline) {
        if (!subprocess_alive(&state->proc)) {
            subprocess_join(&state->proc, &state->exit_code);
            state->joined = true;
            // subprocess_join() closes the child's stdin stream.
            state->in = nullptr;
            return true;
        }
        collect_stderr_tail();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    return false;
}

void agent_mcp_stdio_client::shutdown_process() {
    if (!state) {
        return;
    }

    if (state->running) {
        std::string ignored_error;
        json ignored_response;
        if (state->initialized) {
            std::optional<std::chrono::steady_clock::time_point> shutdown_deadline;
            if (config.shutdown_timeout_ms > 0) {
                shutdown_deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(config.shutdown_timeout_ms);
            }
            send_request("shutdown", json::object(), ignored_response, ignored_error, shutdown_deadline);
            send_notification("exit", json::object(), ignored_error);
        }
        if (state->in != nullptr) {
            std::fclose(state->in);
            state->in = nullptr;
            // subprocess_destroy() owns this field as well.  Clear it after
            // closing the stream so the underlying FILE is not freed twice.
            state->proc.stdin_file = nullptr;
        }
        collect_stderr_tail();
        if (!state->joined) {
            if (subprocess_alive(&state->proc)) {
                subprocess_terminate(&state->proc);
            }
            const auto terminate_deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(config.shutdown_timeout_ms > 0 ? config.shutdown_timeout_ms : 100);
            state->joined = terminate_process_until(terminate_deadline);
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
    // subprocess_join() closes the child's stdin stream.
    state->in = nullptr;
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
    const auto deadline = mcp_request_deadline(
        context,
        context.execution_control.timeout_policy.mcp_request_timeout_ms > 0
            ? context.execution_control.timeout_policy.mcp_request_timeout_ms
            : config.request_timeout_ms);
    if (!ensure_started(error, deadline)) {
        return false;
    }

    json response;
    if (!send_request("tools/list", json::object(), response, error, deadline)) {
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
    const auto deadline = mcp_request_deadline(
        context,
        context.execution_control.timeout_policy.mcp_request_timeout_ms > 0
            ? context.execution_control.timeout_policy.mcp_request_timeout_ms
            : config.request_timeout_ms);
    if (!ensure_started(error, deadline)) {
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
            error,
            deadline)) {
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
