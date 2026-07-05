#include "agent-tool-provider.h"

#include <sheredom/subprocess.h>

#include <cstdio>
#include <cstring>
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

std::string join_text_content(const json & content) {
    if (!content.is_array()) {
        return {};
    }

    std::string joined;
    for (const auto & item : content) {
        if (!item.is_object()) {
            continue;
        }
        if (item.value("type", "") != "text") {
            continue;
        }
        const std::string text = item.value("text", "");
        if (text.empty()) {
            continue;
        }
        if (!joined.empty()) {
            joined += "\n";
        }
        joined += text;
    }
    return joined;
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
    }

    json response;
    if (!send_request("initialize", json({
            {"protocolVersion", "2024-11-05"},
            {"capabilities", json::object()},
            {"clientInfo", {
                {"name", "llama-agent"},
                {"version", "0.1"},
            }},
        }), response, error)) {
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
    return write_json_rpc_message(state->in, json({
        {"jsonrpc", "2.0"},
        {"method", method},
        {"params", params},
    }), error);
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
    if (!write_json_rpc_message(state->in, json({
            {"jsonrpc", "2.0"},
            {"id", request_id},
            {"method", method},
            {"params", params},
        }), error)) {
        return false;
    }

    for (;;) {
        json message;
        if (!read_json_rpc_message(state->out, message, error)) {
            return false;
        }

        if (message.contains("id") && message["id"] == request_id) {
            if (message.contains("error")) {
                error = "MCP server returned JSON-RPC error: " + message["error"].dump();
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
        if (state->in != nullptr) {
            std::fclose(state->in);
            state->in = nullptr;
        }
        subprocess_terminate(&state->proc);
        subprocess_destroy(&state->proc);
    }

    state->running = false;
    state->initialized = false;
    state->out = nullptr;
    state->err = nullptr;
}

bool agent_mcp_stdio_client::list_tools(
        const agent_tool_context & context,
        std::vector<mcp_agent_tool_definition> & tools,
        std::string & error) {
    tools.clear();
    if (!ensure_started(error)) {
        return false;
    }

    json response;
    if (!send_request("tools/list", json::object(), response, error)) {
        return false;
    }

    const auto & result = response["result"];
    if (!result.is_object() || !result.contains("tools") || !result["tools"].is_array()) {
        error = "MCP tools/list response did not contain a tools array";
        return false;
    }

    for (const auto & item : result["tools"]) {
        if (!item.is_object()) {
            continue;
        }

        mcp_agent_tool_definition definition;
        definition.provider_id = config.server_name;
        definition.name = item.value("name", "");
        definition.description = item.value("description", "");
        if (item.contains("inputSchema")) {
            definition.input_schema_json = item["inputSchema"].dump();
        }

        if (item.contains("annotations") && item["annotations"].is_object()) {
            definition.read_only = item["annotations"].value("readOnlyHint", definition.read_only);
        }
        if (item.contains("hostPolicy") && item["hostPolicy"].is_object()) {
            const auto & policy = item["hostPolicy"];
            definition.requires_confirmation = policy.value("requiresConfirmation", definition.requires_confirmation);
            definition.uses_network = policy.value("usesNetwork", definition.uses_network);
            definition.writes_memory = policy.value("writesMemory", definition.writes_memory);
            definition.writes_plan = policy.value("writesPlan", definition.writes_plan);
        }

        if (definition.name.empty()) {
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
    if (!send_request("tools/call", json({
            {"name", tool.name},
            {"arguments", arguments},
        }), response, error)) {
        return false;
    }

    const auto & rpc_result = response["result"];
    if (!rpc_result.is_object()) {
        error = "MCP tools/call response did not contain a result object";
        return false;
    }

    result.ok = !rpc_result.value("isError", false);
    result.text_content = join_text_content(rpc_result.value("content", json::array()));
    if (rpc_result.contains("structuredContent")) {
        result.structured_content_json = rpc_result["structuredContent"].dump();
    }
    if (!result.ok) {
        result.failure_code = "mcp.tool_error";
        result.failure_class = common_tool_failure_class::execution;
        result.retryable = false;
        result.safe_summary = result.text_content.empty()
            ? "The MCP server reported a tool error."
            : result.text_content;
        result.raw_diagnostic = result.safe_summary;
    }

    error.clear();
    (void) context;
    return true;
}
