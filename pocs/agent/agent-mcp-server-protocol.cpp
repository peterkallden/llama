#include "agent-mcp-server-protocol.h"

#include <cstdlib>
#include <cstring>

namespace {

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

} // namespace

bool agent_mcp_read_json_rpc_message(
        FILE * stream,
        agent_mcp_json & message,
        std::string & error) {
    message = agent_mcp_json();
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
        error = "MCP message missing Content-Length header";
        return false;
    }

    std::string body(content_length, '\0');
    if (std::fread(body.data(), 1, content_length, stream) != content_length) {
        error = "failed to read complete MCP message body";
        return false;
    }

    const auto parsed = agent_mcp_json::parse(body, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        error = "MCP message contained invalid JSON-RPC payload";
        return false;
    }

    message = parsed;
    return true;
}

bool agent_mcp_write_json_rpc_message(
        FILE * stream,
        const agent_mcp_json & message,
        std::string & error) {
    error.clear();
    const std::string body = message.dump();
    const std::string framed =
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    if (std::fwrite(framed.data(), 1, framed.size(), stream) != framed.size()) {
        error = "failed to write MCP JSON-RPC message";
        return false;
    }
    if (std::fflush(stream) != 0) {
        error = "failed to flush MCP JSON-RPC message";
        return false;
    }
    return true;
}

bool agent_mcp_write_malformed_json_rpc_result(
        FILE * stream,
        const agent_mcp_json & id,
        std::string & error) {
    error.clear();
    const std::string body = std::string("{\"jsonrpc\":\"2.0\",\"id\":") + id.dump() + ",\"result\":";
    const std::string framed =
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    if (std::fwrite(framed.data(), 1, framed.size(), stream) != framed.size()) {
        error = "failed to write malformed MCP JSON-RPC message";
        return false;
    }
    if (std::fflush(stream) != 0) {
        error = "failed to flush malformed MCP JSON-RPC message";
        return false;
    }
    return true;
}

agent_mcp_json agent_mcp_make_json_rpc_result(
        const agent_mcp_json & id,
        agent_mcp_json result) {
    return {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", std::move(result)},
    };
}

agent_mcp_json agent_mcp_make_json_rpc_error(
        const agent_mcp_json & id,
        int code,
        std::string message) {
    return {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", {
            {"code", code},
            {"message", std::move(message)},
        }},
    };
}
