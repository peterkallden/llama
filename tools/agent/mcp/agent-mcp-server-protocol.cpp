#include "agent-mcp-server-protocol.h"
#include "agent-mcp-protocol.h"

bool agent_mcp_read_json_rpc_message(
        FILE * stream,
        agent_mcp_json & message,
        std::string & error) {
    message = agent_mcp_json();
    error.clear();

    std::string line;
    for (;;) {
        const int ch = std::fgetc(stream);
        if (ch == EOF) {
            if (std::ferror(stream)) {
                error = "failed to read MCP stdio message";
            } else if (!line.empty()) {
                error = "incomplete MCP stdio message at end of stream";
            }
            return false;
        }
        line.push_back(static_cast<char>(ch));
        if (ch == '\n' || line.size() > common_mcp_max_stdio_message_bytes) {
            return parse_mcp_stdio_message(line, message, error) && error.empty();
        }
    }
}

bool agent_mcp_write_json_rpc_message(
        FILE * stream,
        const agent_mcp_json & message,
        std::string & error) {
    return write_mcp_stdio_message(stream, message, error);
}

bool agent_mcp_write_malformed_json_rpc_result(
        FILE * stream,
        const agent_mcp_json & id,
        std::string & error) {
    error.clear();
    const std::string body = std::string("{\"jsonrpc\":\"2.0\",\"id\":") + id.dump() + ",\"result\":";
    const std::string line = body + '\n';
    if (std::fwrite(line.data(), 1, line.size(), stream) != line.size()) {
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
