#include "agent-tool-provider.h"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

namespace {

std::filesystem::path get_server_path(const char * argv0) {
    std::filesystem::path argv_path = argv0 != nullptr ? std::filesystem::path(argv0) : std::filesystem::path();
    if (argv_path.has_parent_path()) {
        argv_path = std::filesystem::absolute(argv_path);
    } else {
        argv_path = std::filesystem::current_path() / argv_path;
    }
#ifdef _WIN32
    return argv_path.parent_path() / "llama-agent-mcp-stdio-server.exe";
#else
    return argv_path.parent_path() / "llama-agent-mcp-stdio-server";
#endif
}

bool has_tool(const std::vector<common_chat_tool> & tools, const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    const auto server_path = get_server_path(argc > 0 ? argv[0] : nullptr);
    if (!std::filesystem::exists(server_path)) {
        std::fprintf(stderr, "MCP stdio server not found: %s\n", server_path.string().c_str());
        return 1;
    }

    std::string error;
    agent_mcp_stdio_client client({
        "local",
        {server_path.string()},
        {},
    });
    mcp_agent_tool_provider provider("local", client);

    agent_tool_context context;
    context.request_id = "mcp-server-smoke";
    context.turn_id = "turn-1";

    std::unique_ptr<agent_tool_view> view = provider.resolve_tools(context, error);
    if (!view) {
        std::fprintf(stderr, "failed to resolve MCP stdio server tool view: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(view->chat_tools(), "local_echo") ||
            !has_tool(view->chat_tools(), "local_time_now") ||
            !has_tool(view->chat_tools(), "local_calculator")) {
        std::fprintf(stderr, "expected MCP stdio server tools were not exposed\n");
        return 1;
    }

    const auto echo_result = view->call({
        "call-1",
        "local_echo",
        R"({"text":"hello from smoke"})",
    }, error);
    if (!echo_result.ok || echo_result.content_json.find("hello from smoke") == std::string::npos) {
        std::fprintf(stderr, "echo tool did not return the expected result: %s\n", echo_result.content_json.c_str());
        return 1;
    }

    const auto calculator_result = view->call({
        "call-2",
        "local_calculator",
        R"({"expression":"17 * 23"})",
    }, error);
    if (!calculator_result.ok || calculator_result.content_json.find("391") == std::string::npos) {
        std::fprintf(stderr, "calculator tool did not return the expected result: %s\n", calculator_result.content_json.c_str());
        return 1;
    }

    const auto invalid_result = view->call({
        "call-3",
        "local_calculator",
        R"({"expression":"1 / 0"})",
    }, error);
    if (invalid_result.ok ||
            invalid_result.failure_class != common_tool_failure_class::validation ||
            invalid_result.failure_code != "tool.invalid_arguments") {
        std::fprintf(stderr, "invalid calculator arguments were not rejected correctly\n");
        return 1;
    }

    std::printf("mcp_server_tools=%zu\n", view->chat_tools().size());
    std::printf("mcp_server_echo=%s\n", echo_result.content_json.c_str());
    std::printf("mcp_server_calculator=%s\n", calculator_result.content_json.c_str());
    std::printf("mcp_server_invalid_code=%s\n", invalid_result.failure_code.c_str());
    return 0;
}
