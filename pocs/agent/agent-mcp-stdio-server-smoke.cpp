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

    agent_mcp_stdio_client minimal_client({
        "local",
        {server_path.string(), "--tool-profile", "minimal"},
        {},
    });
    mcp_agent_tool_provider minimal_provider("local", minimal_client);

    agent_tool_context minimal_context;
    minimal_context.request_id = "mcp-server-smoke";
    minimal_context.turn_id = "turn-1";

    std::unique_ptr<agent_tool_view> minimal_view = minimal_provider.resolve_tools(minimal_context, error);
    if (!minimal_view) {
        std::fprintf(stderr, "failed to resolve minimal MCP stdio server tool view: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(minimal_view->chat_tools(), "local_time_now") ||
            !has_tool(minimal_view->chat_tools(), "local_calculator") ||
            has_tool(minimal_view->chat_tools(), "local_echo")) {
        std::fprintf(stderr, "minimal MCP profile did not expose the expected native tools\n");
        return 1;
    }

    const auto calculator_result = minimal_view->call({
        "call-1",
        "local_calculator",
        R"({"expression":"17 * 23"})",
    }, error);
    if (!calculator_result.ok || calculator_result.content_json.find("391") == std::string::npos) {
        std::fprintf(stderr, "calculator tool did not return the expected result: %s\n", calculator_result.content_json.c_str());
        return 1;
    }

    const auto invalid_result = minimal_view->call({
        "call-2",
        "local_calculator",
        R"({"expression":"1 / 0"})",
    }, error);
    if (invalid_result.ok ||
            invalid_result.failure_class != common_tool_failure_class::validation ||
            invalid_result.failure_code != "tool.calculator.invalid_expression") {
        std::fprintf(stderr, "invalid calculator arguments were not rejected correctly\n");
        return 1;
    }

    const std::string repository_root = std::filesystem::weakly_canonical(std::filesystem::current_path()).string();
    agent_mcp_stdio_client research_client({
        "local",
        {
            server_path.string(),
            "--tool-profile", "research",
            "--repository-root", repository_root,
        },
        {},
    });
    mcp_agent_tool_provider research_provider("local", research_client);

    agent_tool_context research_context;
    research_context.request_id = "mcp-server-smoke";
    research_context.turn_id = "turn-2";
    research_context.allow_network = true;

    std::unique_ptr<agent_tool_view> research_view = research_provider.resolve_tools(research_context, error);
    if (!research_view) {
        std::fprintf(stderr, "failed to resolve research MCP stdio server tool view: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(research_view->chat_tools(), "local_repository_list")) {
        std::fprintf(stderr, "research MCP profile did not expose repository_list\n");
        return 1;
    }

    const auto repository_result = research_view->call({
        "call-3",
        "local_repository_list",
        R"({"path":"pocs/agent","depth":0})",
    }, error);
    if (!repository_result.ok || repository_result.content_json.find("agent-mcp-stdio-server-main.cpp") == std::string::npos) {
        std::fprintf(stderr, "repository_list did not return the expected repository payload: %s\n", repository_result.content_json.c_str());
        return 1;
    }

    std::printf("mcp_server_minimal_tools=%zu\n", minimal_view->chat_tools().size());
    std::printf("mcp_server_calculator=%s\n", calculator_result.content_json.c_str());
    std::printf("mcp_server_research_tools=%zu\n", research_view->chat_tools().size());
    std::printf("mcp_server_repository_list=%s\n", repository_result.content_json.c_str());
    std::printf("mcp_server_invalid_code=%s\n", invalid_result.failure_code.c_str());
    return 0;
}
