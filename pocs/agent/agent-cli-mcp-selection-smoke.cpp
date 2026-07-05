#include "agent-cli-host-adapter.h"

#include "../common/cli-config.h"

#include "memory/memory-in-memory.h"

#include <cstdio>
#include <filesystem>
#include <string>

namespace {

std::filesystem::path get_fake_server_path(const char * argv0) {
    std::filesystem::path argv_path = argv0 != nullptr ? std::filesystem::path(argv0) : std::filesystem::path();
    if (argv_path.has_parent_path()) {
        argv_path = std::filesystem::absolute(argv_path);
    } else {
        argv_path = std::filesystem::current_path() / argv_path;
    }
#ifdef _WIN32
    return argv_path.parent_path() / "llama-agent-mcp-stdio-fake-server.exe";
#else
    return argv_path.parent_path() / "llama-agent-mcp-stdio-fake-server";
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
    const auto server_path = get_fake_server_path(argc > 0 ? argv[0] : nullptr);
    if (!std::filesystem::exists(server_path)) {
        std::fprintf(stderr, "fake MCP stdio server not found: %s\n", server_path.string().c_str());
        return 1;
    }

    common_memory_in_memory_store store;
    args options;
    options.command = "run";
    options.tool_profile = "minimal";
    options.tool_profile_explicit = true;
    options.mcp_tool_command = server_path.string();
    options.mcp_tool_server_name = "github";
    options.mcp_tool_prefix = "github";
    options.max_tool_rounds = 2;
    options.memory_namespace = "local";
    options.memory_session = "smoke";

    common_memory_query query;
    query.text = "resident inference";
    query.scope = common_memory_scope::session;

    common_agent_cli_tool_selection selection;
    std::string error;
    if (!resolve_agent_cli_tool_selection(
            store,
            nullptr,
            nullptr,
            options,
            query,
            false,
            selection,
            error)) {
        std::fprintf(stderr, "CLI MCP tool selection failed: %s\n", error.c_str());
        return 1;
    }

    if (!selection.mcp_client) {
        std::fprintf(stderr, "CLI MCP selection did not retain the MCP client lifetime\n");
        return 1;
    }
    if (!selection.tool_view) {
        std::fprintf(stderr, "CLI MCP selection did not return a tool view\n");
        return 1;
    }
    if (!has_tool(selection.tooling.tools, "github_search_issues")) {
        std::fprintf(stderr, "CLI MCP selection did not expose github_search_issues\n");
        return 1;
    }
    if (!has_tool(selection.tooling.tools, "calculator")) {
        std::fprintf(stderr, "CLI MCP selection did not expose native calculator from the minimal tool profile\n");
        return 1;
    }
    if (has_tool(selection.tooling.tools, "github_create_issue")) {
        std::fprintf(stderr, "CLI MCP selection exposed github_create_issue despite writes being disabled\n");
        return 1;
    }

    const auto native_result = selection.tool_view->call({
        "call-native",
        "calculator",
        R"({"expression":"2 + 3"})",
    }, error);
    if (!native_result.ok || native_result.content_json.find("5") == std::string::npos) {
        std::fprintf(stderr, "CLI MCP selection native tool call did not return the expected result: %s\n", native_result.content_json.c_str());
        return 1;
    }

    const auto result = selection.tool_view->call({
        "call-1",
        "github_search_issues",
        R"({"query":"resident inference"})",
    }, error);
    if (!result.ok || result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "CLI MCP selection tool call did not return the expected result: %s\n", result.content_json.c_str());
        return 1;
    }

    std::printf("cli_mcp_tools=%zu\n", selection.tooling.tools.size());
    std::printf("cli_native_result=%s\n", native_result.content_json.c_str());
    std::printf("cli_mcp_search_result=%s\n", result.content_json.c_str());
    return 0;
}
