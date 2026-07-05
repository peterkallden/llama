#include "agent-tool-provider.h"

#include <cstdio>
#include <filesystem>
#include <memory>
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

    std::string error;
    agent_mcp_stdio_client client({
        "github",
        {server_path.string()},
        {},
    });
    mcp_agent_tool_provider provider("github", client);

    agent_tool_context read_context;
    read_context.request_id = "mcp-stdio-smoke";
    read_context.turn_id = "turn-1";
    read_context.allow_network = true;

    std::unique_ptr<agent_tool_view> read_view = provider.resolve_tools(read_context, error);
    if (!read_view) {
        std::fprintf(stderr, "failed to resolve MCP stdio tool view: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(read_view->chat_tools(), "github_search_issues")) {
        std::fprintf(stderr, "github_search_issues was not exposed through the MCP stdio tool view\n");
        return 1;
    }
    if (has_tool(read_view->chat_tools(), "github_create_issue")) {
        std::fprintf(stderr, "github_create_issue was exposed despite writes being disabled\n");
        return 1;
    }

    const auto search_result = read_view->call({
        "call-1",
        "github_search_issues",
        R"({"query":"resident inference"})",
    }, error);
    if (!search_result.ok || search_result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "github_search_issues did not return the expected result: %s\n", search_result.content_json.c_str());
        return 1;
    }

    agent_tool_context write_context;
    write_context.request_id = "mcp-stdio-smoke";
    write_context.turn_id = "turn-2";
    write_context.allow_network = true;
    write_context.allow_policy_gated_writes = true;

    std::unique_ptr<agent_tool_view> write_view = provider.resolve_tools(write_context, error);
    if (!write_view) {
        std::fprintf(stderr, "failed to resolve write-capable MCP stdio tool view: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(write_view->chat_tools(), "github_create_issue")) {
        std::fprintf(stderr, "github_create_issue was not exposed when writes were enabled\n");
        return 1;
    }

    const auto create_result = write_view->call({
        "call-2",
        "github_create_issue",
        R"({"title":"Add stdio MCP client smoke"})",
    }, error);
    if (!create_result.ok || create_result.content_json.find("created issue #321") == std::string::npos) {
        std::fprintf(stderr, "github_create_issue did not return the expected result: %s\n", create_result.content_json.c_str());
        return 1;
    }

    std::printf("stdio_mcp_tools=%zu\n", write_view->chat_tools().size());
    std::printf("stdio_mcp_search_result=%s\n", search_result.content_json.c_str());
    std::printf("stdio_mcp_create_result=%s\n", create_result.content_json.c_str());
    return 0;
}
