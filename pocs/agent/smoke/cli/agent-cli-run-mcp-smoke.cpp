#include "tools/agent/cli/agent-cli-config.h"
#include "tools/agent/cli/agent-cli-host-adapter.h"
#include "tools/agent/cli/agent-cli-run-adapter.h"

#include "memory/memory-in-memory.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

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

    std::vector<std::string> arg_storage = {
        "llama-agent",
        "run",
        "--model", "fake.gguf",
        "--prompt", "What tools are available?",
        "--thinking-mode", "deliberate",
        "--max-reflection-rounds", "3",
        "--max-plan-revisions", "2",
        "--max-research-iterations", "1",
        "--tool-profile", "minimal",
        "--mcp-tool-command", server_path.string(),
        "--mcp-tool-server-name", "github",
        "--mcp-tool-prefix", "github",
        "--memory-namespace", "local",
        "--memory-session", "run-smoke",
    };
    std::vector<char *> argv_ptrs;
    argv_ptrs.reserve(arg_storage.size());
    for (auto & arg : arg_storage) {
        argv_ptrs.push_back(arg.data());
    }

    args options;
    if (!parse_agent_run_args(static_cast<int>(argv_ptrs.size()), argv_ptrs.data(), options)) {
        std::fprintf(stderr, "CLI MCP run smoke failed to parse synthetic run arguments\n");
        return 1;
    }

    std::string error;
    if (!prepare_agent_cli_args(options, error)) {
        std::fprintf(stderr, "CLI MCP run smoke failed to prepare args: %s\n", error.c_str());
        return 1;
    }
    if (options.thinking_mode != "deliberate" ||
            options.max_reflection_rounds != 3 ||
            options.max_plan_revisions != 2 ||
            options.max_research_iterations != 1) {
        std::fprintf(stderr, "CLI MCP run smoke did not preserve thinking policy arguments\n");
        return 1;
    }

    common_memory_in_memory_store store;
    common_memory_query query;
    query.text = options.prompt;
    query.scope = common_memory_scope::session;

    common_agent_cli_tool_selection selection;
    if (!resolve_agent_cli_tool_selection(
            store,
            nullptr,
            nullptr,
            nullptr,
            options,
            query,
            false,
            selection,
            error)) {
        std::fprintf(stderr, "CLI MCP run smoke failed to resolve tool selection: %s\n", error.c_str());
        return 1;
    }

    if (!has_tool(selection.tooling.tools, "calculator") ||
            !has_tool(selection.tooling.tools, "github_search_issues")) {
        std::fprintf(stderr, "CLI MCP run smoke did not expose the expected composite tool set\n");
        return 1;
    }

    const auto mcp_result = selection.tool_view->call({
        "call-1",
        "github_search_issues",
        R"({"query":"resident inference"})",
    }, error);
    if (!mcp_result.ok || mcp_result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "CLI MCP run smoke MCP tool call failed: %s\n", mcp_result.content_json.c_str());
        return 1;
    }

    std::printf("cli_run_mcp_tools=%zu\n", selection.tooling.tools.size());
    std::printf("cli_run_mcp_result=%s\n", mcp_result.content_json.c_str());
    return 0;
}
