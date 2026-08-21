#include "tools/agent/tooling/agent-tool-provider.h"

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

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

} // namespace

int main(int argc, char ** argv) {
    const auto server_path = get_fake_server_path(argc > 0 ? argv[0] : nullptr);
    if (!std::filesystem::exists(server_path)) {
        std::fprintf(stderr, "fake MCP stdio server not found: %s\n", server_path.string().c_str());
        return 1;
    }

    std::string error;
    size_t write_tool_count = 0;
    std::string search_content_json;
    std::string search_resource_uri;
    std::string resource_text_content;
    std::string failure_code;
    std::string create_content_json;

    agent_tool_context read_context;
    read_context.request_id = "mcp-stdio-smoke";
    read_context.turn_id = "turn-1";
    read_context.allow_network = true;

    {
        agent_mcp_stdio_client client({
            "github",
            {server_path.string()},
            {},
        });
        mcp_agent_tool_provider provider("github", client);

        std::unique_ptr<agent_tool_view> read_view = provider.resolve_tools(read_context, error);
        if (!read_view) {
            std::fprintf(stderr, "failed to resolve MCP stdio tool view: %s\n", error.c_str());
            return 1;
        }
        if (!has_tool(read_view->chat_tools(), "github_search_issues")) {
            std::fprintf(stderr, "github_search_issues was not exposed through the MCP stdio tool view\n");
            return 1;
        }
        if (!has_tool(read_view->chat_tools(), "github_search_recent_failures")) {
            std::fprintf(stderr, "github_search_recent_failures was not exposed through the MCP stdio tool view\n");
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
        if (search_result.resource_refs.empty() ||
                search_result.resource_refs[0].uri != "mcp-resource://github/search_issues/stub-1" ||
                search_result.content_json.find("\"resources\"") == std::string::npos) {
            std::fprintf(stderr, "github_search_issues did not preserve MCP resource links: %s\n", search_result.content_json.c_str());
            return 1;
        }

        std::vector<mcp_agent_resource_definition> resources;
        if (!client.list_resources(resources, error) ||
                resources.empty() ||
                resources[0].resource.uri != "mcp-resource://github/search_issues/stub-1") {
            std::fprintf(stderr, "resources/list did not return the expected MCP resources: %s\n", error.c_str());
            return 1;
        }

        mcp_agent_resource_read_result resource_read;
        if (!client.read_resource("mcp-resource://github/search_issues/stub-1", resource_read, error) ||
                resource_read.resource.mime_type != "application/json" ||
                resource_read.text_content.find("stub issue") == std::string::npos) {
            std::fprintf(stderr, "resources/read did not return the expected MCP resource content: %s\n", error.c_str());
            return 1;
        }

        const auto failure_result = read_view->call({
            "call-err",
            "github_search_recent_failures",
            R"({"query":"timeout"})",
        }, error);
        if (failure_result.ok ||
                failure_result.failure_code != "github.rate_limited" ||
                failure_result.failure_class != common_tool_failure_class::network ||
                !failure_result.retryable) {
            std::fprintf(stderr, "github_search_recent_failures did not return the expected structured MCP failure\n");
            return 1;
        }

        agent_tool_context cancelled_context = read_context;
        cancelled_context.execution_control = make_common_agent_runtime_execution_control({});
        std::unique_ptr<agent_tool_view> cancelled_view = provider.resolve_tools(cancelled_context, error);
        if (!cancelled_view) {
            std::fprintf(stderr, "failed to resolve cancelled MCP tool view: %s\n", error.c_str());
            return 1;
        }
        cancelled_context.execution_control.cancellation->request_cancel("mcp smoke cancelled");
        const auto cancelled_result = cancelled_view->call({
            "call-cancelled",
            "github_search_issues",
            R"({"query":"resident inference"})",
        }, error);
        if (cancelled_result.ok ||
                cancelled_result.failure_code != "tool_call_cancelled") {
            std::fprintf(stderr, "cancelled MCP tool call did not return the expected failure\n");
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

        write_tool_count = write_view->chat_tools().size();
        search_content_json = search_result.content_json;
        search_resource_uri = search_result.resource_refs.empty() ? "" : search_result.resource_refs[0].uri;
        resource_text_content = resource_read.text_content;
        failure_code = failure_result.failure_code;
        create_content_json = create_result.content_json;
    }

    agent_mcp_stdio_client broken_client({
        "github",
        {server_path.string(), "--mode", "bad-tools-list"},
        {},
    });
    mcp_agent_tool_provider broken_provider("github", broken_client);
    error.clear();
    std::unique_ptr<agent_tool_view> broken_view = broken_provider.resolve_tools(read_context, error);
    if (broken_view != nullptr) {
        std::fprintf(stderr, "bad-tools-list MCP server unexpectedly resolved successfully\n");
        return 1;
    }
    // The client may report the protocol error before the child has been
    // joined. The protocol error and captured stderr are the stable contract;
    // exit-code context is best-effort transport metadata.
    if (!contains(error, "invalid JSON-RPC payload") ||
            !contains(error, "fake-mcp: emitting malformed tools/list payload")) {
        std::fprintf(stderr, "broken MCP diagnostic was missing expected context: %s\n", error.c_str());
        return 1;
    }

    agent_mcp_stdio_client hanging_client({
        "github",
        {server_path.string(), "--mode", "hang-tools-list"},
        {},
        100,
        100,
    });
    mcp_agent_tool_provider hanging_provider("github", hanging_client);
    agent_tool_context timeout_context = read_context;
    timeout_context.execution_control = make_common_agent_runtime_execution_control({
        0, 0, 0, 0, 100, 100,
    });
    error.clear();
    std::unique_ptr<agent_tool_view> hanging_view = hanging_provider.resolve_tools(timeout_context, error);
    if (hanging_view != nullptr || !contains(error, "MCP request timeout")) {
        std::fprintf(stderr, "hanging MCP request did not terminate on hard timeout: %s\n", error.c_str());
        return 1;
    }

    std::printf("stdio_mcp_tools=%zu\n", write_tool_count);
    std::printf("stdio_mcp_search_result=%s\n", search_content_json.c_str());
    std::printf("stdio_mcp_resource_uri=%s\n", search_resource_uri.c_str());
    std::printf("stdio_mcp_resource_text=%s\n", resource_text_content.c_str());
    std::printf("stdio_mcp_failure_code=%s\n", failure_code.c_str());
    std::printf("stdio_mcp_create_result=%s\n", create_content_json.c_str());
    std::printf("stdio_mcp_error_context=%s\n", error.c_str());
    return 0;
}
