#include "agent-mcp-stdio-server.h"

#include <cstdio>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

bool register_fake_tools(
        agent_mcp_server_tool_registry & registry,
        std::string & error) {
    if (!registry.register_tool({
            "search_issues",
            "Search GitHub issues.",
            R"({"type":"object","required":["query"]})",
            true,
            false,
            true,
            false,
            false,
            [](const agent_mcp_json &, agent_mcp_server_tool_result & result, std::string &) {
                result.ok = true;
                result.content = agent_mcp_json::array({
                    {{"type", "text"}, {"text", "stub issue"}},
                    {
                        {"type", "resource_link"},
                        {"uri", "mcp-resource://github/search_issues/stub-1"},
                        {"name", "search-results.json"},
                        {"description", "Full GitHub issue search result set"},
                        {"mimeType", "application/json"},
                        {"sizeBytes", 128},
                    },
                });
                result.structured_content = {
                    {"items", agent_mcp_json::array({
                        {{"title", "stub issue"}},
                    })},
                };
                result.safe_summary = "Returned a stub issue result.";
                return true;
            },
        }, error)) {
        return false;
    }

    if (!registry.register_tool({
            "search_recent_failures",
            "Simulate a retryable upstream failure.",
            R"({"type":"object","required":["query"]})",
            true,
            false,
            true,
            false,
            false,
            [](const agent_mcp_json &, agent_mcp_server_tool_result & result, std::string &) {
                result.ok = false;
                result.failure_code = "github.rate_limited";
                result.failure_class = "network";
                result.retryable = true;
                result.safe_summary = "The upstream GitHub search provider is temporarily rate limited.";
                result.raw_diagnostic = "upstream search provider is rate limited";
                result.content = agent_mcp_json::array({
                    {{"type", "text"}, {"text", "upstream search provider is rate limited"}},
                });
                return true;
            },
        }, error)) {
        return false;
    }

    if (!registry.register_tool({
            "create_issue",
            "Create a GitHub issue.",
            R"({"type":"object","required":["title"]})",
            false,
            true,
            true,
            false,
            false,
            [](const agent_mcp_json & arguments, agent_mcp_server_tool_result & result, std::string &) {
                result.ok = true;
                result.content = agent_mcp_json::array({
                    {{"type", "text"}, {"text", "created issue #321: " + arguments.value("title", std::string())}},
                });
                return true;
            },
        }, error)) {
        return false;
    }

    error.clear();
    return true;
}

} // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    std::string mode;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        }
    }

    if (mode == "crash-before-initialize") {
        std::fprintf(stderr, "fake-mcp: crash-before-initialize\n");
        return 7;
    }

    std::string error;
    agent_mcp_server_tool_registry registry;
    if (!register_fake_tools(registry, error)) {
        std::fprintf(stderr, "fake-mcp: failed to register tools: %s\n", error.c_str());
        return 1;
    }

    agent_mcp_stdio_server server(
        std::move(registry),
        {
            "fake-mcp",
            "0.1",
            "2024-11-05",
            mode == "bad-tools-list",
            mode == "hang-tools-list",
            mode == "exit-after-initialize",
            [](agent_mcp_json & result, std::string & error) {
                result = {
                    {"resources", agent_mcp_json::array({
                        {
                            {"uri", "mcp-resource://github/search_issues/stub-1"},
                            {"name", "search-results.json"},
                            {"description", "Full GitHub issue search result set"},
                            {"mimeType", "application/json"},
                            {"sizeBytes", 128},
                        },
                    })},
                };
                error.clear();
                return true;
            },
            [](const agent_mcp_json & params, agent_mcp_json & result, std::string & error) {
                const auto uri = params.value("uri", "");
                if (uri != "mcp-resource://github/search_issues/stub-1") {
                    error = "resource was not found";
                    return false;
                }
                result = {
                    {"contents", agent_mcp_json::array({
                        {
                            {"uri", uri},
                            {"name", "search-results.json"},
                            {"description", "Full GitHub issue search result set"},
                            {"mimeType", "application/json"},
                            {"sizeBytes", 128},
                            {"text", R"({"items":[{"title":"stub issue"}]})"},
                        },
                    })},
                };
                error.clear();
                return true;
            },
        });
    return server.run(stdin, stdout, stderr);
}
