#include "tools/agent/tooling/agent-tool-provider.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

bool has_tool(const std::vector<common_chat_tool> & tools, const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
}

class fake_mcp_tool_client final : public agent_mcp_tool_client {
public:
    std::string last_arguments;

    bool list_tools(
            const agent_tool_context &,
            std::vector<mcp_agent_tool_definition> & tools,
            std::string & error) override {
        tools = {
            {
                "github",
                "create_issue",
                "Create a GitHub issue.",
                R"({"type":"object","additionalProperties":false,"required":["title"],"properties":{"title":{"type":"string","minLength":1}}})",
                false,
                true,
                true,
                false,
                false,
            },
            {
                "github",
                "search_issues",
                "Search GitHub issues.",
                R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string","minLength":1}}})",
                true,
                false,
                true,
                false,
                false,
            },
        };
        error.clear();
        return true;
    }

    bool call_tool(
            const agent_tool_context &,
            const mcp_agent_tool_definition & tool,
            const std::string & arguments_json,
            mcp_agent_tool_call_result & result,
            std::string & error) override {
        last_arguments = arguments_json;
        if (tool.name == "search_issues") {
            result.ok = true;
            result.structured_content_json = R"({"items":[{"title":"stub issue"}]})";
            error.clear();
            return true;
        }

        if (tool.name == "create_issue") {
            result.ok = true;
            result.text_content = "created issue #123";
            error.clear();
            return true;
        }

        error = "unexpected MCP tool call: " + tool.name + " with args " + arguments_json;
        return false;
    }
};

} // namespace

int main() {
    std::string error;
    fake_mcp_tool_client client;
    mcp_agent_tool_provider provider("github", client);

    agent_tool_context read_only_context;
    read_only_context.request_id = "mcp-provider-smoke";
    read_only_context.turn_id = "turn-1";
    read_only_context.allow_network = true;

    std::unique_ptr<agent_tool_view> read_only_view = provider.resolve_tools(read_only_context, error);
    if (!read_only_view) {
        std::fprintf(stderr, "MCP provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(read_only_view->chat_tools(), "github_search_issues")) {
        std::fprintf(stderr, "search_issues was not exposed through the MCP tool view\n");
        return 1;
    }
    if (has_tool(read_only_view->chat_tools(), "github_create_issue")) {
        std::fprintf(stderr, "create_issue was exposed despite writes being disabled\n");
        return 1;
    }

    const auto search_result = read_only_view->call({
        "call-1",
        "github_search_issues",
        R"({ "query": "resident inference" })",
    }, error);
    if (!search_result.ok || search_result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "search_issues did not return the expected result: %s\n", search_result.content_json.c_str());
        return 1;
    }
    if (client.last_arguments != R"({"query":"resident inference"})") {
        std::fprintf(stderr, "MCP provider did not execute normalized arguments: %s\n", client.last_arguments.c_str());
        return 1;
    }

    agent_tool_context limited_context = read_only_context;
    limited_context.turn_id = "turn-limited-result";
    limited_context.default_max_result_bytes = 8;
    std::unique_ptr<agent_tool_view> limited_view = provider.resolve_tools(limited_context, error);
    if (!limited_view) {
        std::fprintf(stderr, "limited MCP provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    const auto limited_result = limited_view->call({
        "call-limited",
        "github_search_issues",
        R"({"query":"x"})",
    }, error);
    if (limited_result.ok || limited_result.failure_code != "tool_result_too_large") {
        std::fprintf(stderr, "MCP result-size limit was not enforced\n");
        return 1;
    }

    agent_tool_context filtered_context;
    filtered_context.request_id = "mcp-provider-smoke";
    filtered_context.turn_id = "turn-filtered";
    filtered_context.allow_network = true;
    filtered_context.allowed_exposed_tool_names = {"github_search_issues"};

    std::unique_ptr<agent_tool_view> filtered_view = provider.resolve_tools(filtered_context, error);
    if (!filtered_view) {
        std::fprintf(stderr, "filtered MCP provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(filtered_view->chat_tools(), "github_search_issues")) {
        std::fprintf(stderr, "filtered MCP provider dropped the allowed exposed tool\n");
        return 1;
    }
    if (has_tool(filtered_view->chat_tools(), "github_create_issue")) {
        std::fprintf(stderr, "filtered MCP provider did not enforce exposed-name filtering\n");
        return 1;
    }

    agent_tool_context write_context;
    write_context.request_id = "mcp-provider-smoke";
    write_context.turn_id = "turn-2";
    write_context.allow_network = true;
    write_context.allow_policy_gated_writes = true;

    std::unique_ptr<agent_tool_view> write_view = provider.resolve_tools(write_context, error);
    if (!write_view) {
        std::fprintf(stderr, "MCP write-capable provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(write_view->chat_tools(), "github_create_issue")) {
        std::fprintf(stderr, "create_issue was not exposed when policy-gated writes were enabled\n");
        return 1;
    }

    const auto create_result = write_view->call({
        "call-2",
        "github_create_issue",
        R"({"title":"Add MCP provider smoke"})",
    }, error);
    if (!create_result.ok || create_result.content_json.find("created issue #123") == std::string::npos) {
        std::fprintf(stderr, "create_issue did not return the expected result: %s\n", create_result.content_json.c_str());
        return 1;
    }

    const auto invalid_result = write_view->call({
        "call-3",
        "github_search_issues",
        R"("not-an-object")",
    }, error);
    if (invalid_result.ok || invalid_result.failure_class != common_tool_failure_class::validation) {
        std::fprintf(stderr, "invalid MCP arguments were not rejected correctly\n");
        return 1;
    }

    const auto missing_required_result = write_view->call({
        "call-4",
        "github_search_issues",
        R"({})",
    }, error);
    if (missing_required_result.ok || missing_required_result.failure_class != common_tool_failure_class::validation) {
        std::fprintf(stderr, "MCP required-field validation was not enforced\n");
        return 1;
    }

    const auto wrong_type_result = write_view->call({
        "call-5",
        "github_search_issues",
        R"({"query":42})",
    }, error);
    if (wrong_type_result.ok || wrong_type_result.failure_class != common_tool_failure_class::validation) {
        std::fprintf(stderr, "MCP property-type validation was not enforced\n");
        return 1;
    }

    std::printf("mcp_exposed_tools=%zu\n", write_view->chat_tools().size());
    std::printf("mcp_search_result=%s\n", search_result.content_json.c_str());
    std::printf("mcp_create_result=%s\n", create_result.content_json.c_str());
    return 0;
}
