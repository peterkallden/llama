#include "agent-tool-provider.h"

#include "agent/tool-catalog.h"

#include <cstdio>
#include <memory>
#include <string>

namespace {

bool has_tool(const std::vector<common_chat_tool> & tools, const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    std::string error;
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("research", bootstrap, error)) {
        std::fprintf(stderr, "tool bootstrap failed: %s\n", error.c_str());
        return 1;
    }

    native_agent_tool_provider minimal_provider(
        catalog,
        [](const agent_tool_context &, common_native_tool_bindings &, std::string &) {
            return true;
        });

    agent_tool_context minimal_context;
    minimal_context.request_id = "provider-smoke";
    minimal_context.turn_id = "turn-1";
    minimal_context.profile_id = "minimal";
    minimal_context.max_calls = 1;

    std::unique_ptr<agent_tool_view> minimal_view = minimal_provider.resolve_tools(minimal_context, error);
    if (!minimal_view) {
        std::fprintf(stderr, "minimal provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(minimal_view->chat_tools(), "calculator")) {
        std::fprintf(stderr, "calculator was not exposed through minimal tool view\n");
        return 1;
    }

    const auto first_result = minimal_view->call({
        "call-1",
        "calculator",
        R"({"expression":"17 * 23"})",
    }, error);
    if (!first_result.ok) {
        std::fprintf(stderr, "calculator call failed: %s\n", error.c_str());
        return 1;
    }
    if (first_result.content_json.find("391") == std::string::npos) {
        std::fprintf(stderr, "calculator result payload did not contain expected value: %s\n", first_result.content_json.c_str());
        return 1;
    }

    const auto second_result = minimal_view->call({
        "call-2",
        "calculator",
        R"({"expression":"1 + 1"})",
    }, error);
    if (second_result.ok ||
            second_result.failure_class != common_tool_failure_class::limit ||
            second_result.failure_code != "tool_call_limit_reached") {
        std::fprintf(stderr, "tool call limit was not enforced\n");
        return 1;
    }

    native_agent_tool_provider research_provider(
        catalog,
        [](const agent_tool_context &, common_native_tool_bindings & bindings, std::string &) {
            bindings.web_search = [](const std::string &) {
                return common_tool_execution_result::success(R"({"items":[{"title":"stub"}]})");
            };
            return true;
        });

    agent_tool_context research_context;
    research_context.request_id = "provider-smoke";
    research_context.turn_id = "turn-2";
    research_context.profile_id = "research";
    research_context.allow_network = false;

    std::unique_ptr<agent_tool_view> research_view = research_provider.resolve_tools(research_context, error);
    if (!research_view) {
        std::fprintf(stderr, "research provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (has_tool(research_view->chat_tools(), "web_search")) {
        std::fprintf(stderr, "web_search was exposed despite network access being disabled\n");
        return 1;
    }

    std::printf("provider_tools=%zu\n", minimal_view->chat_tools().size());
    std::printf("calculator_result=%s\n", first_result.content_json.c_str());
    std::printf("network_tool_exposed=%s\n", has_tool(research_view->chat_tools(), "web_search") ? "yes" : "no");
    return 0;
}
