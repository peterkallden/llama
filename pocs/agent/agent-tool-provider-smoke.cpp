#include "agent-tool-provider.h"
#include "agent-resource-store.h"

#include "agent/tool-catalog.h"
#include "memory/memory-in-memory.h"

#include <cstdio>
#include <memory>
#include <string>

namespace {

agent_in_memory_resource_store g_resource_store;

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
        [](const agent_tool_context & context, common_native_tool_bindings & bindings, std::string &) {
            bindings.resource_store = &g_resource_store;
            bindings.resource_namespace_id = context.scope.namespace_id;
            bindings.resource_session_id = context.scope.session_id;
            bindings.resource_project_id = context.scope.project_id;
            bindings.resource_turn_id = context.scope.turn_id;
            const std::string namespace_id = bindings.resource_namespace_id;
            const std::string session_id = bindings.resource_session_id;
            const std::string project_id = bindings.resource_project_id;
            const std::string turn_id = bindings.resource_turn_id;
            agent_resource_store * resource_store = bindings.resource_store;
            bindings.web_search = [resource_store, namespace_id, session_id, project_id, turn_id](const std::string &) {
                agent_resource_descriptor descriptor;
                std::string error;
                if (!resource_store->put_text({
                        "web-search-results.json",
                        "Full web search result set for the current turn.",
                        "application/json",
                        R"({"results":[{"title":"stub issue","url":"https://example.com/stub"}],"provider":"stub"})",
                        common_runtime_resource_scope::turn,
                        namespace_id,
                        session_id,
                        project_id,
                        turn_id,
                        "",
                        "native",
                        "web_search",
                        0,
                        0,
                        {
                            "Preserve the full bounded web search candidate set outside the inline model context.",
                            "Stubbed search candidates for resident inference.",
                            "Use this resource when a later step needs the complete candidate list.",
                            "Provider smoke uses stubbed results.",
                            {"resident inference", "llama.cpp"},
                            {},
                        },
                    }, descriptor, error)) {
                    return common_tool_execution_result::failure(
                        "tool.web_search.resource_store_failed",
                        common_tool_failure_class::execution,
                        false,
                        "Provider smoke failed to write a resource.",
                        error);
                }

                common_runtime_resource_ref resource = descriptor;
                return common_tool_execution_result::success(
                    R"({"results":[{"title":"stub issue"}],"provider":"stub"})",
                    "Web search returned one stub candidate; the full result set was stored as a turn resource.",
                    {resource});
            };
            return true;
        });

    agent_tool_context research_context;
    research_context.request_id = "provider-smoke";
    research_context.turn_id = "turn-2";
    research_context.profile_id = "research";
    research_context.allow_network = true;
    research_context.scope.namespace_id = "provider-smoke";
    research_context.scope.session_id = "session-1";
    research_context.scope.project_id = "project-1";
    research_context.scope.turn_id = "turn-2";

    std::unique_ptr<agent_tool_view> research_view = research_provider.resolve_tools(research_context, error);
    if (!research_view) {
        std::fprintf(stderr, "research provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(research_view->chat_tools(), "web_search")) {
        std::fprintf(stderr, "web_search was not exposed in the research tool view\n");
        return 1;
    }

    const auto search_result = research_view->call({
        "call-2b",
        "web_search",
        R"({"query":"resident inference architecture in llama.cpp","limit":5})",
    }, error);
    if (!search_result.ok) {
        std::fprintf(stderr, "web_search call failed: %s\n", search_result.content_json.c_str());
        return 1;
    }
    if (search_result.resource_refs.empty()) {
        std::fprintf(stderr, "web_search did not materialize a resource reference for the full result set\n");
        return 1;
    }
    if (search_result.resource_refs[0].metadata.content_summary.empty()) {
        std::fprintf(stderr, "web_search resource metadata content summary was empty\n");
        return 1;
    }
    if (search_result.content_json.find("\"resources\"") == std::string::npos) {
        std::fprintf(stderr, "web_search result payload did not include rendered resources: %s\n", search_result.content_json.c_str());
        return 1;
    }

    const auto resource_read_result = research_view->call({
        "call-2c",
        "resource_read",
        std::string(R"({"uri":")") + search_result.resource_refs[0].uri + R"(","max_bytes":4096})",
    }, error);
    if (!resource_read_result.ok || resource_read_result.content_json.find("stub issue") == std::string::npos) {
        std::fprintf(stderr, "resource_read did not return the expected stored payload: %s\n", resource_read_result.content_json.c_str());
        return 1;
    }

    common_memory_in_memory_store memory_store;
    if (!memory_store.open("", error)) {
        std::fprintf(stderr, "memory store open failed: %s\n", error.c_str());
        return 1;
    }

    native_agent_tool_provider memory_provider(
        catalog,
        [&memory_store](const agent_tool_context &, common_native_tool_bindings & bindings, std::string &) {
            bindings.memory_store = &memory_store;
            bindings.memory_query.namespace_id = "provider-smoke";
            bindings.memory_query.session_id = "session-1";
            bindings.memory_query.project_id = "project-1";
            bindings.memory_query.scope = common_memory_scope::session;
            return true;
        });

    agent_tool_context memory_context;
    memory_context.request_id = "provider-smoke";
    memory_context.turn_id = "turn-3";
    memory_context.profile_id = "memory";
    memory_context.allow_policy_gated_writes = true;
    memory_context.allow_memory_proposals = true;

    std::unique_ptr<agent_tool_view> memory_view = memory_provider.resolve_tools(memory_context, error);
    if (!memory_view) {
        std::fprintf(stderr, "memory provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(memory_view->chat_tools(), "memory_remember")) {
        std::fprintf(stderr, "memory_remember was not exposed through memory tool view\n");
        return 1;
    }

    const auto memory_result = memory_view->call({
        "call-3",
        "memory_remember",
        R"({"kind":"fact","content":"The provider smoke stores native memory proposals."})",
    }, error);
    if (!memory_result.ok || memory_result.content_json.find("\"decision\":\"accept\"") == std::string::npos) {
        std::fprintf(stderr, "memory_remember call did not succeed: %s\n", memory_result.content_json.c_str());
        return 1;
    }

    std::printf("provider_tools=%zu\n", minimal_view->chat_tools().size());
    std::printf("calculator_result=%s\n", first_result.content_json.c_str());
    std::printf("network_tool_exposed=%s\n", has_tool(research_view->chat_tools(), "web_search") ? "yes" : "no");
    std::printf("web_search_resource_uri=%s\n", search_result.resource_refs.empty() ? "" : search_result.resource_refs[0].uri.c_str());
    std::printf("memory_remember_result=%s\n", memory_result.content_json.c_str());
    return 0;
}
