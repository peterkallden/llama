#include "agent/tool-adapters.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>

int main() {
    std::string error;
    common_memory_in_memory_store memories;
    assert(memories.open("", error));
    common_memory_record memory;
    memory.id = "memory-1";
    memory.kind = common_memory_kind::fact;
    memory.content = "The plan store uses optimistic version checks.";
    memory.scope = common_memory_scope::session;
    memory.session_id = "session-1";
    assert(memories.put(memory, error));

    common_plan_in_memory_store plans;
    assert(plans.open("", error));
    common_plan_state plan;
    plan.id = "plan-1";
    plan.goal = "Verify native adapters";
    assert(plans.create(plan, error));

    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    assert(catalog.bootstrap("memory-read", bootstrap, error));
    common_tool_registry registry;
    common_native_tool_bindings bindings;
    bindings.memory_store = &memories;
    bindings.plan_store = &plans;
    bindings.memory_query.scope = common_memory_scope::session;
    bindings.memory_query.session_id = "session-1";
    std::string active_plan_id = "plan-1";
    bindings.plan_id = &active_plan_id;
    common_tool_adapter_result adapters;
    assert(common_register_native_tool_adapters(catalog, "memory-read", bindings, registry, adapters, error));
    assert(adapters.registered.size() == 5);
    auto result = registry.execute({"calculator", R"({"expression":"(18 + 2) * 3"})"});
    assert(result.ok);
    auto output = result.output;
    assert(output == R"({"value":60.0})" || output == R"({"value":60})");
    result = registry.execute({"memory_search", R"({"query":"optimistic checks"})"});
    assert(result.ok && result.output.find("memory-1") != std::string::npos);
    result = registry.execute({"memory_get", R"({"id":"memory-1"})"});
    assert(result.ok && result.output.find("optimistic version checks") != std::string::npos);
    result = registry.execute({"plan_get", "{}"});
    assert(result.ok && result.output.find("plan-1") != std::string::npos);
    result = registry.execute({"memory_remember", "{}"});
    assert(!result.ok);

    common_tool_registry proposal_registry;
    common_native_tool_bindings proposal_bindings;
    proposal_bindings.memory_remember_proposal = [](const std::string &) { return common_tool_execution_result::success(R"({"decision":"accept"})"); };
    assert(common_register_native_tool_adapters(catalog, "memory", proposal_bindings, proposal_registry, adapters, error));
    assert(proposal_registry.is_policy_gated("memory_remember"));
    result = proposal_registry.execute({"memory_remember", R"({"kind":"fact","content":"verified"})"});
    assert(result.ok && result.output.find("accept") != std::string::npos);

    const auto repository = std::filesystem::temp_directory_path() / "llama-agent-repository-tool-test";
    std::filesystem::create_directories(repository / "src");
    { std::ofstream file(repository / "src" / "sample.txt"); file << "alpha\nneedle in a haystack\n"; }
    const auto git_init = "git -C \"" + repository.string() + "\" init -q";
    assert(std::system(git_init.c_str()) == 0);
    common_tool_catalog research_catalog;
    assert(research_catalog.bootstrap("research", bootstrap, error));
    common_tool_registry repository_registry;
    common_native_tool_bindings repository_bindings;
    repository_bindings.repository_root = repository.string();
    repository_bindings.web_search = [](const std::string & input) {
        if (input.find("\"query\":\"llama\"") == std::string::npos) {
            return common_tool_execution_result::failure("tool.web_search.unexpected_query", common_tool_failure_class::validation, false, "Unexpected search query.", "unexpected search query");
        }
        return common_tool_execution_result::success(R"({"results":[{"title":"llama.cpp","url":"https://example.com/llama","snippet":"native tools","source":"test"}],"provider":"test"})");
    };
    repository_bindings.web_fetch = [](const std::string & input) {
        if (input.find("\"url\":\"https://example.com/llama\"") == std::string::npos) {
            return common_tool_execution_result::failure("tool.web_fetch.unexpected_url", common_tool_failure_class::validation, false, "Unexpected fetch URL.", "unexpected fetch url");
        }
        return common_tool_execution_result::success(R"({"url":"https://example.com/llama","final_url":"https://example.com/llama","status":200,"content_type":"text/html","title":"llama.cpp","text":"native tools","truncated":false})");
    };
    assert(common_register_native_tool_adapters(research_catalog, "research", repository_bindings, repository_registry, adapters, error));
    result = repository_registry.execute({"repository_list", R"({"path":"src","depth":1})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos);
    result = repository_registry.execute({"repository_search", R"({"query":"needle","path":"src"})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos && result.output.find("needle") != std::string::npos);
    result = repository_registry.execute({"repository_read", R"({"path":"src/sample.txt","start_line":2,"end_line":2})"});
    assert(result.ok && result.output.find("needle in a haystack") != std::string::npos);
    result = repository_registry.execute({"web_search", R"({"query":"llama","limit":1})"});
    assert(result.ok && result.output.find("https://example.com/llama") != std::string::npos);
    result = repository_registry.execute({"web_fetch", R"({"url":"https://example.com/llama","max_bytes":4096})"});
    assert(result.ok && result.output.find("\"status\":200") != std::string::npos && result.output.find("native tools") != std::string::npos);
    result = repository_registry.execute({"repository_read", R"({"path":"../outside.txt"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::validation);

    common_tool_catalog developer_catalog;
    assert(developer_catalog.bootstrap("developer-read", bootstrap, error));
    common_tool_registry developer_registry;
    common_tool_adapter_result developer_adapters;
    assert(common_register_native_tool_adapters(developer_catalog, "developer-read", repository_bindings, developer_registry, developer_adapters, error));
    result = developer_registry.execute({"workspace_list", R"({"path":"src","depth":1})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos);
    result = developer_registry.execute({"workspace_search", R"({"query":"needle","path":"src"})"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos);
    result = developer_registry.execute({"workspace_read", R"({"path":"src/sample.txt","start_line":1,"end_line":1})"});
    assert(result.ok && result.output.find("alpha") != std::string::npos);
    result = developer_registry.execute({"repository_status", "{}"});
    assert(result.ok && result.output.find("src") != std::string::npos);
    result = developer_registry.execute({"repository_changed_files", "{}"});
    assert(result.ok && result.output.find("sample.txt") != std::string::npos);
    common_tool_registry native_network_registry;
    common_native_tool_bindings native_network_bindings;
    assert(common_register_native_tool_adapters(research_catalog, "research", native_network_bindings, native_network_registry, adapters, error));
    result = native_network_registry.execute({"web_fetch", R"({"url":"http://127.0.0.1/test"})"});
    assert(!result.ok && result.failure_class == common_tool_failure_class::network);
    std::filesystem::remove_all(repository);
    return 0;
}
