#include "agent-cli-host-adapter.h"

#include "memory/memory-in-memory.h"

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
    common_memory_in_memory_store store;
    if (!store.open("", error)) {
        std::fprintf(stderr, "memory store open failed: %s\n", error.c_str());
        return 1;
    }

    args options;
    options.max_tool_rounds = 2;
    options.memory_scope = "session";
    options.memory_turn = "legacy-smoke-turn";
    options.memory_namespace = "legacy-smoke";
    options.memory_session = "session-1";
    options.memory_project = "project-1";

    std::unique_ptr<agent_tool_view> tool_view = make_agent_cli_legacy_memory_tool_view(
        store,
        options,
        true,
        true);
    if (!tool_view) {
        std::fprintf(stderr, "legacy memory tool view was not created\n");
        return 1;
    }
    if (!has_tool(tool_view->chat_tools(), "memory_search") || !has_tool(tool_view->chat_tools(), "memory_remember")) {
        std::fprintf(stderr, "legacy memory tool view did not expose expected tools\n");
        return 1;
    }

    const auto remember_result = tool_view->call({
        "legacy-call-1",
        "memory_remember",
        R"({"kind":"fact","content":"Legacy memory tools now dispatch through agent_tool_view."})",
    }, error);
    if (!remember_result.ok || remember_result.content_json.find("\"decision\":\"accept\"") == std::string::npos) {
        std::fprintf(stderr, "legacy memory_remember failed: %s\n", remember_result.content_json.c_str());
        return 1;
    }

    const auto unavailable_result = tool_view->call({
        "legacy-call-2",
        "calculator",
        R"({"expression":"1+1"})",
    }, error);
    if (unavailable_result.ok || unavailable_result.failure_class != common_tool_failure_class::not_found) {
        std::fprintf(stderr, "legacy tool view did not reject unavailable tool as expected\n");
        return 1;
    }

    std::printf("legacy_tools=%zu\n", tool_view->chat_tools().size());
    std::printf("legacy_memory_remember=%s\n", remember_result.content_json.c_str());
    std::printf("legacy_unavailable=%s\n", unavailable_result.content_json.c_str());
    return 0;
}
