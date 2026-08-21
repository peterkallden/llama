#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/bridge/tool-chat-bridge.h"

#include <cassert>

int main() {
    std::string error;
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    assert(catalog.bootstrap("minimal", bootstrap, error));
    common_tool_registry registry;
    common_tool_adapter_result adapters;
    assert(common_register_native_tool_adapters(catalog, "minimal", {}, registry, adapters, error));

    std::vector<common_chat_tool> tools;
    assert(common_tool_profile_to_chat_tools(catalog, "minimal", registry, tools, error));
    assert(tools.size() == 2);
    assert(tools[0].name == "calculator");
    assert(tools[0].description.find("args: expression:string") != std::string::npos);
    assert(tools[0].description.find("returns:") != std::string::npos);
    assert(tools[0].parameters.find("expression") != std::string::npos);

    common_chat_msg assistant;
    assistant.role = "assistant";
    assistant.tool_calls.push_back({"calculator", R"({"expression":"7 * 6"})", ""});
    common_tool_chat_dispatch_result dispatched;
    assert(common_tool_dispatch_chat_calls(assistant, registry, 1, dispatched, error));
    assert(assistant.tool_calls[0].id == "native-tool-1");
    assert(dispatched.executed == 1 && dispatched.tool_messages.size() == 1);
    assert(dispatched.tool_messages[0].role == "tool");
    assert(dispatched.tool_messages[0].tool_call_id == "native-tool-1");
    assert(dispatched.tool_messages[0].content.find("42") != std::string::npos);

    assistant.tool_calls.push_back({"time_now", "{}", "second"});
    assert(!common_tool_dispatch_chat_calls(assistant, registry, 1, dispatched, error));
    assert(error == "tool call batch exceeds configured limit");

    common_tool_catalog memory_catalog;
    assert(memory_catalog.bootstrap("memory", bootstrap, error));
    common_tool_registry proposal_registry;
    common_native_tool_bindings proposal_bindings;
    proposal_bindings.memory_remember_proposal = [](const std::string &) { return common_tool_execution_result::success(R"({"decision":"accept"})"); };
    assert(common_register_native_tool_adapters(memory_catalog, "memory", proposal_bindings, proposal_registry, adapters, error));
    assert(common_tool_profile_to_chat_tools(memory_catalog, "memory", proposal_registry, tools, error));
    bool has_remember = false;
    for (const auto & tool : tools) has_remember = has_remember || tool.name == "memory_remember";
    assert(has_remember);
    return 0;
}
