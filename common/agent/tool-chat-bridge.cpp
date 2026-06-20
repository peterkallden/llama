#include "agent/tool-chat-bridge.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

bool common_tool_profile_to_chat_tools(const common_tool_catalog & catalog, const std::string & profile_id,
        const common_tool_registry & registry, std::vector<common_chat_tool> & tools, std::string & error) {
    tools.clear();
    const auto definitions = catalog.load_profile(profile_id, error);
    if (!error.empty()) return false;
    for (const auto & definition : definitions) {
        if (!definition.enabled || !registry.contains(definition.name)) continue;
        const bool read_only = definition.risk_class == common_tool_risk_class::local_read && registry.is_read_only(definition.name);
        const bool proposal = definition.risk_class == common_tool_risk_class::memory_proposal && definition.requires_confirmation && registry.is_policy_gated(definition.name);
        if (!read_only && !proposal) continue;
        tools.push_back({definition.name, definition.description, definition.input_schema_json});
    }
    error.clear();
    return true;
}

bool common_tool_dispatch_chat_calls(common_chat_msg & assistant_message, const common_tool_registry & registry,
        size_t max_calls, common_tool_chat_dispatch_result & result, std::string & error) {
    result = {};
    if (assistant_message.role != "assistant") { error = "only assistant messages may contain tool calls"; return false; }
    if (assistant_message.tool_calls.size() > max_calls) { error = "tool call batch exceeds configured limit"; return false; }
    for (size_t index = 0; index < assistant_message.tool_calls.size(); ++index) {
        auto & call = assistant_message.tool_calls[index];
        if (call.id.empty()) call.id = "native-tool-" + std::to_string(index + 1);
        common_chat_msg tool_message;
        tool_message.role = "tool";
        tool_message.tool_name = call.name;
        tool_message.tool_call_id = call.id;
        std::string output, call_error;
        if (!registry.contains(call.name)) {
            tool_message.content = json({{"ok", false}, {"error", {{"code", "tool_unavailable"}, {"message", "tool is not registered"}}}}).dump();
        } else if (!registry.is_read_only(call.name) && !registry.is_policy_gated(call.name)) {
            tool_message.content = json({{"ok", false}, {"error", {{"code", "tool_not_read_only"}, {"message", "tool is not available in a read-only batch"}}}}).dump();
        } else if (!registry.execute({call.name, call.arguments}, output, call_error)) {
            tool_message.content = json({{"ok", false}, {"error", {{"code", "tool_call_rejected"}, {"message", call_error}}}}).dump();
        } else {
            if (output.size() > 4096) output.resize(4096);
            const auto value = json::parse(output, nullptr, false);
            tool_message.content = value.is_discarded() ? json({{"ok", true}, {"result_text", output}}).dump() : json({{"ok", true}, {"result", value}}).dump();
            ++result.executed;
        }
        result.tool_messages.push_back(std::move(tool_message));
    }
    error.clear();
    return true;
}
