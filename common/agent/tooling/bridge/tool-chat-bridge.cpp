#include "agent/tooling/bridge/tool-chat-bridge.h"
#include "agent/tooling/contracts/tool-result-contracts.h"
#include "agent/tooling/schema/tool-schema-compact.h"

bool common_tool_profile_to_chat_tools(const common_tool_catalog & catalog, const std::string & profile_id,
        const common_tool_registry & registry, std::vector<common_chat_tool> & tools, std::string & error) {
    tools.clear();
    const auto definitions = catalog.load_profile(profile_id, error);
    if (!error.empty()) return false;
    for (const auto & definition : definitions) {
        if (!definition.enabled || !registry.matches_binding(definition.name, definition.version, definition.executor_id)) continue;
        const bool read_only = definition.risk_class == common_tool_risk_class::local_read && registry.is_read_only(definition.name);
        const bool proposal = definition.risk_class == common_tool_risk_class::memory_proposal && definition.requires_confirmation && registry.is_policy_gated(definition.name);
        const bool sandbox = definition.risk_class == common_tool_risk_class::sandbox_execution && definition.requires_confirmation && registry.is_policy_gated(definition.name);
        if (!read_only && !proposal && !sandbox) continue;
        std::string compact_error;
        const auto model_description = common_render_compact_tool_description(
            definition.name,
            definition.description,
            common_tool_model_input_schema(definition),
            common_tool_model_result_schema(definition),
            compact_error);
        if (!compact_error.empty()) {
            error = compact_error;
            return false;
        }
        tools.push_back({definition.name, model_description,
            common_tool_model_input_schema(definition),
            common_tool_model_result_schema(definition)});
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
        if (!registry.contains(call.name)) {
            tool_message.content = common_tool_chat_failure_payload_to_json(
                "tool_unavailable",
                "tool is not registered",
                false,
                common_tool_failure_class::not_found).dump();
        } else if (!registry.is_read_only(call.name) && !registry.is_policy_gated(call.name)) {
            tool_message.content = common_tool_chat_failure_payload_to_json(
                "tool_not_read_only",
                "tool is not available in a read-only batch",
                false,
                common_tool_failure_class::policy).dump();
        } else {
            const auto execution = registry.execute({call.name, call.arguments});
            if (!execution.ok) {
                tool_message.content = common_tool_chat_failure_payload_to_json(
                    execution.failure_code.empty() ? "tool_call_rejected" : execution.failure_code,
                    execution.safe_summary.empty() ? "The tool call was rejected by its native contract or executor." : execution.safe_summary,
                    execution.retryable,
                    execution.failure_class).dump();
                result.tool_messages.push_back(std::move(tool_message));
                continue;
            }
            auto output = execution.output;
            if (output.size() > 4096) output.resize(4096);
            tool_message.content = common_tool_chat_success_payload_to_json(output).dump();
            ++result.executed;
        }
        result.tool_messages.push_back(std::move(tool_message));
    }
    error.clear();
    return true;
}
