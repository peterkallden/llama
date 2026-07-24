#include "agent-tool-runtime-adapter.h"

#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

class provider_agent_tool_runtime final : public common_agent_tool_runtime {
public:
    explicit provider_agent_tool_runtime(agent_tool_view & tool_view)
        : tool_view(tool_view) {}

    bool is_read_only(const std::string & tool_name) const override {
        return tool_view.is_read_only(tool_name);
    }

    bool is_policy_gated(const std::string & tool_name) const override {
        return tool_view.is_policy_gated(tool_name);
    }

    bool is_available(const std::string & tool_name) const override {
        return tool_view.exposes_tool(tool_name);
    }

    common_agent_tool_repair_context make_repair_context(
            const common_agent_tool_call & call,
            const std::string & validation_error) const override {
        common_agent_tool_repair_context result;
        result.tool_name = call.name;
        result.validation_error = validation_error;
        for (const auto & tool : tool_view.chat_tools()) {
            result.available_tools.push_back(tool.name);
            if (tool.name != call.name || !result.arguments_skeleton.empty()) continue;
            const auto schema = json::parse(tool.parameters, nullptr, false);
            if (!schema.is_object() || !schema.contains("properties") || !schema["properties"].is_object()) continue;
            json skeleton = json::object();
            size_t count = 0;
            for (const auto & [name, property] : schema["properties"].items()) {
                if (count++ >= 24) break;
                if (property.contains("default")) skeleton[name] = property["default"];
                else if (property.value("type", "") == "string") skeleton[name] = "";
                else if (property.value("type", "") == "integer" || property.value("type", "") == "number") skeleton[name] = 0;
                else if (property.value("type", "") == "boolean") skeleton[name] = false;
                else if (property.value("type", "") == "array") skeleton[name] = json::array();
                else skeleton[name] = json::object();
            }
            result.arguments_skeleton = skeleton.dump();
        }
        std::sort(result.available_tools.begin(), result.available_tools.end());
        return result;
    }

    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        return tool_view.validate({"", call.name, call.arguments_json}, error);
    }

    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        std::string error;
        const auto result = tool_view.call({"", call.name, call.arguments_json}, error);
        if (result.ok) {
            return common_tool_execution_result::success(
                result.content_json,
                result.content_summary,
                result.resource_refs);
        }
        return common_tool_execution_result::failure(
            result.failure_code.empty() ? "tool.execution_failed" : result.failure_code,
            result.failure_class,
            result.retryable,
            result.safe_summary.empty() ? "The tool failed." : result.safe_summary,
            result.raw_diagnostic);
    }

    bool supports_async(const common_agent_tool_call & call) const override {
        return tool_view.supports_async_call(call.name);
    }

    bool begin_async(
            const common_agent_tool_call & call,
            common_runtime_operation_ref & pending,
            std::string & error) const override {
        if (!tool_view.begin_call_async(
                {"", call.name, call.arguments_json}, pending, error)) {
            return false;
        }
        error.clear();
        return true;
    }

    bool poll_async(
            const common_runtime_operation_ref & pending,
            bool & ready,
            common_tool_execution_result & output,
            std::string & error) const override {
        agent_tool_result result;
        if (!tool_view.poll_call_async(pending, ready, result, error)) {
            return false;
        }
        if (!ready) {
            error.clear();
            return true;
        }
        if (result.ok) {
            output = common_tool_execution_result::success(
                result.content_json,
                result.content_summary,
                result.resource_refs);
        } else {
            output = common_tool_execution_result::failure(
                result.failure_code.empty() ? "tool.execution_failed" : result.failure_code,
                result.failure_class,
                result.retryable,
                result.safe_summary.empty() ? "The tool failed." : result.safe_summary,
                result.raw_diagnostic);
        }
        error.clear();
        return true;
    }

private:
    agent_tool_view & tool_view;
};

} // namespace

std::unique_ptr<common_agent_tool_runtime> make_provider_agent_tool_runtime(
        agent_tool_view & tool_view) {
    return std::make_unique<provider_agent_tool_runtime>(tool_view);
}
