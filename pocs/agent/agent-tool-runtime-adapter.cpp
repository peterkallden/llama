#include "agent-tool-runtime-adapter.h"

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

    bool validate(const common_registered_tool_call & call, std::string & error) const override {
        return tool_view.validate({"", call.name, call.arguments_json}, error);
    }

    common_tool_execution_result execute(const common_registered_tool_call & call) const override {
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

private:
    agent_tool_view & tool_view;
};

} // namespace

std::unique_ptr<common_agent_tool_runtime> make_provider_agent_tool_runtime(
        agent_tool_view & tool_view) {
    return std::make_unique<provider_agent_tool_runtime>(tool_view);
}
