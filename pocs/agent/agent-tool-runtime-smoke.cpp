#include "agent-tool-provider.h"

#include "agent/agent-runtime.h"
#include "agent/tool-catalog.h"
#include "plan/plan-in-memory.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

class provider_agent_tool_runtime final : public common_agent_tool_runtime {
public:
    explicit provider_agent_tool_runtime(agent_tool_view & tool_view)
        : tool_view(tool_view) {}

    bool is_read_only(const std::string & tool_name) const override {
        return tool_view.is_read_only(tool_name);
    }

    bool is_policy_gated(const std::string &) const override {
        return false;
    }

    bool validate(const common_registered_tool_call & call, std::string & error) const override {
        return tool_view.validate({"", call.name, call.arguments_json}, error);
    }

    common_tool_execution_result execute(const common_registered_tool_call & call) const override {
        std::string error;
        const auto result = tool_view.call({"", call.name, call.arguments_json}, error);
        if (result.ok) {
            return common_tool_execution_result::success(result.content_json);
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

class smoke_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        common_plan_proposal proposal;
        proposal.plan.id = "runtime-smoke-plan";
        proposal.plan.namespace_id = request.namespace_id;
        proposal.plan.session_id = request.session_id;
        proposal.plan.project_id = request.project_id;
        proposal.plan.turn_id = request.turn_id;
        proposal.plan.scope = request.plan_scope;
        proposal.plan.status = common_plan_status::active;
        proposal.plan.purpose = request.prompt;
        proposal.plan.goal = "Run calculator and answer with the result.";
        proposal.plan.success_criteria = "Return the calculator result.";

        common_plan_step tool_step;
        tool_step.id = "step-tool";
        tool_step.title = "Run calculator";
        tool_step.objective = "Evaluate a simple arithmetic expression.";
        tool_step.intended_contribution = tool_step.objective;
        tool_step.status = common_plan_step_status::active;
        tool_step.mode = common_plan_step_mode::tool;
        tool_step.selected_tool = "calculator";
        tool_step.tool_call = common_plan_tool_call{"calculator", R"({"expression":"17 * 23"})"};

        common_plan_step final_step;
        final_step.id = "step-final";
        final_step.title = "Return answer";
        final_step.objective = "Respond with the calculator result.";
        final_step.intended_contribution = final_step.objective;
        final_step.status = common_plan_step_status::pending;
        final_step.mode = common_plan_step_mode::final_response;
        final_step.depends_on.push_back(tool_step.id);

        proposal.plan.active_step_id = tool_step.id;
        proposal.plan.steps.push_back(std::move(tool_step));
        proposal.plan.steps.push_back(std::move(final_step));
        error.clear();
        return proposal;
    }
};

class smoke_executor final : public common_action_executor {
public:
    std::string generate_draft(
            const common_agent_request &,
            const common_plan_state & plan,
            const std::vector<std::string> &,
            std::string & error) override {
        for (const auto & observation : plan.observations) {
            if (observation.source == "calculator" && observation.summary.find("391") != std::string::npos) {
                error.clear();
                return "The result is 391.";
            }
        }
        error = "calculator observation missing expected value";
        return {};
    }
};

class smoke_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(
            const common_agent_request &,
            const common_plan_state &,
            const std::string &,
            std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        result.confidence = 1.0f;
        return result;
    }
};

} // namespace

int main() {
    std::string error;
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("minimal", bootstrap, error)) {
        std::fprintf(stderr, "tool bootstrap failed: %s\n", error.c_str());
        return 1;
    }

    native_agent_tool_provider provider(
        catalog,
        [](const agent_tool_context &, common_native_tool_bindings &, std::string &) {
            return true;
        });

    agent_tool_context tool_context;
    tool_context.request_id = "runtime-smoke";
    tool_context.turn_id = "turn-1";
    tool_context.profile_id = "minimal";
    tool_context.max_calls = 2;

    std::unique_ptr<agent_tool_view> tool_view = provider.resolve_tools(tool_context, error);
    if (!tool_view) {
        std::fprintf(stderr, "provider resolve failed: %s\n", error.c_str());
        return 1;
    }

    provider_agent_tool_runtime tool_runtime(*tool_view);
    common_plan_in_memory_store plan_store;
    if (!plan_store.open("", error)) {
        std::fprintf(stderr, "failed to open plan store: %s\n", error.c_str());
        return 1;
    }

    smoke_planner planner;
    smoke_executor executor;
    smoke_reflector reflector;
    common_agent_runtime runtime(plan_store, planner, executor, reflector, &tool_runtime, nullptr);

    common_agent_request request;
    request.prompt = "What is 17 * 23?";
    request.namespace_id = "runtime-smoke";
    request.session_id = "runtime-smoke-session";
    request.turn_id = "runtime-smoke-turn";
    request.plan_scope = common_plan_scope::session;
    request.enable_planning = true;
    request.enable_reflection = false;
    request.max_iterations = 1;
    request.max_tool_batches = 1;

    const auto result = runtime.run(request);
    if (!result.error.empty()) {
        std::fprintf(stderr, "runtime run failed: %s\n", result.error.c_str());
        return 1;
    }
    if (result.response.find("391") == std::string::npos) {
        std::fprintf(stderr, "runtime response missing expected value: %s\n", result.response.c_str());
        return 1;
    }
    if (!result.plan_id || *result.plan_id != "runtime-smoke-plan") {
        std::fprintf(stderr, "runtime did not retain expected plan id\n");
        return 1;
    }

    std::printf("runtime_response=%s\n", result.response.c_str());
    std::printf("tool_events=%zu\n", result.events.size());
    std::printf("plan_id=%s\n", result.plan_id ? result.plan_id->c_str() : "");
    return 0;
}
