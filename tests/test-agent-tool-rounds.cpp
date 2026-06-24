#include "agent/agent-runtime.h"
#include "plan/plan-in-memory.h"

#include <cassert>

class planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "two-rounds";
        proposal.plan.goal = "Use two bounded lookups";
        common_plan_step first{"first", "First lookup", "Get first fact"};
        first.status = common_plan_step_status::active;
        first.selected_tool = "lookup";
        first.tool_call = common_plan_tool_call{"lookup", R"({"id":"first"})"};
        common_plan_step second{"second", "Second lookup", "Get follow-up fact"};
        second.selected_tool = "lookup";
        second.tool_call = common_plan_tool_call{"lookup", R"({"id":"second"})"};
        second.depends_on = {"first"};
        common_plan_step answer{"answer", "Answer", "Synthesize the verified result"};
        answer.depends_on = {"second"};
        proposal.plan.steps = {first, second, answer};
        proposal.plan.active_step_id = "first";
        proposal.plan.status = common_plan_status::active;
        return proposal;
    }
};

class executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state & plan, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "observations=" + std::to_string(plan.observations.size());
    }
};

class reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state & plan, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        assert(plan.observations.size() == 2);
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        return result;
    }
};

int main() {
    common_plan_in_memory_store store;
    std::string error;
    assert(store.open("", error));
    common_tool_registry registry;
    common_registered_tool tool;
    tool.name = "lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string & input) {
        return common_tool_execution_result::success(input.find("second") == std::string::npos ? "first result" : "second result");
    };
    assert(registry.register_tool(std::move(tool), error));
    planner p; executor e; reflector r;
    common_agent_runtime runtime(store, p, e, r, &registry);
    common_agent_request request;
    request.prompt = "two rounds";
    request.max_iterations = 2;
    request.max_reflection_rounds = 2;
    request.max_tool_batches = 2;
    const auto result = runtime.run(request);
    assert(result.error.empty());
    assert(result.response == "observations=2");
    auto plan = store.get("two-rounds", error);
    assert(plan && plan->status == common_plan_status::completed && plan->observations.size() == 2 && plan->steps[0].status == common_plan_step_status::completed && plan->steps[1].status == common_plan_step_status::completed && plan->steps[2].status == common_plan_step_status::completed);
    return 0;
}
