#include "agent/agent-runtime.h"
#include "plan/plan-in-memory.h"

#include <cassert>

class planner final : public common_planner {
public:
    int calls = 0;

    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        ++calls;
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "turn-1";
        proposal.plan.session_id = request.session_id;
        proposal.plan.goal = request.prompt;
        proposal.plan.success_criteria = "answer";
        common_plan_step step{"lookup", "Lookup", "Get facts"};
        step.status = common_plan_step_status::active;
        step.selected_tool = "lookup";
        step.tool_call = common_plan_tool_call{"lookup", R"({"id":"status"})"};
        proposal.plan.steps.push_back(std::move(step));
        proposal.plan.active_step_id = "lookup";
        proposal.plan.status = common_plan_status::active;
        return proposal;
    }
};

class executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state &, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "draft";
    }
};

class reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        return result;
    }
};

int main() {
    common_plan_in_memory_store store;
    std::string error;
    assert(store.open("", error));
    common_tool_registry tools;
    common_registered_tool tool;
    tool.name = "lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string &, std::string & value, std::string & err) { value = "current status"; err.clear(); return true; };
    assert(tools.register_tool(std::move(tool), error));

    planner p;
    executor e;
    reflector r;
    common_agent_runtime runtime(store, p, e, r, &tools);
    common_agent_request request;
    request.prompt = "answer";
    request.session_id = "s";
    request.plan_scope = common_plan_scope::global;

    const auto created = runtime.run(request);
    assert(created.error.empty() && created.response == "draft");
    assert(created.plan_id && *created.plan_id == "turn-1" && created.reflected);
    const auto plan = store.get("turn-1", error);
    assert(plan && plan->scope == common_plan_scope::global && plan->observations.size() == 1 && plan->observations[0].summary == "current status");

    request.plan_id = "turn-1";
    const auto resumed = runtime.run(request);
    assert(resumed.error.empty() && resumed.plan_id && *resumed.plan_id == "turn-1");
    assert(p.calls == 1 && !resumed.events.empty() && resumed.events.front().detail == "existing plan resumed");
    return 0;
}
