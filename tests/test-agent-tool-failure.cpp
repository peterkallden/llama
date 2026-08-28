#include "agent/agent-runtime.h"
#include "agent/adaptation/adaptation-observer.h"
#include "agent/tooling/registry/tool-registry.h"
#include "plan/plan-in-memory.h"
#include "test-tool-runtime-registry-adapter.h"

#include <cassert>

class planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "tool-failure";
        proposal.plan.status = common_plan_status::active;
        common_plan_step fetch{"fetch", "Fetch", "Fetch an unavailable resource"};
        fetch.status = common_plan_step_status::active;
        fetch.selected_tool = "failing_lookup";
        fetch.tool_call = common_plan_tool_call{"failing_lookup", R"({"id":"missing"})"};
        common_plan_step answer{"answer", "Answer", "Report the failed evidence honestly"};
        answer.depends_on = {"fetch"};
        proposal.plan.steps = {fetch, answer};
        proposal.plan.active_step_id = "fetch";
        return proposal;
    }
};

class executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state & plan, const std::vector<std::string> &, std::string & error) override {
        assert(plan.steps[0].status == common_plan_step_status::failed);
        assert(plan.observations.size() == 1);
        error.clear();
        return "The lookup failed; no result was claimed.";
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

class adaptation_observer final : public common_agent_adaptation_observer {
public:
    bool called = false;
    bool observe(const common_agent_request &, const common_plan_state & plan,
            const common_agent_result & result, std::string & error) override {
        called = true;
        assert(plan.id == "tool-failure");
        assert(!result.learning_signals.empty());
        error.clear();
        return true;
    }
};

int main() {
    std::string error;
    common_plan_in_memory_store store;
    assert(store.open("", error));
    common_tool_registry registry;
    common_registered_tool tool;
    tool.name = "failing_lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string &) {
        return common_tool_execution_result::failure("tool.lookup.temporary_network_failure", common_tool_failure_class::network, true,
            "Lookup failed because the network is temporarily unavailable.", "temporary network failure");
    };
    assert(registry.register_tool(std::move(tool), error));
    planner p; executor e; reflector r;
    test_tool_runtime_registry_adapter tool_runtime(registry);
    adaptation_observer adaptation;
    common_agent_runtime runtime(store, p, e, r, &tool_runtime);
    runtime.set_adaptation_observer(&adaptation);
    common_agent_request request;
    request.prompt = "fetch";
    request.max_tool_batches = 1;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "The lookup failed; no result was claimed.");
    assert(adaptation.called);
    const auto plan = store.get("tool-failure", error);
    assert(plan && plan->steps[0].status == common_plan_step_status::failed && plan->observations.size() == 1 && plan->observations[0].summary.find("temporarily unavailable") != std::string::npos);
    return 0;
}
