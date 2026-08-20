#include "agent/agent-runtime.h"
#include "agent/tooling/registry/tool-registry.h"
#include "plan/plan-in-memory.h"
#include "test-tool-runtime-registry-adapter.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

class planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "memory-search-fallback";
        proposal.plan.goal = "Find a stored preference";
        proposal.plan.status = common_plan_status::active;
        common_plan_step step{"search", "Search memory", "Find relevant memory"};
        step.status = common_plan_step_status::active;
        step.selected_tool = "memory_search";
        step.tool_call = common_plan_tool_call{"memory_search", "{}"};
        proposal.plan.steps.push_back(std::move(step));
        proposal.plan.active_step_id = "search";
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
        return {};
    }
};

int main() {
    common_plan_in_memory_store store;
    std::string error;
    assert(store.open("", error));
    common_tool_registry tools;
    common_registered_tool tool;
    tool.name = "memory_search";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string"}}})";
    tool.handler = [](const std::string & input) {
        assert(input == R"({"query":"What is my favorite breakfast?"})");
        return common_tool_execution_result::success("stored preference");
    };
    assert(tools.register_tool(std::move(tool), error));

    planner p;
    executor e;
    reflector r;
    test_tool_runtime_registry_adapter tool_runtime(tools);
    common_agent_runtime runtime(store, p, e, r, &tool_runtime);
    common_agent_request request;
    request.prompt = "What is my favorite breakfast?";
    request.max_tool_batches = 1;
    const auto result = runtime.run(request);
    assert(result.error.empty());
    assert(result.response == "draft");
    return 0;
}
