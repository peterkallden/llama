#include "agent/agent-runtime.h"
#include "agent/tooling/adapters/tool-adapters.h"
#include "plan/plan-in-memory.h"
#include "test-tool-runtime-registry-adapter.h"

#include <cassert>
#include <filesystem>
#include <fstream>

class planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "repository-chain";
        proposal.plan.status = common_plan_status::active;
        common_plan_step search{"search", "Search", "Find the target source"};
        search.status = common_plan_step_status::active;
        search.selected_tool = "repository.search";
        // Runtime supplies bounded read-only defaults for path and max_results.
        search.tool_call = common_plan_tool_call{"repository.search", R"({"query":"target-symbol"})"};
        common_plan_step read{"read", "Read", "Read the matched source"};
        read.depends_on = {"search"};
        read.selected_tool = "repository.read";
        read.tool_call = common_plan_tool_call{"repository.read", R"({"path":{"$from_step":"search","$json_pointer":"/matches/0/path"}})"};
        common_plan_step assess{"assess", "Assess", "Assess whether the selected file is relevant"};
        assess.mode = common_plan_step_mode::reasoning;
        assess.depends_on = {"read"};
        common_plan_step answer{"answer", "Answer", "Summarize the evidence"};
        answer.mode = common_plan_step_mode::final_response;
        answer.depends_on = {"assess"};
        proposal.plan.steps = {search, read, assess, answer};
        proposal.plan.active_step_id = "search";
        return proposal;
    }
};

class executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state & plan, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        assert(plan.observations.size() == 3);
        return "verified repository evidence";
    }

    std::string generate_reasoning(const common_agent_request &, const common_plan_state &, const common_plan_step & step, std::string & error) override {
        assert(step.id == "assess");
        error.clear();
        return R"({"summary":"target source contains verified content"})";
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
    const auto root = std::filesystem::temp_directory_path() / "llama-agent-repository-chain";
    std::filesystem::create_directories(root / "src");
    { std::ofstream file(root / "src" / "target.txt"); file << "target-symbol\nverified content\n"; }
    std::string error;
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    assert(catalog.bootstrap("research", bootstrap, error));
    common_tool_registry registry;
    common_native_tool_bindings bindings;
    bindings.repository_root = root.string();
    common_tool_adapter_result adapters;
    assert(common_register_native_tool_adapters(catalog, "research", bindings, registry, adapters, error));
    test_tool_runtime_registry_adapter tool_runtime(registry);
    common_plan_in_memory_store store;
    assert(store.open("", error));
    planner p; executor e; reflector r;
    common_agent_runtime runtime(store, p, e, r, &tool_runtime);
    common_agent_request request;
    request.prompt = "Find target-symbol";
    request.max_tool_batches = 2;
    request.max_reflection_rounds = 1;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "verified repository evidence");
    const auto plan = store.get("repository-chain", error);
    assert(plan && plan->status == common_plan_status::completed && plan->observations.size() == 3 && plan->steps[2].status == common_plan_step_status::completed);
    std::filesystem::remove_all(root);
    return 0;
}
