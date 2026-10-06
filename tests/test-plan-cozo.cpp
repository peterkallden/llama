#include "plan/cozo/plan-cozo.h"

#include <cassert>
#include <filesystem>

int main() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / "llama-plan-cozo-test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    std::string error;

    {
        common_plan_cozo_store store;
        assert(store.open((dir / "plan.db").string(), error));
        common_plan_state plan;
        plan.id = "p";
        plan.scope = common_plan_scope::global;
        plan.purpose = "persist and resume";
        plan.goal = "persist";
        plan.workflow_policy = common_plan_workflow_policy::required;
        plan.procedure_refs = {"data-understanding-v1"};
        plan.workflow_bindings = {{"workflow://dataset/analysis", "v1"}};
        plan.selected_workflow = common_plan_workflow_binding{"workflow://dataset/analysis", "v1"};
        plan.workflow_definition = common_plan_workflow_definition{
            "workflow://dataset/analysis", "v1", "dataset", "graph-v1", {"dataset.select"},
            {"dataset.resolve", "dataset.inspect"}, {}, {"context.dataset.available"}};
        plan.workflow_definition->required_capabilities = {"dataset.resolve", "dataset.inspect"};
        plan.workflow_definition->optional_capabilities = {"data.aggregate"};
        plan.workflow_definition->required_context = {"context.dataset.available"};
        plan.route_binding = common_plan_route_binding{
            "blueprint:dataset-analysis:workflow:workflow://dataset/analysis@v1",
            "dataset-analysis", "bp-v1", "workflow://dataset/analysis", "v1",
            "graph-v1", "env-v1", "route-policy-v1"};
        assert(store.create(plan, error));

        auto op = common_plan_operation{};
        op.kind = common_plan_operation_kind::add_step;
        op.plan_id = "p";
        op.expected_version = 0;
        op.reason_summary = "test";
        op.step = common_plan_step{"s", "step", "objective"};
        op.step->intended_contribution = "preserve the persisted evidence";
        op.step->selected_tool = "web_search";
        op.step->tool_call = common_plan_tool_call{"web_search", R"({"query":"cozo"})"};
        assert(store.apply(op, plan, error));
    }

    {
        common_plan_cozo_store store;
        assert(store.open((dir / "plan.db").string(), error));
        auto plan = store.get("p", error);
        assert(plan && plan->scope == common_plan_scope::global &&
            plan->purpose == "persist and resume" && plan->steps.size() == 1 &&
            plan->steps[0].intended_contribution == "preserve the persisted evidence" &&
            plan->steps[0].tool_call && plan->steps[0].tool_call->name == "web_search" &&
            plan->version == 1 && plan->workflow_policy == common_plan_workflow_policy::required &&
            plan->procedure_refs == std::vector<std::string>{"data-understanding-v1"} &&
            plan->workflow_bindings.size() == 1 && plan->selected_workflow &&
            plan->selected_workflow->workflow_ref == "workflow://dataset/analysis" &&
            plan->workflow_definition && plan->workflow_definition->family == "dataset" &&
            plan->workflow_definition->required_capabilities ==
                std::vector<std::string>({"dataset.resolve", "dataset.inspect"}) &&
            plan->workflow_definition->optional_capabilities ==
                std::vector<std::string>({"data.aggregate"}) &&
            plan->workflow_definition->required_context ==
                std::vector<std::string>({"context.dataset.available"}) &&
            plan->route_binding && plan->route_binding->route_id.find("blueprint:dataset-analysis") == 0 &&
            plan->route_binding->execution_envelope_fingerprint == "env-v1");
        auto events = store.history("p", error);
        assert(events.size() == 1);
        assert(store.erase("p", error));
    }

    fs::remove_all(dir, ec);
    return 0;
}
