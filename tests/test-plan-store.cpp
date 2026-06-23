#include "plan/plan-in-memory.h"
#include <cassert>

static common_plan_operation op(common_plan_operation_kind kind, const common_plan_state & plan, const std::string & step = {}) { common_plan_operation out; out.kind = kind; out.plan_id = plan.id; out.expected_version = plan.version; if (!step.empty()) out.step_id = step; out.reason_summary = "unit test"; return out; }
int main() {
    common_plan_in_memory_store store; std::string error; assert(store.open("", error));
    common_plan_state plan; plan.id = "plan-1"; plan.goal = "test"; plan.status = common_plan_status::proposed; assert(store.create(plan, error));
    auto add_first = op(common_plan_operation_kind::add_step, plan); common_plan_step first; first.id = "first"; first.title = "First"; first.required_evidence = {"memory:one"}; add_first.step = first; assert(store.apply(add_first, plan, error));
    auto activate = op(common_plan_operation_kind::activate_step, plan, "first"); assert(store.apply(activate, plan, error));
    auto complete_missing = op(common_plan_operation_kind::complete_step, plan, "first"); assert(!store.apply(complete_missing, plan, error));
    auto complete = op(common_plan_operation_kind::complete_step, plan, "first"); complete.evidence_ids = {"memory:one"}; assert(store.apply(complete, plan, error));
    auto done = op(common_plan_operation_kind::complete_plan, plan); assert(store.apply(done, plan, error)); assert(plan.status == common_plan_status::completed);
    auto events = store.history("plan-1", error); assert(events.size() == 5); assert(!events[2].accepted); assert(events.back().new_version == plan.version);
    auto stale = op(common_plan_operation_kind::fail_plan, plan); stale.expected_version = 0; assert(!store.apply(stale, plan, error));

    common_plan_state bindable; bindable.id = "bindable"; bindable.goal = "test"; bindable.status = common_plan_status::active;
    common_plan_step target; target.id = "orient"; target.title = "Orient"; target.objective = "Inspect"; target.mode = common_plan_step_mode::reasoning; target.status = common_plan_step_status::active;
    bindable.steps = {target}; bindable.active_step_id = "orient"; assert(store.create(bindable, error));
    auto bind = op(common_plan_operation_kind::revise_step, bindable); auto replacement = target; replacement.mode = common_plan_step_mode::tool; replacement.selected_tool = "repository_search"; replacement.tool_call = common_plan_tool_call{"repository_search", R"({"query":"plan"})"}; bind.step = replacement; assert(store.apply(bind, bindable, error));
    assert(bindable.steps.front().tool_call && bindable.steps.front().selected_tool == "repository_search");
    auto alter_dependency = op(common_plan_operation_kind::revise_step, bindable); replacement.depends_on = {"missing"}; alter_dependency.step = replacement; assert(!store.apply(alter_dependency, bindable, error));
    auto alter_objective = op(common_plan_operation_kind::revise_step, bindable); replacement = bindable.steps.front(); replacement.objective = "changed"; alter_objective.step = replacement; assert(!store.apply(alter_objective, bindable, error));
    store.close(); return 0;
}
