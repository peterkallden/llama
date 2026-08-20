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

    common_plan_state tool_evidence; tool_evidence.id = "tool-evidence"; tool_evidence.goal = "tool evidence"; tool_evidence.status = common_plan_status::active;
    common_plan_step tool_step; tool_step.id = "read"; tool_step.title = "Read"; tool_step.objective = "Read source";
    tool_step.status = common_plan_step_status::active; tool_step.tool_call = common_plan_tool_call{"resource_read", R"({"uri":"resource://one"})"};
    tool_evidence.steps = {tool_step}; tool_evidence.active_step_id = "read"; assert(store.create(tool_evidence, error));
    auto missing_tool_observation = op(common_plan_operation_kind::complete_step, tool_evidence, "read");
    assert(!store.apply(missing_tool_observation, tool_evidence, error));
    auto prose_evidence = op(common_plan_operation_kind::complete_step, tool_evidence, "read"); prose_evidence.evidence_ids = {"model:claim"};
    assert(!store.apply(prose_evidence, tool_evidence, error));
    auto tool_observation = op(common_plan_operation_kind::record_observation, tool_evidence);
    tool_observation.observation = common_plan_observation{"tool:read:resource_read", "resource_read", "read result", 1.0f, {}, {}, 0};
    assert(store.apply(tool_observation, tool_evidence, error));
    auto observed_complete = op(common_plan_operation_kind::complete_step, tool_evidence, "read"); observed_complete.evidence_ids = {"tool:read:resource_read"};
    assert(store.apply(observed_complete, tool_evidence, error));

    common_plan_state bindable; bindable.id = "bindable"; bindable.goal = "test"; bindable.status = common_plan_status::active;
    common_plan_step target; target.id = "orient"; target.title = "Orient"; target.objective = "Inspect"; target.mode = common_plan_step_mode::reasoning; target.status = common_plan_step_status::active;
    bindable.steps = {target}; bindable.active_step_id = "orient"; assert(store.create(bindable, error));
    auto bind = op(common_plan_operation_kind::revise_step, bindable); auto replacement = target; replacement.mode = common_plan_step_mode::tool; replacement.selected_tool = "repository.search"; replacement.tool_call = common_plan_tool_call{"repository.search", R"({"query":"plan"})"}; bind.step = replacement; assert(store.apply(bind, bindable, error));
    assert(bindable.steps.front().tool_call && bindable.steps.front().selected_tool == "repository.search");
    auto alter_dependency = op(common_plan_operation_kind::revise_step, bindable); replacement.depends_on = {"missing"}; alter_dependency.step = replacement; assert(!store.apply(alter_dependency, bindable, error));
    auto alter_objective = op(common_plan_operation_kind::revise_step, bindable); replacement = bindable.steps.front(); replacement.objective = "changed"; alter_objective.step = replacement; assert(!store.apply(alter_objective, bindable, error));

    common_plan_state repairable; repairable.id = "repairable"; repairable.goal = "repair"; repairable.status = common_plan_status::active;
    common_plan_step failed; failed.id = "fetch"; failed.title = "Fetch"; failed.objective = "Fetch evidence"; failed.status = common_plan_step_status::failed; failed.selected_tool = "lookup"; failed.tool_call = common_plan_tool_call{"lookup", R"({"id":"old"})"};
    repairable.steps = {failed}; assert(store.create(repairable, error));
    auto reset = op(common_plan_operation_kind::reset_step, repairable, "fetch"); assert(store.apply(reset, repairable, error)); assert(repairable.steps.front().status == common_plan_step_status::pending);
    auto activate_repair = op(common_plan_operation_kind::activate_step, repairable, "fetch"); assert(store.apply(activate_repair, repairable, error));
    auto fail_again = op(common_plan_operation_kind::fail_step, repairable, "fetch"); assert(store.apply(fail_again, repairable, error));
    auto replace = op(common_plan_operation_kind::replace_step, repairable, "fetch"); auto corrected = repairable.steps.front(); corrected.tool_call = common_plan_tool_call{"lookup", R"({"id":"new"})"}; replace.step = corrected; assert(store.apply(replace, repairable, error));
    assert(repairable.steps.front().status == common_plan_step_status::pending && repairable.steps.front().tool_call->arguments_json.find("new") != std::string::npos);

    common_plan_state completed; completed.id = "completed-step"; completed.goal = "completed"; completed.status = common_plan_status::active;
    common_plan_step done_step; done_step.id = "done"; done_step.title = "Done"; done_step.status = common_plan_step_status::completed;
    completed.steps = {done_step}; assert(store.create(completed, error));
    auto reset_completed = op(common_plan_operation_kind::reset_step, completed, "done"); assert(!store.apply(reset_completed, completed, error));
    auto replace_completed = op(common_plan_operation_kind::replace_step, completed, "done"); replace_completed.step = done_step; assert(!store.apply(replace_completed, completed, error));
    store.close(); return 0;
}
