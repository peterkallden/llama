#include "plan/plan-blueprint.h"

#include <unordered_map>

bool common_plan_instantiate_blueprint(
        const common_plan_state & blueprint,
        const std::string & instance_id,
        const std::string & session_id,
        common_plan_state & instance,
        std::string & error,
        common_plan_scope scope,
        int64_t now) {
    if (blueprint.kind != common_plan_kind::blueprint) { error = "source plan is not a blueprint"; return false; }
    if (blueprint.id.empty() || instance_id.empty() || session_id.empty()) { error = "blueprint id, instance id and session id are required"; return false; }
    std::unordered_map<std::string, std::string> ids;
    for (const auto & step : blueprint.steps) {
        if (step.id.empty() || !ids.emplace(step.id, instance_id + ":" + step.id).second) { error = "blueprint contains an empty or duplicate step id"; return false; }
    }
    common_plan_state out;
    out.id = instance_id;
    out.session_id = session_id;
    out.kind = common_plan_kind::task;
    out.derived_from_plan_id = blueprint.id;
    out.scope = scope;
    out.purpose = blueprint.purpose.empty() ? blueprint.goal : blueprint.purpose;
    out.goal = blueprint.goal;
    out.success_criteria = blueprint.success_criteria;
    out.required_capabilities = blueprint.required_capabilities;
    out.constraints = blueprint.constraints;
    out.assumptions = blueprint.assumptions;
    out.next_action = blueprint.next_action;
    out.created_at = now;
    out.updated_at = now;
    for (const auto & source : blueprint.steps) {
        auto step = source;
        step.id = ids.at(source.id);
        step.status = common_plan_step_status::pending;
        step.blocked_by.clear();
        step.selected_tool.reset();
        step.tool_call.reset();
        step.result_summary.reset();
        step.created_at = now;
        step.updated_at = now;
        for (auto & dependency : step.depends_on) {
            const auto found = ids.find(dependency);
            if (found == ids.end()) { error = "blueprint step references an unknown dependency"; return false; }
            dependency = found->second;
        }
        out.steps.push_back(std::move(step));
    }
    for (auto & assumption : out.assumptions) assumption.valid = true;
    for (auto & step : out.steps) if (step.depends_on.empty()) {
        step.status = common_plan_step_status::active;
        out.active_step_id = step.id;
        out.status = common_plan_status::active;
        break;
    }
    instance = std::move(out);
    error.clear();
    return true;
}
