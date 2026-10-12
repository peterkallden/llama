#include "plan/plan-template.h"

#include <unordered_map>

bool common_plan_instantiate_template(const common_plan_template & source, const std::string & instance_id, const std::string & session_id, common_plan_state & plan, std::string & error, const common_plan_template_options & options) {
    if (source.id.empty() || instance_id.empty() || session_id.empty()) { error = "template id, instance id and session id are required"; return false; }
    if (source.goal.empty() || source.success_criteria.empty()) { error = "template goal and success criteria are required"; return false; }
    common_plan_state out;
    out.id = instance_id; out.session_id = session_id; out.scope = options.scope; out.status = common_plan_status::proposed;
    out.purpose = source.goal; out.goal = source.goal; out.success_criteria = source.success_criteria; out.constraints = source.constraints; out.next_action = source.next_action; out.created_at = options.now; out.updated_at = options.now;
    std::unordered_map<std::string, std::string> ids;
    for (const auto & step : source.steps) {
        if (step.id.empty() || ids.count(step.id)) { error = "template contains an empty or duplicate step id"; return false; }
        ids[step.id] = instance_id + ":" + step.id;
    }
    for (const auto & source_step : source.steps) {
        auto step = source_step;
        step.id = ids.at(source_step.id); step.status = common_plan_step_status::pending; step.blocked_by.clear(); step.selected_tool.reset(); step.result_summary.reset(); step.created_at = options.now; step.updated_at = options.now;
        for (auto & dependency : step.depends_on) { const auto found = ids.find(dependency); if (found == ids.end()) { error = "template step references an unknown dependency"; return false; } dependency = found->second; }
        out.steps.push_back(std::move(step));
    }
    if (options.copy_assumptions) { out.assumptions = source.assumptions; for (auto & assumption : out.assumptions) { assumption.id = instance_id + ":" + assumption.id; assumption.valid = true; } }
    if (options.activate_first_ready_step) for (auto & step : out.steps) if (step.depends_on.empty()) { step.status = common_plan_step_status::active; out.active_step_id = step.id; out.status = common_plan_status::active; break; }
    plan = std::move(out); error.clear(); return true;
}
