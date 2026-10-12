#include "plan/plan-goal.h"

#include <algorithm>

common_plan_goal_evaluation common_plan_evaluate_goal(const common_plan_state & plan) {
    common_plan_goal_evaluation result;
    result.structurally_complete = plan.status == common_plan_status::completed;
    if (!result.structurally_complete) result.unmet_criteria.push_back("plan is not completed");
    if (plan.purpose.empty()) result.unmet_criteria.push_back("plan has no purpose");
    if (plan.goal.empty()) result.unmet_criteria.push_back("plan has no goal");
    if (plan.success_criteria.empty()) result.unmet_criteria.push_back("plan has no success criteria");
    size_t required = 0, evidenced = 0;
    for (const auto & step : plan.steps) {
        if (step.optional || common_plan_step_effective_mode(step) == common_plan_step_mode::final_response) continue;
        ++required;
        const bool observation = std::any_of(plan.observations.begin(), plan.observations.end(), [&](const auto & item) {
            return item.id == step.id || item.id.rfind("tool:" + step.id + ":", 0) == 0 || item.id == "reasoning:" + step.id;
        });
        if (step.result_summary || observation) ++evidenced;
        else result.unmet_criteria.push_back("step lacks evidence: " + step.id);
    }
    result.evidence_sufficient = result.structurally_complete && !plan.purpose.empty() && !plan.goal.empty() && !plan.success_criteria.empty() && evidenced == required;
    result.confidence = required == 0 ? (result.evidence_sufficient ? 1.0f : 0.0f) : static_cast<float>(evidenced) / required;
    return result;
}
