#include "plan/plan-types.h"

bool common_plan_scope_matches(const common_plan_state & plan, common_plan_scope scope,
        const std::string & namespace_id, const std::string & session_id,
        const std::string & project_id, const std::string & turn_id) {
    if (plan.scope != scope || plan.namespace_id != namespace_id) return false;
    switch (scope) {
        case common_plan_scope::turn:
            return !turn_id.empty() && !session_id.empty() && plan.turn_id == turn_id &&
                plan.session_id == session_id && (project_id.empty() || plan.project_id == project_id);
        case common_plan_scope::session:
            return !session_id.empty() && plan.session_id == session_id &&
                (project_id.empty() || plan.project_id == project_id);
        case common_plan_scope::project: return !project_id.empty() && plan.project_id == project_id;
        case common_plan_scope::global: return true;
    }
    return false;
}
