#pragma once

#include "plan/plan-types.h"

// Instantiates a persisted blueprint as an independently mutable task plan.
bool common_plan_instantiate_blueprint(
    const common_plan_state & blueprint,
    const std::string & instance_id,
    const std::string & session_id,
    common_plan_state & instance,
    std::string & error,
    common_plan_scope scope = common_plan_scope::turn,
    int64_t now = 0);
