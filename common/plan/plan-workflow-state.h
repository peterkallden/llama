#pragma once

#include "plan/plan-types.h"

#include <string>

// Replays completed, host-owned workflow transition aliases from the active
// route binding. Pending transitions and model prose never advance state.
std::string common_plan_workflow_completed_state(
    const common_plan_state & plan,
    const common_plan_workflow_definition & definition);

bool common_plan_workflow_is_terminal(
    const common_plan_state & plan,
    const common_plan_workflow_definition & definition);
