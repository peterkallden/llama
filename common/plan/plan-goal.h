#pragma once

#include "plan/plan-types.h"

#include <string>
#include <vector>

struct common_plan_goal_evaluation {
    bool structurally_complete = false;
    bool evidence_sufficient = false;
    float confidence = 0.0f;
    std::vector<std::string> unmet_criteria;
};

// Native structural/evidence evaluation. It intentionally does not decide
// semantic correctness; reflection may assess that separately from evidence.
common_plan_goal_evaluation common_plan_evaluate_goal(const common_plan_state & plan);
