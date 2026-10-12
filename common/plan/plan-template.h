#pragma once

#include "plan/plan-types.h"

#include <optional>

// A template is declarative reusable structure, not a persisted execution record.
struct common_plan_template {
    std::string id;
    std::string goal;
    std::string success_criteria;
    std::vector<common_plan_step> steps;
    std::vector<common_plan_constraint> constraints;
    std::vector<common_plan_assumption> assumptions;
    std::optional<std::string> next_action;
};

struct common_plan_template_options {
    common_plan_scope scope = common_plan_scope::turn;
    bool copy_assumptions = true;
    bool activate_first_ready_step = true;
    int64_t now = 0;
};

// Uses the caller-supplied instance id to make cloning deterministic and auditable.
bool common_plan_instantiate_template(
    const common_plan_template & source,
    const std::string & instance_id,
    const std::string & session_id,
    common_plan_state & plan,
    std::string & error,
    const common_plan_template_options & options = {});
