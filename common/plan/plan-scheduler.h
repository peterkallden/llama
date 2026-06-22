#pragma once

#include "plan/plan-types.h"

#include <string>
#include <vector>

// A deterministic view of which pending steps may progress.  It deliberately
// contains no model decisions: callers choose how to execute a ready step.
enum class common_plan_schedule_state { runnable, blocked, complete, inactive };
struct common_plan_schedule_result {
    std::vector<std::string> ready_step_ids;
    std::vector<std::string> blocked_step_ids;
    common_plan_schedule_state state = common_plan_schedule_state::inactive;
    bool blocked = false;
    bool terminal = false;
    bool complete = false;
};

common_plan_schedule_result common_plan_schedule(const common_plan_state & plan);
