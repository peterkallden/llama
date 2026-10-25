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

// Binding evidence must be structured, successful host output. Confidence
// alone does not establish success; explicit failure envelopes are rejected.
bool common_plan_workflow_observation_is_usable(const common_plan_observation & observation);

// Resolve only completed transitions in the active workflow segment. Choice
// evidence must be tied to that step's successful tool observation and alias.
const common_plan_observation * common_plan_workflow_observation(
    const common_plan_state & plan,
    const std::string & source_transition_id,
    bool require_dataset = false,
    const std::string & tool_name = {});

// Required observation pointers establish availability, not evidential truth.
bool common_plan_workflow_choice_available(const common_plan_observation & observation,
    const common_plan_workflow_transition::choice & choice);
