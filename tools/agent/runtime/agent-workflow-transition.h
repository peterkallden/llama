#pragma once

#include "agent-route-compiler.h"
#include "agent/contracts/agent-request.h"

#include <functional>

// The selector receives only host-recorded candidates from one observation.
// It returns an exact candidate id; the transition runtime validates it again
// before placing it into the next host-owned tool call.
using common_agent_workflow_choice_selector = std::function<bool(
        const common_plan_observation &,
        const common_plan_workflow_transition &,
        std::string &,
        std::string &)>;

// Returns a continuation provider for a selected declarative workflow graph.
// An empty provider means the workflow retains its existing planner behaviour.
common_agent_workflow_continuation_provider make_agent_workflow_graph_continuation(
        common_agent_route_candidate route,
        common_agent_workflow_choice_selector select_choice = {});

// Finds the first host-approved transition on the cheapest bounded graph path.
// It is used at initial materialization and is deterministic for equal costs.
const common_plan_workflow_transition * common_agent_workflow_next_transition(
        const common_plan_workflow_definition & definition,
        const std::string & state);
