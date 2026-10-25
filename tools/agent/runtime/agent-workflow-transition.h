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

// Generate only the remaining arguments of one declared transition. The seed
// contains host-owned templates and verified observation bindings.
using common_agent_workflow_argument_selector = std::function<bool(
        const common_plan_state &, const common_plan_observation &,
        const common_plan_workflow_transition &, const std::string &,
        std::string &, std::string &)>;

// Returns a continuation provider for a selected declarative workflow graph.
// An empty provider means the workflow retains its existing planner behaviour.
common_agent_workflow_continuation_provider make_agent_workflow_graph_continuation(
        common_agent_route_candidate route,
        common_agent_workflow_choice_selector select_choice = {},
        common_agent_workflow_argument_selector select_arguments = {});

// Finds the first host-approved transition on the cheapest bounded graph path.
// It is used at initial materialization and is deterministic for equal costs.
const common_plan_workflow_transition * common_agent_workflow_next_transition(
        const common_plan_workflow_definition & definition,
        const std::string & state,
        const std::vector<std::string> * allowed_tools = nullptr);

// Returns true when a workflow's static transition arguments fully satisfy the
// selected tool schema.  Returns false with an empty error when model-owned
// required arguments remain; invalid static workflow arguments set error.
bool common_agent_workflow_static_tool_arguments(
        const common_plan_workflow_transition & transition,
        const common_chat_tool & tool,
        std::string & normalized_arguments_json,
        std::string & error);

// Attach the retained blueprint phase's intent to its materialized operation.
// This is provenance; it never marks the phase complete or authorizes a tool.
void common_agent_workflow_apply_blueprint_phase(const common_plan_state &,
    const common_plan_workflow_transition &, common_plan_step &);
