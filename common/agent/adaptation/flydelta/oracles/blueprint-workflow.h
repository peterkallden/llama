#pragma once

#include "agent/adaptation/flydelta/oracles/workflow-proposal.h"
#include "agent/learning/blueprint-selector.h"
#include "plan/plan-types.h"

#include <functional>
#include <string>

// Host-owned input for the first blueprint/state-graph adapter. The host
// supplies an already resolved dataset and the desired terminal operation;
// the adapter only creates a bounded graph and delegates search to A*.
struct common_flydelta_dataset_blueprint_request {
    int schema_version = 1;
    std::string proposal_id;
    std::string blueprint_ref;
    std::string blueprint_revision;
    std::string graph_revision;
    std::string workflow_ref;
    std::string workflow_revision;
    std::string dataset_ref;
    std::string dataset_name;
    // Join workflows bind both inputs by host-resolved identity; a model-supplied
    // path or unregistered dataset name is never promoted to a dataset ref.
    std::string second_dataset_ref;
    std::string second_dataset_name;
    bool schema_known = false;
    bool second_schema_known = false;
    std::string terminal_tool;
    std::string terminal_arguments_json;
    size_t max_expansions = 16;
    size_t max_path_length = 4;
};

// Produces a proof-carrying workflow proposal for the registered
// dataset-inspect-summarize blueprint. No tool is executed and no Oracle or
// lifecycle result is created here.
bool common_flydelta_propose_dataset_blueprint_workflow(
        const common_flydelta_dataset_blueprint_request & request,
        common_flydelta_workflow_proposal & result,
        std::string & error);

// Converts a verified host workflow proposal into the ordinary plan shape
// consumed by the existing runtime. This is a pure materialization step: it
// does not execute tools, write lifecycle state, or persist a second plan.
bool common_flydelta_materialize_workflow_proposal_plan(
        const common_flydelta_workflow_proposal & proposal,
        const common_plan_state & blueprint_instance,
        common_plan_state & materialized,
        std::string & error);

using common_flydelta_dataset_blueprint_request_provider = std::function<bool(
        const common_agent_request & request,
        const common_plan_state & blueprint_instance,
        common_flydelta_dataset_blueprint_request & dataset_request,
        std::string & error)>;

using common_flydelta_workflow_proposal_verifier = std::function<bool(
        const common_flydelta_workflow_proposal & proposal,
        std::string & error)>;

// Creates the generic selector callback used by the existing blueprint
// selection seam. The host supplies the current task-specific dataset state;
// without it the callback is not applicable and normal blueprint planning is
// preserved.
common_blueprint_instance_materializer
common_flydelta_make_dataset_blueprint_plan_materializer(
        common_flydelta_dataset_blueprint_request_provider request_provider,
        common_flydelta_workflow_proposal_verifier verifier);
