#pragma once

#include "agent/adaptation/flydelta/oracles/workflow-proposal.h"

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
    bool schema_known = false;
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
