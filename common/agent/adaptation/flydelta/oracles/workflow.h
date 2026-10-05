#pragma once

#include "agent/adaptation/flydelta/oracles/contracts.h"
#include "agent/tool-workflow-index.h"

#include <string>
#include <vector>

// A host-owned workflow contract. The model never supplies this structure;
// it supplies only a plan which the host canonicalizes before evaluation.
struct common_flydelta_workflow_contract {
    int schema_version = 1;
    std::string workflow_ref;
    std::string workflow_revision;
    std::vector<common_tool_workflow> workflows;
    std::vector<std::string> selected_workflow_ids;
    std::string applicability_fingerprint;
};

// The observed value is a host-canonical plan/execution view, not raw model
// text. Its bounded shape is:
// {"stage":"plan|execution","steps":[
//   {"tool":"...","arguments":"..."}
// ]}
common_flydelta_oracle_evaluator common_flydelta_make_workflow_oracle(
        common_flydelta_workflow_contract contract);

