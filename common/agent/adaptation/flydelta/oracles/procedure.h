#pragma once

#include "agent/adaptation/flydelta/oracles/workflow.h"

#include <string>

// A procedure/blueprint contract is a host-owned identity and applicability
// wrapper around an existing workflow contract. The model supplies only the
// canonical execution view; procedure and blueprint identity are attached by
// the host before this Oracle runs.
struct common_flydelta_procedure_contract {
    int schema_version = 1;
    std::string procedure_ref;
    std::string procedure_revision;
    std::string blueprint_ref;
    std::string blueprint_revision;
    common_flydelta_workflow_contract workflow;
    std::string applicability_fingerprint;
};

common_flydelta_oracle_evaluator common_flydelta_make_procedure_oracle(
        common_flydelta_procedure_contract contract);

