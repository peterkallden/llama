#pragma once

#include "agent/adaptation/flydelta/oracles/flydelta-oracle-contracts.h"

// Deterministic semantic oracle for the dataset-operation family. It parses
// model-shaped calls into the shared SemanticDecision IR and never executes a
// tool. Host execution belongs in the host_supported evaluator seam.
bool common_flydelta_dataset_operation_oracle(
        const common_flydelta_oracle_request & request,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error);
