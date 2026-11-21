#pragma once

#include "agent-openapi-catalog.h"

#include "agent/adaptation/flydelta/oracles/contracts.h"
#include "agent/adaptation/flydelta/oracles/openapi-operation.h"

#include <string>

// Build the host-owned Oracle descriptor from the same effective OpenAPI
// operation that is exposed to the model. The common Oracle does not depend
// on the agent OpenAPI catalog; this file is the host adapter.
bool make_agent_openapi_flydelta_oracle_contract(
        const agent_openapi_catalog & catalog,
        const agent_openapi_operation & operation,
        common_flydelta_openapi_operation_contract & contract,
        std::string & error);

// Register one immutable evaluator per exposed OpenAPI operation in the
// existing in-memory Oracle registry. This does not create a store, execute
// HTTP, or alter tool routing.
bool register_agent_openapi_flydelta_oracles(
        const agent_openapi_catalog & catalog,
        common_flydelta_oracle_registry & registry,
        const std::string & oracle_revision,
        std::string & error);

