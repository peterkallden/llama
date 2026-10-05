#pragma once

#include "agent/adaptation/flydelta/oracles/tool-contract.h"

#include <functional>
#include <string>
#include <vector>

// Host-facing OpenAPI projection. The common Oracle target deliberately does
// not depend on agent_openapi_catalog; the OpenAPI host adapter materializes
// this descriptor from the same effective model-facing operation definition
// that it gives to the provider/model.
struct common_flydelta_openapi_operation_contract {
    common_flydelta_model_tool_contract tool;
    std::string provider_id;
    std::string operation_id;
    std::string method;
    std::string path;
    std::vector<std::string> path_parameters;
    std::vector<std::string> query_parameters;
    std::vector<std::string> host_required_parameters;
    bool read_only = true;
    bool requires_confirmation = false;
};

common_flydelta_oracle_evaluator common_flydelta_make_openapi_operation_oracle(
        common_flydelta_openapi_operation_contract contract);

