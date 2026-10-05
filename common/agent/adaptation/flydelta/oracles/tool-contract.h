#pragma once

#include "agent/adaptation/flydelta/oracles/contracts.h"
#include "agent/tooling/contracts/tool-runtime-contract.h"
#include "agent/tooling/schema/tool-output-codec.h"

#include <functional>
#include <string>

// This is the effective model-facing contract, not the raw provider schema.
// Providers construct it from the same projected descriptor that is sent to
// the model. Host policy fields are included only when they affect whether a
// call can be accepted; secrets and executable bindings never enter here.
struct common_flydelta_model_tool_contract {
    std::string provider_ref;
    std::string exposed_tool_name;
    std::string contract_ref;
    std::string contract_revision;
    std::string contract_fingerprint;
    std::string model_input_schema_json;
    std::string host_input_schema_json;

    bool read_only = true;
    bool requires_confirmation = false;
    bool uses_network = false;
    bool host_allows_network = true;
    bool confirmation_satisfied = true;
};

bool common_flydelta_parse_model_tool_text(
        common_agent_tool_output_format format,
        const std::string & observed,
        common_agent_tool_call & call,
        std::string & error);

// Validates an already parsed native call. This is the native adapter seam;
// it shares all argument and policy checks with textual model output.
bool common_flydelta_validate_model_tool_call(
        const common_flydelta_model_tool_contract & contract,
        const common_agent_tool_call & call,
        common_flydelta_oracle_result & result,
        std::string & error);

// Parses JSONL/compact DSL and then applies the same validator as native
// calls. Parse failures that are attributable to the model-facing protocol
// become a known contract violation; malformed host schemas remain UNKNOWN.
bool common_flydelta_validate_model_tool_text(
        const common_flydelta_model_tool_contract & contract,
        common_agent_tool_output_format format,
        const std::string & observed,
        common_flydelta_oracle_result & result,
        std::string & error);

// Host assembly uses this closure to register one immutable effective tool
// descriptor in the existing Oracle registry. The registry remains the only
// dispatch path; this factory does not create a store or runtime evaluator.
common_flydelta_oracle_evaluator common_flydelta_make_tool_contract_oracle(
        common_flydelta_model_tool_contract contract);
