#pragma once

#include "agent-flydelta-dataset-repair-host.h"

#include "agent/adaptation/flydelta/oracles/contracts.h"
#include "tools/agent/cli/agent-cli-generation.h"

#include <nlohmann/json.hpp>

#include <string>

namespace agent_flydelta_repair_smoke_support {

using json = nlohmann::ordered_json;

struct host_tool_verdict {
    bool known = false;
    bool passed = false;
    std::string reason;
};

bool parse_model_tool_call(
        const common_agent_generation_result & generation,
        json & parsed);

bool verify_model_tool_contract(
        const common_agent_generation_result & generation,
        const std::string & expected_tool,
        const json & expected_arguments,
        agent_flydelta_dataset_repair_host & host,
        host_tool_verdict & verdict,
        std::string & error);

common_flydelta_counterfactual_outcome classify_host_verdicts(
        const host_tool_verdict & baseline,
        const host_tool_verdict & candidate);

std::string output_preview(const common_agent_generation_result & result);

} // namespace agent_flydelta_repair_smoke_support
