#pragma once

#include "agent/agent-contract.h"
#include "agent/agent-runtime.h"

#include <nlohmann/json.hpp>

#include <string>

nlohmann::ordered_json common_agent_runtime_user_correction_to_json(
    const std::string & source_turn_id,
    const std::string & statement);

nlohmann::ordered_json common_agent_runtime_failure_observation_to_json(
    const common_agent_failure & failure);

nlohmann::ordered_json common_agent_runtime_reflection_learning_hint_to_json(
    const common_reflection_learning_hint & hint);

std::string common_agent_runtime_normalize_reasoning_observation_json(
    const std::string & reasoning_text);
