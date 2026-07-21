#pragma once

#include "agent/agent-contract.h"
#include "agent/agent-runtime.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>

nlohmann::ordered_json common_agent_runtime_user_correction_to_json(
    const std::string & source_turn_id,
    const std::string & statement);

std::string common_agent_runtime_user_correction_json(
    const std::string & source_turn_id,
    const std::string & statement);

nlohmann::ordered_json common_agent_runtime_failure_observation_to_json(
    const common_agent_failure & failure);

std::string common_agent_runtime_failure_observation_json(
    const common_agent_failure & failure);

nlohmann::ordered_json common_agent_runtime_reflection_learning_hint_to_json(
    const common_reflection_learning_hint & hint);

std::string common_agent_runtime_reflection_learning_hint_json(
    const common_reflection_learning_hint & hint);

nlohmann::ordered_json common_agent_runtime_reasoning_observation_to_json(
    const std::string & reasoning_text);

std::string common_agent_runtime_normalize_reasoning_observation_json(
    const std::string & reasoning_text);

bool common_agent_runtime_apply_safe_tool_defaults_to_json(
    const common_agent_request & request,
    const std::string & tool_name,
    const nlohmann::ordered_json & arguments,
    nlohmann::ordered_json & normalized_arguments,
    bool & changed,
    std::string & error);

bool common_agent_runtime_apply_safe_tool_defaults(
    const common_agent_request & request,
    common_agent_tool_call & call);
