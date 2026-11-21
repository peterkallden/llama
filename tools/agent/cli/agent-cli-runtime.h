#pragma once

#include "tools/agent/cli/agent-cli-options.h"

#include "agent/agent-inference.h"
#include "agent/agent-runtime.h"
#include "agent/learning/memory-learning.h"

#include <memory>
#include <vector>

struct common_agent_generation_config;

std::unique_ptr<common_planner> make_llama_cli_planner(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const std::vector<common_chat_tool> & tools);

std::unique_ptr<common_action_executor> make_llama_cli_action_executor(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config);

std::unique_ptr<common_reflection_engine> make_llama_cli_reflection_engine(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const std::vector<common_chat_tool> & tools);

// Compatibility overload for callers that do not have a tool view. Runtime
// assembly should use the overload above so reflection can render relevant
// compact contracts.
std::unique_ptr<common_reflection_engine> make_llama_cli_reflection_engine(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config);

std::unique_ptr<common_memory_candidate_extractor> make_llama_cli_memory_candidate_extractor(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config);
