#pragma once

#include "../common/cli-config.h"

#include "agent/agent-inference.h"
#include "agent/agent-runtime.h"
#include "agent/memory-learning.h"

#include <memory>
#include <vector>

std::unique_ptr<common_planner> make_llama_cli_planner(
    common_agent_inference & inference,
    const args & options,
    const std::vector<common_chat_tool> & tools);

std::unique_ptr<common_action_executor> make_llama_cli_action_executor(
    common_agent_inference & inference,
    const args & options);

std::unique_ptr<common_reflection_engine> make_llama_cli_reflection_engine(
    common_agent_inference & inference,
    const args & options);

std::unique_ptr<common_memory_candidate_extractor> make_llama_cli_memory_candidate_extractor(
    common_agent_inference & inference,
    const args & options);
