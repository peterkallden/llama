#pragma once

#include "agent/agent-runtime.h"
#include "agent/memory-learning.h"
#include "chat.h"
#include "common/cli-config.h"
#include "llama.h"

#include <memory>
#include <vector>

std::unique_ptr<common_planner> make_llama_cli_planner(
    llama_model * model,
    const common_chat_templates * templates,
    const args & options,
    const std::vector<common_chat_tool> & tools);

std::unique_ptr<common_action_executor> make_llama_cli_action_executor(
    llama_model * model,
    const common_chat_templates * templates,
    const args & options);

std::unique_ptr<common_reflection_engine> make_llama_cli_reflection_engine(
    llama_model * model,
    const common_chat_templates * templates,
    const args & options);

std::unique_ptr<common_memory_candidate_extractor> make_llama_cli_memory_candidate_extractor(
    llama_model * model,
    const common_chat_templates * templates,
    const args & options);
