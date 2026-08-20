#pragma once

#include "agent/agent-inference.h"
#include "agent/agent-prepared-generation.h"
#include "chat.h"
#include "llama.h"

#include <string>
#include <vector>

bool generate_chat_turn_result(
    llama_model * model,
    const common_chat_templates * chat_templates,
    const std::vector<common_chat_msg> & messages,
    const std::vector<common_chat_tool> & tools,
    common_chat_tool_choice tool_choice,
    const common_agent_generation_options & options,
    common_agent_generation_result & result,
    common_chat_params * chat_params = nullptr,
    const std::string & json_schema = {});

bool generate_chat_turn(
    llama_model * model,
    const common_chat_templates * chat_templates,
    const std::vector<common_chat_msg> & messages,
    const std::vector<common_chat_tool> & tools,
    common_chat_tool_choice tool_choice,
    const common_agent_generation_options & options,
    std::string & output,
    common_chat_params & chat_params,
    int & n_decode,
    const std::string & json_schema = {});
