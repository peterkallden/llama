#pragma once

#include "agent/agent-inference.h"
#include "chat.h"
#include "llama.h"

#include <string>
#include <vector>

struct common_agent_prepared_generation {
    std::string prompt;
    common_grammar grammar;
    bool grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::string generation_prompt;
    std::string parser_generation_prompt;
    common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string parser;
    bool parse_tool_calls = false;
    bool ignore_eos = false;
    bool suppress_eog = false;
    bool stream = false;
};

bool prepare_chat_generation(
    const common_chat_templates * chat_templates,
    const common_agent_generation_request & request,
    common_agent_prepared_generation & prepared,
    common_chat_params * chat_params = nullptr);

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
