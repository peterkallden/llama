#pragma once

#include "agent/agent-inference.h"
#include "chat.h"

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

bool common_agent_prepare_chat_generation(
    const common_chat_templates * chat_templates,
    const common_agent_generation_request & request,
    common_agent_prepared_generation & prepared,
    common_chat_params * chat_params = nullptr);
