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
    common_peg_arena parser;
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

// Count the fully rendered prompt with the backend tokenizer before admission.
// Never trim mandatory messages or schemas to make an oversized request fit.
inline bool common_agent_generation_context_fits(size_t prompt_tokens, size_t context_tokens,
        int requested_output_tokens, std::string & error) {
    error.clear();
    if (context_tokens == 0) return true;
    const size_t reserve = requested_output_tokens > 0 ? static_cast<size_t>(requested_output_tokens) : 1;
    if (prompt_tokens >= context_tokens || reserve > context_tokens - prompt_tokens) {
        error = "generation context budget exceeded: prompt=" + std::to_string(prompt_tokens) +
            " reserved_output=" + std::to_string(reserve) + " context=" + std::to_string(context_tokens);
        return false;
    }
    return true;
}
