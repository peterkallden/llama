#include "agent-prepared-generation.h"

#include "json-schema-to-grammar.h"

#include <nlohmann/json.hpp>

bool common_agent_prepare_chat_generation(
        const common_chat_templates * chat_templates,
        const common_agent_generation_request & request,
        common_agent_prepared_generation & prepared,
        common_chat_params * chat_params) {
    common_chat_templates_inputs chat_inputs;
    chat_inputs.messages = request.messages;
    chat_inputs.tools = request.tools;
    chat_inputs.tool_choice = request.tool_choice;
    chat_inputs.parallel_tool_calls = false;
    chat_inputs.add_generation_prompt = true;

    const common_chat_params generated_chat_params = common_chat_templates_apply(chat_templates, chat_inputs);
    if (chat_params != nullptr) {
        *chat_params = generated_chat_params;
    }

    prepared = {};
    prepared.prompt = generated_chat_params.prompt;
    prepared.grammar_lazy = generated_chat_params.grammar_lazy;
    prepared.grammar_triggers = generated_chat_params.grammar_triggers;
    prepared.parser_generation_prompt = generated_chat_params.generation_prompt;
    prepared.chat_format = generated_chat_params.format;
    prepared.parser = generated_chat_params.parser;
    prepared.parse_tool_calls = !request.tools.empty();

    if (request.json_schema.empty()) {
        prepared.grammar = common_grammar{ COMMON_GRAMMAR_TYPE_TOOL_CALLS, generated_chat_params.grammar };
        prepared.generation_prompt = generated_chat_params.generation_prompt;
        return true;
    }

    prepared.grammar = common_grammar{
        COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
        json_schema_to_grammar(nlohmann::ordered_json::parse(request.json_schema)),
    };
    prepared.ignore_eos = true;
    prepared.suppress_eog = true;
    prepared.stream = true;
    return true;
}
