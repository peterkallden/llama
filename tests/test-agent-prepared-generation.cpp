#include "agent/agent-inference.h"
#include "agent/agent-prepared-generation.h"
#include "agent-server-generation.h"
#include "chat.h"

#include <cassert>
#include <cmath>

namespace {

common_chat_templates_ptr make_templates() {
    const std::string mock_template =
        "{%- for message in messages -%}"
        "{{- '<|' + message.role + '|>' + message.content + '<|end|>' -}}"
        "{%- if message.tool_calls is defined and message.tool_calls -%}"
        "{{- '<tool_calls>' + message.tool_calls + '</tool_calls>' -}}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if tools -%}"
        "{{- 'Tools: ' + tools -}}"
        "{%- endif -%}"
        "{%- if add_generation_prompt -%}"
        "{{- '<|assistant|>' -}}"
        "{%- endif -%}";
    return common_chat_templates_ptr(common_chat_templates_init(nullptr, mock_template));
}

common_agent_generation_request make_base_request() {
    common_agent_generation_options options;
    options.n_predict = 77;
    options.t_max_prompt_ms = 111;
    options.t_max_predict_ms = 222;

    common_chat_msg user;
    user.role = "user";
    user.content = "Check status";

    return common_agent_make_generation_request(
        common_agent_generation_purpose::draft,
        std::string("trace-42"),
        std::nullopt,
        {user},
        options);
}

void test_prepare_tool_generation() {
    auto templates = make_templates();
    auto request = make_base_request();
    request.tools.push_back({
        "lookup",
        "Look up a record",
        R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})",
    });
    request.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_agent_prepared_generation prepared;
    common_chat_params chat_params;
    const bool ok = common_agent_prepare_chat_generation(templates.get(), request, prepared, &chat_params);
    assert(ok);
    assert(prepared.prompt.find("<|user|>Check status<|end|>") != std::string::npos);
    assert(prepared.prompt.find("<|assistant|>") != std::string::npos);
    assert(prepared.grammar.type == COMMON_GRAMMAR_TYPE_TOOL_CALLS);
    assert(!prepared.grammar.empty());
    assert(!prepared.generation_prompt.empty());
    assert(prepared.parser_generation_prompt == prepared.generation_prompt);
    assert(prepared.parse_tool_calls);
    assert(!prepared.ignore_eos);
    assert(!prepared.suppress_eog);
    assert(!prepared.stream);
    assert(chat_params.prompt == prepared.prompt);
}

void test_prepare_json_schema_generation() {
    auto templates = make_templates();
    auto request = make_base_request();
    request.json_schema = R"({"type":"object","additionalProperties":false,"required":["answer"],"properties":{"answer":{"type":"string"}}})";

    common_agent_prepared_generation prepared;
    const bool ok = common_agent_prepare_chat_generation(templates.get(), request, prepared);
    assert(ok);
    assert(prepared.grammar.type == COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT);
    assert(!prepared.grammar.empty());
    assert(prepared.generation_prompt.empty());
    assert(!prepared.parser_generation_prompt.empty());
    assert(prepared.ignore_eos);
    assert(prepared.suppress_eog);
    assert(prepared.stream);
    assert(!prepared.parse_tool_calls);
}

void test_prepare_plain_chat_has_no_tool_grammar() {
    auto templates = make_templates();
    const auto request = make_base_request();

    common_agent_prepared_generation prepared;
    assert(common_agent_prepare_chat_generation(templates.get(), request, prepared));
    assert(prepared.grammar.empty());
    assert(!prepared.parse_tool_calls);
    assert(!prepared.stream);
}

void test_server_task_params_from_prepared_generation() {
    auto request = make_base_request();
    common_params params_base;
    params_base.n_keep = 9;
    params_base.n_cache_reuse = 17;
    params_base.cache_prompt = false;
    params_base.antiprompt = {"<|stop|>"};
    params_base.sampling.temp = 0.6f;
    params_base.speculative.draft.n_max = 3;

    common_agent_prepared_generation prepared;
    prepared.prompt = "<|user|>Check status<|end|><|assistant|>";
    prepared.grammar = common_grammar{COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, "root ::= object"};
    prepared.grammar_lazy = true;
    prepared.generation_prompt.clear();
    prepared.parser_generation_prompt = "<|assistant|>";
    prepared.parse_tool_calls = true;
    prepared.ignore_eos = true;
    prepared.suppress_eog = true;
    prepared.stream = true;

    std::vector<llama_logit_bias> logit_bias_eog = {
        {1, -INFINITY},
        {2, -INFINITY},
    };

    const auto params = make_server_task_params_from_prepared_generation(params_base, request, prepared, logit_bias_eog);
    assert(params.stream);
    assert(!params.cache_prompt);
    assert(params.n_keep == 9);
    assert(params.n_cache_reuse == 17);
    assert(params.n_predict == 77);
    assert(params.t_max_prompt_ms == 111);
    assert(params.t_max_predict_ms == 222);
    assert(std::fabs(params.sampling.temp) < 1e-6f);
    assert(params.sampling.grammar.type == COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT);
    assert(params.sampling.grammar.grammar == "root ::= object");
    assert(params.sampling.grammar_lazy);
    assert(params.sampling.generation_prompt.empty());
    assert(params.sampling.ignore_eos);
    assert(params.sampling.logit_bias.size() == 2);
    assert(params.speculative.draft.n_max == 3);
    assert(params.antiprompt.size() == 1);
    assert(params.antiprompt[0] == "<|stop|>");
    assert(params.chat_parser_params.generation_prompt == "<|assistant|>");
    assert(params.chat_parser_params.parse_tool_calls);
}

} // namespace

int main() {
    test_prepare_tool_generation();
    test_prepare_json_schema_generation();
    test_prepare_plain_chat_has_no_tool_grammar();
    test_server_task_params_from_prepared_generation();
    return 0;
}
