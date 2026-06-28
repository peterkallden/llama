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
        "{%- endfor -%}"
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

void test_server_task_params_from_prepared_generation() {
    auto request = make_base_request();

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

    const auto params = make_server_task_params_from_prepared_generation(request, prepared, logit_bias_eog);
    assert(params.stream);
    assert(params.n_predict == 77);
    assert(params.t_max_prompt_ms == 111);
    assert(params.t_max_predict_ms == 222);
    assert(params.sampling.grammar.type == COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT);
    assert(params.sampling.grammar.grammar == "root ::= object");
    assert(params.sampling.grammar_lazy);
    assert(params.sampling.generation_prompt.empty());
    assert(params.sampling.ignore_eos);
    assert(params.sampling.logit_bias.size() == 2);
    assert(params.chat_parser_params.generation_prompt == "<|assistant|>");
    assert(params.chat_parser_params.parse_tool_calls);
}

} // namespace

int main() {
    test_prepare_tool_generation();
    test_prepare_json_schema_generation();
    test_server_task_params_from_prepared_generation();
    return 0;
}
