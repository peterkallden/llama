#include "agent/agent-inference.h"
#include "agent/agent-prepared-generation.h"
#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent-server-generation.h"
#include "chat.h"
#include "../src/llama-adapter.h"
#include "../src/llama-graph.h"

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

class teacher_forced_batch_test_inference final : public common_agent_inference {
public:
    bool generate(
            const common_agent_generation_request &,
            common_agent_generation_result &) override {
        return false;
    }

    bool score_teacher_forced_choice(
            const common_agent_teacher_forced_choice_request & request,
            common_agent_teacher_forced_choice_result & result) override {
        result = {};
        result.sequence_id = request.sequence_id;
        result.available = true;
        result.positive_total_logprob = static_cast<float>(request.positive_choice.size());
        result.negative_total_logprob = static_cast<float>(request.negative_choice.size());
        result.positive_token_count = request.positive_choice.size();
        result.negative_token_count = request.negative_choice.size();
        return true;
    }
};

class scalar_batch_fallback_test_inference final : public common_agent_inference {
public:
    bool generate(
            const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
        result = {};
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::eos;
        result.content = request.trace_id.value_or("no-trace");
        result.decoded_tokens = static_cast<int>(request.messages.size());
        return true;
    }
};

void test_generation_batch_scalar_fallback() {
    scalar_batch_fallback_test_inference inference;
    auto first = make_base_request();
    first.trace_id = "batch:first";
    auto second = make_base_request();
    second.trace_id = "batch:second";

    std::vector<common_agent_generation_result> results;
    assert(inference.generate_batch({first, second}, results));
    assert(results.size() == 2);
    assert(results[0].content == "batch:first");
    assert(results[1].content == "batch:second");
    assert(results[0].status == common_agent_generation_status::completed);
    assert(results[1].status == common_agent_generation_status::completed);
}

void test_teacher_forced_batch_fallback() {
    teacher_forced_batch_test_inference inference;
    common_agent_teacher_forced_choice_batch_request request;
    request.choices.resize(2);
    request.choices[0].sequence_id = "arm:alpha-01";
    request.choices[1].sequence_id = "arm:alpha-02";
    request.choices[0].positive_choice = "inspect";
    request.choices[0].negative_choice = "describe";
    request.choices[1].positive_choice = "aggregate";
    request.choices[1].negative_choice = "query";

    common_agent_teacher_forced_choice_batch_result result;
    assert(inference.score_teacher_forced_choice_batch(request, result));
    assert(result.error_message.empty());
    assert(result.choices.size() == 2);
    assert(result.choices[0].available && result.choices[1].available);
    assert(result.choices[0].sequence_id == "arm:alpha-01");
    assert(result.choices[1].sequence_id == "arm:alpha-02");
    assert(result.choices[0].positive_token_count == 7);
    assert(result.choices[1].negative_token_count == 5);
    request.choices[1].sequence_id = request.choices[0].sequence_id;
    result = {};
    assert(!inference.score_teacher_forced_choice_batch(request, result));
    assert(result.error_message.find("duplicate sequence identity") != std::string::npos);
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

void test_flydelta_activation_is_per_request_and_server_accepts_active() {
    auto activation = std::make_shared<common_flydelta_activation_result>();
    activation->gate.apply = true;
    activation->gate.scale = 0.1f;
    activation->overlay.enabled = true;
    activation->overlay.artifact_id = "flydelta://artifact/1";
    activation->overlay.n_embd = 2;
    activation->overlay.il_start = 1;
    activation->overlay.il_end = 2;
    activation->overlay.data.assign(4, 0.0f);

    auto request = make_base_request();
    request.flydelta_activation = activation;
    auto generation = common_agent_make_generation_request(
        request.purpose,
        request.trace_id,
        request.scope,
        request.messages,
        request.options,
        request.json_schema,
        request.tools,
        request.tool_choice,
        request.flydelta_activation);
    assert(generation.flydelta_activation == activation);

    std::string error;
    assert(common_flydelta_activation_result_validate(
        *generation.flydelta_activation, 2, 3, 1024, error));
    assert(server_context_agent_generation_supports_flydelta(generation, error));

    auto no_op = make_base_request();
    auto no_op_generation = common_agent_make_generation_request(
        no_op.purpose,
        no_op.trace_id,
        no_op.scope,
        no_op.messages,
        no_op.options);
    assert(!no_op_generation.flydelta_activation);
    assert(server_context_agent_generation_supports_flydelta(no_op_generation, error));
}

void test_prepare_plain_chat_has_no_tool_grammar() {
    auto templates = make_templates();
    const auto request = make_base_request();

    common_agent_prepared_generation prepared;
    assert(common_agent_prepare_chat_generation(templates.get(), request, prepared));
    assert(prepared.grammar.empty());
    assert(!prepared.grammar_lazy);
    assert(prepared.grammar_triggers.empty());
    assert(!prepared.parse_tool_calls);
    assert(!prepared.stream);
}

void test_server_task_params_from_prepared_generation() {
    auto request = make_base_request();
    auto activation = std::make_shared<common_flydelta_activation_result>();
    activation->gate.apply = true;
    activation->gate.scale = 0.1f;
    activation->overlay.enabled = true;
    activation->overlay.artifact_id = "flydelta://artifact/task";
    activation->overlay.n_embd = 2;
    activation->overlay.il_start = 1;
    activation->overlay.il_end = 2;
    activation->overlay.scale = 0.5f;
    activation->overlay.data.assign(4, 0.25f);
    request.flydelta_activation = activation;
    common_params params_base;
    params_base.n_keep = 9;
    params_base.n_cache_reuse = 17;
    params_base.cache_prompt = true;
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
    assert(params.cvec);
    assert(params.cvec->identity == "flydelta://artifact/task");
    assert(params.cvec->n_embd == 2);
    assert(params.cvec->il_start == 1);
    assert(params.cvec->il_end == 2);
    assert(params.cvec->data.size() == 4);
    assert(std::fabs(params.cvec->data[0] - 0.125f) < 1e-6f);
    assert(!params.cache_prompt);
    assert(params.n_cache_reuse == 0);
    const auto serialized_params = params.to_json();
    assert(!serialized_params.contains("cvec"));

    // Distinct per-sequence overlays must never compare as the same cvec.
    // server_context uses this identity boundary to clear prompt/KV state
    // before applying a different request-scoped overlay.
    auto second_activation = std::make_shared<common_flydelta_activation_result>(*activation);
    second_activation->overlay.artifact_id = "flydelta://artifact/task-2";
    second_activation->overlay.data[0] = 0.5f;
    request.flydelta_activation = second_activation;
    const auto second_params = make_server_task_params_from_prepared_generation(
        params_base, request, prepared, logit_bias_eog);
    assert(second_params.cvec);
    assert(!server_task_cvec_equal(params.cvec, second_params.cvec));
    assert(!params.cache_prompt && !second_params.cache_prompt);

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

void test_server_task_cvec_contract() {
    auto first = std::make_shared<server_task_cvec>();
    first->identity = "flydelta://artifact/slot";
    first->content_hash = "sha256:one";
    first->n_embd = 2;
    first->il_start = 1;
    first->il_end = 2;
    first->data.assign(4, 0.25f);

    auto same = std::make_shared<server_task_cvec>(*first);
    auto different = std::make_shared<server_task_cvec>(*first);
    different->content_hash = "sha256:two";
    different->data[0] = 0.5f;

    assert(server_task_cvec_equal(first, same));
    assert(!server_task_cvec_equal(first, different));

    std::string error;
    assert(server_task_cvec_validate(*first, 2, 3, 1024, error));

    auto wrong_dimensions = *first;
    wrong_dimensions.n_embd = 4;
    assert(!server_task_cvec_validate(wrong_dimensions, 2, 3, 1024, error));

    auto over_bound = *first;
    over_bound.data.assign(1025, 0.0f);
    assert(!server_task_cvec_validate(over_bound, 2, 3, 1024, error));
}

void test_cvec_batch_graph_identity() {
    llama_adapter_cvec_batch_ref first;
    llama_adapter_cvec_batch_ref second;

    llm_graph_params params{};
    params.cvec_batch = &first;

    auto same = params;
    assert(params.allow_reuse(same));

    same.cvec_batch = &second;
    assert(!params.allow_reuse(same));
}

void test_server_task_cvec_batch_view() {
    auto first = std::make_shared<server_task_cvec>();
    first->identity = "flydelta://artifact/slot-a";
    first->n_embd = 2;
    first->il_start = 1;
    first->il_end = 2;
    first->data = {0.25f, 0.5f, 0.75f, 1.0f};

    auto second = std::make_shared<server_task_cvec>(*first);
    second->identity = "flydelta://artifact/slot-b";
    second->data[0] = -0.25f;

    server_task_cvec_batch batch;
    std::string error;
    assert(batch.add(3, first, error));
    assert(batch.add(7, second, error));
    assert(batch.size() == 2);

    std::vector<llama_seq_id> seq_ids;
    std::vector<const float *> data;
    size_t data_len = 0;
    int32_t n_embd = 0;
    int32_t il_start = 0;
    int32_t il_end = 0;
    assert(batch.materialize(seq_ids, data, data_len, n_embd, il_start, il_end, error));
    assert(seq_ids == std::vector<llama_seq_id>({3, 7}));
    assert(data.size() == 2);
    assert(data[0] == first->data.data());
    assert(data[1] == second->data.data());
    assert(data_len == first->data.size());
    assert(n_embd == 2 && il_start == 1 && il_end == 2);

    assert(!batch.add(3, second, error));

    auto incompatible = std::make_shared<server_task_cvec>(*first);
    incompatible->il_end = 1;
    assert(!batch.add(9, incompatible, error));

    server_task_cvec_batch empty;
    assert(!empty.materialize(seq_ids, data, data_len, n_embd, il_start, il_end, error));
}

} // namespace

int main() {
    test_generation_batch_scalar_fallback();
    test_teacher_forced_batch_fallback();
    test_prepare_tool_generation();
    test_prepare_json_schema_generation();
    test_flydelta_activation_is_per_request_and_server_accepts_active();
    test_prepare_plain_chat_has_no_tool_grammar();
    test_server_task_params_from_prepared_generation();
    test_server_task_cvec_contract();
    test_cvec_batch_graph_identity();
    test_server_task_cvec_batch_view();
    return 0;
}
