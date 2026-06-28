#include "memory-cli-chat.h"

#include "common.h"
#include "json-schema-to-grammar.h"
#include "sampling.h"

#include <chrono>
#include <cmath>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using steady_clock = std::chrono::steady_clock;

bool budget_exceeded(
        const steady_clock::time_point & started_at,
        const std::optional<int64_t> & budget_ms) {
    if (!budget_ms || *budget_ms < 0) {
        return false;
    }
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        steady_clock::now() - started_at).count();
    return elapsed_ms > *budget_ms;
}

} // namespace

bool generate_chat_turn_result(
        llama_model * model,
        const common_chat_templates * chat_templates,
        const std::vector<common_chat_msg> & messages,
        const std::vector<common_chat_tool> & tools,
        common_chat_tool_choice tool_choice,
        const common_agent_generation_options & options,
        common_agent_generation_result & result,
        const std::string & json_schema) {
    result = {};

    const auto prompt_started_at = steady_clock::now();
    common_chat_templates_inputs chat_inputs;
    chat_inputs.messages = messages;
    chat_inputs.tools = tools;
    chat_inputs.tool_choice = tool_choice;
    chat_inputs.parallel_tool_calls = false;
    chat_inputs.add_generation_prompt = true;
    result.chat_params = common_chat_templates_apply(chat_templates, chat_inputs);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, result.chat_params.prompt.c_str(), result.chat_params.prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        result.error_message = "failed to tokenize chat prompt";
        fprintf(stderr, "%s\n", result.error_message.c_str());
        return false;
    }
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, result.chat_params.prompt.c_str(), result.chat_params.prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        result.error_message = "failed to tokenize chat prompt";
        fprintf(stderr, "%s\n", result.error_message.c_str());
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + options.n_predict;
    ctx_params.n_batch = n_prompt;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        result.error_message = "failed to create llama context";
        fprintf(stderr, "%s\n", result.error_message.c_str());
        return false;
    }

    common_params_sampling sampling;
    sampling.temp = 0.0f;
    sampling.grammar = json_schema.empty()
        ? common_grammar{ COMMON_GRAMMAR_TYPE_TOOL_CALLS, result.chat_params.grammar }
        : common_grammar{ COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(nlohmann::ordered_json::parse(json_schema)) };
    sampling.grammar_lazy = result.chat_params.grammar_lazy;
    sampling.grammar_triggers = result.chat_params.grammar_triggers;
    sampling.generation_prompt = json_schema.empty() ? result.chat_params.generation_prompt : std::string{};
    if (!json_schema.empty()) {
        sampling.ignore_eos = true;
        for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token) {
            if (llama_vocab_is_eog(vocab, token)) {
                sampling.logit_bias.push_back({ token, -INFINITY });
            }
        }
    }
    common_sampler_ptr sampler(common_sampler_init(model, sampling));

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    result.content.clear();
    result.decoded_tokens = 0;
    common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::limit;
    bool completed_json_schema = false;
    std::optional<steady_clock::time_point> predict_started_at;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + options.n_predict; ) {
        if (n_pos == 0 && budget_exceeded(prompt_started_at, options.t_max_prompt_ms)) {
            result.stop_reason = common_agent_generation_stop_reason::limit;
            result.error_message = "prompt time budget exceeded";
            llama_free(ctx);
            return false;
        }
        if (llama_decode(ctx, batch)) {
            result.error_message = "failed to decode";
            fprintf(stderr, "%s\n", result.error_message.c_str());
            llama_free(ctx);
            return false;
        }
        n_pos += batch.n_tokens;
        if (n_pos == n_prompt && budget_exceeded(prompt_started_at, options.t_max_prompt_ms)) {
            result.stop_reason = common_agent_generation_stop_reason::limit;
            result.error_message = "prompt time budget exceeded";
            llama_free(ctx);
            return false;
        }
        if (n_pos == n_prompt && !predict_started_at) {
            predict_started_at = steady_clock::now();
        }
        if (predict_started_at && budget_exceeded(*predict_started_at, options.t_max_predict_ms)) {
            stop_reason = common_agent_generation_stop_reason::limit;
            break;
        }
        llama_token token = common_sampler_sample(sampler.get(), ctx, -1, true);
        common_sampler_accept(sampler.get(), token, true);
        if (llama_vocab_is_eog(vocab, token)) {
            stop_reason = common_agent_generation_stop_reason::eos;
            break;
        }
        const std::string piece = common_token_to_piece(vocab, token, true);
        result.content += piece;
        batch = llama_batch_get_one(&token, 1);
        result.decoded_tokens++;
        if (!json_schema.empty()) {
            const auto parsed = nlohmann::ordered_json::parse(result.content, nullptr, false);
            if (!parsed.is_discarded()) {
                completed_json_schema = true;
                stop_reason = common_agent_generation_stop_reason::json_schema;
                break;
            }
        }
    }

    llama_free(ctx);
    if (!json_schema.empty() && !completed_json_schema) {
        result.status = common_agent_generation_status::errored;
        result.stop_reason = stop_reason;
        result.error_message = "generation ended before producing valid JSON for the requested schema";
        return false;
    }

    result.status = common_agent_generation_status::completed;
    result.stop_reason = stop_reason;
    return true;
}

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
        const std::string & json_schema) {
    common_agent_generation_result result;
    if (!generate_chat_turn_result(
            model,
            chat_templates,
            messages,
            tools,
            tool_choice,
            options,
            result,
            json_schema)) {
        output = std::move(result.content);
        chat_params = std::move(result.chat_params);
        n_decode = result.decoded_tokens;
        return false;
    }
    output = std::move(result.content);
    chat_params = std::move(result.chat_params);
    n_decode = result.decoded_tokens;
    return true;
}
