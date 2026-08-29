#include "agent-cli-generation.h"

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
    common_chat_params * chat_params,
    const std::string & json_schema,
    const std::vector<llama_adapter_lora *> & adapters,
    const std::vector<float> & adapter_scales) {
    result = {};

    common_agent_generation_request request;
    request.messages = messages;
    request.tools = tools;
    request.tool_choice = tool_choice;
    request.options = options;
    request.json_schema = json_schema;

    const auto prompt_started_at = steady_clock::now();
    common_agent_prepared_generation prepared;
    if (!common_agent_prepare_chat_generation(chat_templates, request, prepared, chat_params)) {
        result.error_message = "failed to prepare chat generation";
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prepared.prompt.c_str(), prepared.prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        result.error_message = "failed to tokenize chat prompt";
        fprintf(stderr, "%s\n", result.error_message.c_str());
        return false;
    }
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prepared.prompt.c_str(), prepared.prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        result.error_message = "failed to tokenize chat prompt";
        fprintf(stderr, "%s\n", result.error_message.c_str());
        return false;
    }

    if (options.generation_trace) {
        fprintf(stderr,
            "agent generation trace: prompt_tokens=%d n_predict=%d threads=%d tools=%zu tool_choice=%d adapters=%zu\n",
            n_prompt,
            options.n_predict,
            options.n_threads,
            tools.size(),
            static_cast<int>(tool_choice),
            adapters.size());
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + options.n_predict;
    ctx_params.n_batch = n_prompt;
    ctx_params.n_threads = options.n_threads;
    ctx_params.n_threads_batch = options.n_threads;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        result.error_message = "failed to create llama context";
        fprintf(stderr, "%s\n", result.error_message.c_str());
        return false;
    }
    if (adapters.size() != adapter_scales.size()) {
        result.error_message = "adapter and scale counts differ";
        llama_free(ctx);
        return false;
    }
    if (!adapters.empty() && llama_set_adapters_lora(
            ctx, const_cast<llama_adapter_lora **>(adapters.data()),
            adapters.size(), const_cast<float *>(adapter_scales.data())) < 0) {
        result.error_message = "failed to apply model adapters";
        llama_free(ctx);
        return false;
    }

    common_params_sampling sampling;
    sampling.temp = 0.0f;
    sampling.grammar = prepared.grammar;
    sampling.grammar_lazy = prepared.grammar_lazy;
    sampling.grammar_triggers = prepared.grammar_triggers;
    sampling.generation_prompt = prepared.generation_prompt;
    if (prepared.ignore_eos) {
        sampling.ignore_eos = true;
        if (prepared.suppress_eog) {
            for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token) {
                if (llama_vocab_is_eog(vocab, token)) {
                    sampling.logit_bias.push_back({ token, -INFINITY });
                }
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

    bool logged_prompt_decode = false;
    int next_progress_log = 16;
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + options.n_predict; ) {
        if (n_pos == 0 && budget_exceeded(prompt_started_at, options.t_max_prompt_ms)) {
            result.stop_reason = common_agent_generation_stop_reason::limit;
            result.error_message = "prompt time budget exceeded";
            llama_free(ctx);
            return false;
        }
        const auto decode_started_at = steady_clock::now();
        if (llama_decode(ctx, batch)) {
            result.error_message = "failed to decode";
            fprintf(stderr, "%s\n", result.error_message.c_str());
            llama_free(ctx);
            return false;
        }
        if (!logged_prompt_decode && batch.n_tokens == n_prompt) {
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                steady_clock::now() - decode_started_at).count();
            fprintf(stderr,
                "generation: prompt decoded tokens=%d threads=%d elapsed_ms=%lld\n",
                n_prompt,
                options.n_threads,
                static_cast<long long>(elapsed_ms));
            logged_prompt_decode = true;
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
        if (options.generation_trace && result.decoded_tokens >= next_progress_log) {
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                steady_clock::now() - *predict_started_at).count();
            fprintf(stderr,
                "agent generation trace: decoded_tokens=%d elapsed_ms=%lld\n",
                result.decoded_tokens,
                static_cast<long long>(elapsed_ms));
            next_progress_log += 16;
        }
        if (prepared.stream) {
            const auto parsed = nlohmann::ordered_json::parse(result.content, nullptr, false);
            if (!parsed.is_discarded()) {
                completed_json_schema = true;
                stop_reason = common_agent_generation_stop_reason::json_schema;
                break;
            }
        }
    }

    llama_free(ctx);
    if (prepared.stream && !completed_json_schema) {
        result.status = common_agent_generation_status::errored;
        result.stop_reason = stop_reason;
        result.error_message = "generation ended before producing valid JSON for the requested schema";
        return false;
    }

    result.status = common_agent_generation_status::completed;
    result.stop_reason = stop_reason;
    if (options.generation_trace) {
        std::string preview = result.content.substr(0, 2048);
        for (char & ch : preview) {
            if (static_cast<unsigned char>(ch) < 0x20 || ch == 0x7f) ch = ' ';
        }
        fprintf(stderr,
            "agent generation trace: completed decoded_tokens=%d stop_reason=%s content_preview=%s\n",
            result.decoded_tokens,
            common_agent_generation_stop_reason_name(result.stop_reason),
            preview.c_str());
    }
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
            &chat_params,
            json_schema)) {
        output = std::move(result.content);
        n_decode = result.decoded_tokens;
        return false;
    }
    output = std::move(result.content);
    n_decode = result.decoded_tokens;
    return true;
}
