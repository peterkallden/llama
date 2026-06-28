#include "memory-cli-chat.h"

#include "common.h"
#include "json-schema-to-grammar.h"
#include "sampling.h"

#include <cmath>
#include <vector>

#include <nlohmann/json.hpp>

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
    common_chat_templates_inputs chat_inputs;
    chat_inputs.messages = messages;
    chat_inputs.tools = tools;
    chat_inputs.tool_choice = tool_choice;
    chat_inputs.parallel_tool_calls = false;
    chat_inputs.add_generation_prompt = true;
    chat_params = common_chat_templates_apply(chat_templates, chat_inputs);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, chat_params.prompt.c_str(), chat_params.prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "failed to tokenize chat prompt\n");
        return false;
    }
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, chat_params.prompt.c_str(), chat_params.prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "failed to tokenize chat prompt\n");
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + options.n_predict;
    ctx_params.n_batch = n_prompt;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create llama context\n");
        return false;
    }

    common_params_sampling sampling;
    sampling.temp = 0.0f;
    sampling.grammar = json_schema.empty()
        ? common_grammar{ COMMON_GRAMMAR_TYPE_TOOL_CALLS, chat_params.grammar }
        : common_grammar{ COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(nlohmann::ordered_json::parse(json_schema)) };
    sampling.grammar_lazy = chat_params.grammar_lazy;
    sampling.grammar_triggers = chat_params.grammar_triggers;
    sampling.generation_prompt = json_schema.empty() ? chat_params.generation_prompt : std::string{};
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
    output.clear();
    n_decode = 0;
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + options.n_predict; ) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "failed to decode\n");
            llama_free(ctx);
            return false;
        }
        n_pos += batch.n_tokens;
        llama_token token = common_sampler_sample(sampler.get(), ctx, -1, true);
        common_sampler_accept(sampler.get(), token, true);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        const std::string piece = common_token_to_piece(vocab, token, true);
        output += piece;
        batch = llama_batch_get_one(&token, 1);
        n_decode++;
        if (!json_schema.empty()) {
            const auto parsed = nlohmann::ordered_json::parse(output, nullptr, false);
            if (!parsed.is_discarded()) {
                break;
            }
        }
    }

    llama_free(ctx);
    return true;
}
