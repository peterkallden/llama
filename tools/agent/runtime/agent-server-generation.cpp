#include "agent-server-generation.h"

task_params make_server_task_params_from_prepared_generation(
        const common_params & params_base,
        const common_agent_generation_request & request,
        const common_agent_prepared_generation & prepared,
        const std::vector<llama_logit_bias> & logit_bias_eog) {
    task_params params;
    params.sampling = params_base.sampling;
    params.speculative = params_base.speculative;
    params.n_keep = params_base.n_keep;
    params.n_cache_reuse = params_base.n_cache_reuse;
    params.cache_prompt = params_base.cache_prompt;
    params.antiprompt = params_base.antiprompt;
    params.stream = prepared.stream;
    params.n_predict = request.options.n_predict;
    if (request.options.t_max_prompt_ms) {
        params.t_max_prompt_ms = *request.options.t_max_prompt_ms;
    }
    if (request.options.t_max_predict_ms) {
        params.t_max_predict_ms = *request.options.t_max_predict_ms;
    }
    params.sampling.temp = 0.0f;
    params.sampling.grammar = prepared.grammar;
    params.sampling.grammar_lazy = prepared.grammar_lazy;
    params.sampling.grammar_triggers = prepared.grammar_triggers;
    params.sampling.generation_prompt = prepared.generation_prompt;
    params.sampling.ignore_eos = prepared.ignore_eos;
    if (prepared.suppress_eog) {
        params.sampling.logit_bias = logit_bias_eog;
    }
    params.chat_parser_params.format = prepared.chat_format;
    params.chat_parser_params.generation_prompt = prepared.parser_generation_prompt;
    params.chat_parser_params.parse_tool_calls = prepared.parse_tool_calls;

    if (!prepared.parser.empty()) {
        params.chat_parser_params.parser.load(prepared.parser);
    }

    return params;
}
