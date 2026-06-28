#include "agent-cli-inference.h"

#include "json-schema-to-grammar.h"
#include "server-context.h"
#include "server-task.h"

#include <cmath>
#include <cstdio>
#include <nlohmann/json.hpp>

namespace {

common_chat_params apply_generation_request(
        const common_chat_templates * templates,
        const common_agent_generation_request & request) {
    common_chat_templates_inputs chat_inputs;
    chat_inputs.messages = request.messages;
    chat_inputs.tools = request.tools;
    chat_inputs.tool_choice = request.tool_choice;
    chat_inputs.parallel_tool_calls = false;
    chat_inputs.add_generation_prompt = true;
    return common_chat_templates_apply(templates, chat_inputs);
}

task_params make_server_task_params(
        const common_agent_generation_request & request,
        const common_chat_params & chat_params,
        const std::vector<llama_logit_bias> & logit_bias_eog) {
    task_params params;
    params.stream = false;
    params.n_predict = request.options.n_predict;
    if (request.options.t_max_prompt_ms) {
        params.t_max_prompt_ms = *request.options.t_max_prompt_ms;
    }
    if (request.options.t_max_predict_ms) {
        params.t_max_predict_ms = *request.options.t_max_predict_ms;
    }
    params.sampling.temp = 0.0f;
    params.sampling.grammar_lazy = chat_params.grammar_lazy;
    params.sampling.grammar_triggers = chat_params.grammar_triggers;
    params.chat_parser_params = common_chat_parser_params(chat_params);
    params.chat_parser_params.parse_tool_calls = !request.tools.empty();

    if (!chat_params.parser.empty()) {
        params.chat_parser_params.parser.load(chat_params.parser);
    }

    if (request.json_schema.empty()) {
        params.sampling.grammar = common_grammar{COMMON_GRAMMAR_TYPE_TOOL_CALLS, chat_params.grammar};
        params.sampling.generation_prompt = chat_params.generation_prompt;
        return params;
    }

    params.sampling.grammar = common_grammar{
        COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
        json_schema_to_grammar(nlohmann::ordered_json::parse(request.json_schema)),
    };
    params.sampling.ignore_eos = true;
    params.sampling.logit_bias = logit_bias_eog;

    return params;
}

common_agent_generation_stop_reason map_server_stop_reason(stop_type stop) {
    switch (stop) {
        case STOP_TYPE_NONE:  return common_agent_generation_stop_reason::none;
        case STOP_TYPE_EOS:   return common_agent_generation_stop_reason::eos;
        case STOP_TYPE_LIMIT: return common_agent_generation_stop_reason::limit;
        case STOP_TYPE_WORD:  return common_agent_generation_stop_reason::limit;
    }
    return common_agent_generation_stop_reason::error;
}

void apply_server_success(
        common_agent_generation_result & result,
        std::string content,
        int decoded_tokens,
        common_agent_generation_stop_reason stop_reason) {
    result.content = std::move(content);
    result.decoded_tokens = decoded_tokens;
    result.status = common_agent_generation_status::completed;
    result.stop_reason = stop_reason;
    result.error_message.clear();
}

void apply_server_error(
        const server_task_result * response,
        common_agent_generation_result & result,
        common_agent_generation_status status = common_agent_generation_status::errored,
        common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::error,
        std::string fallback_error = {}) {
    result.status = status;
    result.stop_reason = stop_reason;
    if (auto * error = dynamic_cast<const server_task_result_error *>(response)) {
        result.error_message = error->err_msg;
    } else {
        result.error_message = std::move(fallback_error);
    }
}

class server_context_agent_inference final : public common_agent_inference {
public:
    server_context_agent_inference(
            server_context & server,
            const std::vector<llama_logit_bias> & logit_bias_eog,
            const common_chat_templates * templates)
        : server(server), logit_bias_eog(logit_bias_eog), templates(templates) {}

    bool generate(
            const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
        result = {};

        try {
            result.chat_params = apply_generation_request(templates, request);

            server_response_reader reader = server.get_response_reader();
            server_task task(SERVER_TASK_TYPE_COMPLETION);
            task.id = reader.get_new_id();
            task.cli = true;
            task.cli_prompt = result.chat_params.prompt;
            task.params = make_server_task_params(
                request,
                result.chat_params,
                logit_bias_eog);
            task.params.stream = !request.json_schema.empty();

            reader.post_task(std::move(task));

            if (!request.json_schema.empty()) {
                std::string content;
                int decoded_tokens = 0;
                server_task_result_cmpl_final * final_result = nullptr;

                while (auto response = reader.next([]() { return false; })) {
                    if (response->is_error()) {
                        apply_server_error(response.get(), result);
                        return false;
                    }
                    if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(response.get())) {
                        content += partial->content;
                        decoded_tokens = partial->n_decoded;
                        const auto parsed = nlohmann::ordered_json::parse(content, nullptr, false);
                        if (!parsed.is_discarded()) {
                            apply_server_success(result, content, decoded_tokens, common_agent_generation_stop_reason::json_schema);
                            reader.stop();
                            return true;
                        }
                    }
                    if ((final_result = dynamic_cast<server_task_result_cmpl_final *>(response.get())) != nullptr) {
                        break;
                    }
                }

                if (final_result != nullptr) {
                    content = final_result->content;
                    decoded_tokens = final_result->n_decoded;
                    const auto parsed = nlohmann::ordered_json::parse(content, nullptr, false);
                    if (!parsed.is_discarded()) {
                        apply_server_success(result, content, decoded_tokens, common_agent_generation_stop_reason::json_schema);
                        return true;
                    }
                }

                result.content = std::move(content);
                result.decoded_tokens = decoded_tokens;
                apply_server_error(
                    final_result,
                    result,
                    common_agent_generation_status::errored,
                    final_result == nullptr ? common_agent_generation_stop_reason::error : map_server_stop_reason(final_result->stop),
                    "generation ended before producing valid JSON for the requested schema");
                return false;
            }

            auto responses = reader.wait_for_all([]() { return false; });
            if (responses.is_terminated || responses.error || responses.results.empty()) {
                if (responses.error) {
                    apply_server_error(responses.error.get(), result);
                } else {
                    apply_server_error(
                        nullptr,
                        result,
                        responses.is_terminated ? common_agent_generation_status::cancelled : common_agent_generation_status::errored,
                        responses.is_terminated ? common_agent_generation_stop_reason::cancelled : common_agent_generation_stop_reason::error,
                        responses.is_terminated ? "server task was terminated" : "server task produced no completion result");
                }
                return false;
            }

            auto * final_result = dynamic_cast<server_task_result_cmpl_final *>(responses.results.front().get());
            if (final_result == nullptr) {
                apply_server_error(nullptr, result, common_agent_generation_status::errored, common_agent_generation_stop_reason::error,
                    "server returned a non-completion result");
                return false;
            }

            apply_server_success(result, final_result->content, final_result->n_decoded, map_server_stop_reason(final_result->stop));
            return true;
        } catch (const std::exception & err) {
            apply_server_error(nullptr, result, common_agent_generation_status::errored, common_agent_generation_stop_reason::error, err.what());
            std::fprintf(stderr, "server_context agent inference failed: %s\n", err.what());
            return false;
        }
    }

private:
    server_context & server;
    std::vector<llama_logit_bias> logit_bias_eog;
    const common_chat_templates * templates;
};

} // namespace

std::unique_ptr<common_agent_inference> make_server_context_agent_inference(
    server_context & server,
    const std::vector<llama_logit_bias> & logit_bias_eog,
    const common_chat_templates * templates) {
    return std::make_unique<server_context_agent_inference>(server, logit_bias_eog, templates);
}
