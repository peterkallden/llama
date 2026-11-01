#include "agent-cli-inference.h"

#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-decision-margin.h"
#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"
#include "tools/agent/cli/agent-cli-generation.h"

#include <algorithm>

namespace {

class llama_cli_agent_inference final : public common_agent_inference {
public:
    llama_cli_agent_inference(
            llama_model * model,
            const common_chat_templates * templates,
            std::vector<llama_adapter_lora *> adapters,
            std::vector<float> adapter_scales, size_t context_size_tokens)
        : model(model), templates(templates), adapters(std::move(adapters)),
          adapter_scales(std::move(adapter_scales)), context_size_tokens(context_size_tokens) {}

    size_t context_capacity_tokens() const override {
        if (context_size_tokens != 0) return context_size_tokens;
        return model ? static_cast<size_t>(std::max(0, llama_model_n_ctx_train(model))) : 0;
    }
    std::optional<size_t> count_context_text_tokens(const std::string & text) const override {
        if (!model) return std::nullopt;
        return common_tokenize(llama_model_get_vocab(model), text, false, true).size();
    }

    bool generate(
            const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
        common_agent_prepared_generation prepared;
        if (!model || !common_agent_prepare_chat_generation(templates, request, prepared)) {
            result = {};
            result.error_message = "failed to prepare CLI generation for context validation";
            return false;
        }
        const size_t prompt_tokens = common_tokenize(llama_model_get_vocab(model), prepared.prompt, true, true).size();
        std::string budget_error;
        if (!common_agent_generation_context_fits(prompt_tokens, context_capacity_tokens(), request.options.n_predict, budget_error)) {
            result = {};
            result.error_message = budget_error;
            return false;
        }
        common_chat_params chat_params;
        common_flydelta_static_overlay no_overlay;
        const common_flydelta_static_overlay * flydelta_overlay = &no_overlay;
        if (request.flydelta_activation) {
            std::string activation_error;
            if (!common_flydelta_activation_result_validate(
                    *request.flydelta_activation,
                    static_cast<size_t>(llama_model_n_embd(model)),
                    static_cast<size_t>(llama_model_n_layer(model)),
                    64U * 1024U * 1024U,
                    activation_error)) {
                result = {};
                result.error_message = "invalid FlyDelta activation: " + activation_error;
                return false;
            }
            flydelta_overlay = &request.flydelta_activation->overlay;
        }
        const bool ok = generate_chat_turn_result(
            model,
            templates,
            request.messages,
            request.tools,
            request.tool_choice,
            request.options,
            result,
            &chat_params,
            request.json_schema,
            adapters,
            adapter_scales,
            *flydelta_overlay,
            request.flydelta_capture,
            request.grammar);
        result.chat_params = chat_params;
        return ok;
    }

    bool score_teacher_forced_choice(
            const common_agent_teacher_forced_choice_request & request,
            common_agent_teacher_forced_choice_result & result) override {
        result = {};
        result.sequence_id = request.sequence_id;
        common_flydelta_static_overlay no_overlay;
        const common_flydelta_static_overlay * overlay = &no_overlay;
        if (request.context.flydelta_activation) {
            std::string activation_error;
            if (!common_flydelta_activation_result_validate(
                    *request.context.flydelta_activation,
                    static_cast<size_t>(llama_model_n_embd(model)),
                    static_cast<size_t>(llama_model_n_layer(model)),
                    64U * 1024U * 1024U,
                    activation_error)) {
                result.error_message = "invalid FlyDelta scoring activation: " + activation_error;
                return false;
            }
            overlay = &request.context.flydelta_activation->overlay;
        }
        common_flydelta_decision_margin margin;
        std::string error;
        const std::string & positive = request.positive_continuation.empty()
            ? request.positive_choice : request.positive_continuation;
        const std::string & negative = request.negative_continuation.empty()
            ? request.negative_choice : request.negative_continuation;
        if (!common_agent_teacher_forced_contrast_is_valid(
                request.choice_prefix, positive, negative, result.error_message)) {
            return false;
        }
        if (!score_chat_choice_margin(
                model, templates, request.context.messages, request.context.tools,
                request.context.tool_choice, request.context.options,
                request.choice_prefix, positive, negative,
                margin, nullptr, request.context.json_schema, adapters, adapter_scales,
                *overlay, &error)) {
            result.error_message = std::move(error);
            return false;
        }
        result.available = margin.available;
        result.positive_total_logprob = margin.positive_total_logprob;
        result.negative_total_logprob = margin.negative_total_logprob;
        result.positive_token_count = margin.positive_token_count;
        result.negative_token_count = margin.negative_token_count;
        return true;
    }

private:
    llama_model * model;
    const common_chat_templates * templates;
    std::vector<llama_adapter_lora *> adapters;
    std::vector<float> adapter_scales;
    size_t context_size_tokens;
};

} // namespace

std::unique_ptr<common_agent_inference> make_llama_cli_agent_inference(
    llama_model * model,
    const common_chat_templates * templates,
    const std::vector<llama_adapter_lora *> & adapters,
    const std::vector<float> & adapter_scales,
    size_t context_size_tokens) {
    return std::make_unique<llama_cli_agent_inference>(model, templates, adapters, adapter_scales, context_size_tokens);
}
