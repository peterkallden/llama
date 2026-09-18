#include "agent-cli-inference.h"

#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-decision-margin.h"
#include "agent/adaptation/flydelta/flydelta-hidden-state-hook.h"
#include "tools/agent/cli/agent-cli-generation.h"

namespace {

class llama_cli_agent_inference final : public common_agent_inference {
public:
    llama_cli_agent_inference(
            llama_model * model,
            const common_chat_templates * templates,
            std::vector<llama_adapter_lora *> adapters,
            std::vector<float> adapter_scales)
        : model(model), templates(templates), adapters(std::move(adapters)),
          adapter_scales(std::move(adapter_scales)) {}

    bool generate(
            const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
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
            request.flydelta_capture);
        result.chat_params = chat_params;
        return ok;
    }

    bool score_teacher_forced_choice(
            const common_agent_teacher_forced_choice_request & request,
            common_agent_teacher_forced_choice_result & result) override {
        result = {};
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
};

} // namespace

std::unique_ptr<common_agent_inference> make_llama_cli_agent_inference(
    llama_model * model,
    const common_chat_templates * templates,
    const std::vector<llama_adapter_lora *> & adapters,
    const std::vector<float> & adapter_scales) {
    return std::make_unique<llama_cli_agent_inference>(model, templates, adapters, adapter_scales);
}
