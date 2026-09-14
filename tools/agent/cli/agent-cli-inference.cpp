#include "agent-cli-inference.h"

#include "agent/adaptation/flydelta/flydelta-activation.h"
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
