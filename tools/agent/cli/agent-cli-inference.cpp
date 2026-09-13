#include "agent-cli-inference.h"

#include "tools/agent/cli/agent-cli-generation.h"

namespace {

class llama_cli_agent_inference final : public common_agent_inference {
public:
    llama_cli_agent_inference(
            llama_model * model,
            const common_chat_templates * templates,
            std::vector<llama_adapter_lora *> adapters,
            std::vector<float> adapter_scales,
            common_flydelta_static_overlay flydelta_overlay)
        : model(model), templates(templates), adapters(std::move(adapters)),
          adapter_scales(std::move(adapter_scales)),
          flydelta_overlay(std::move(flydelta_overlay)) {}

    bool generate(
            const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
        common_chat_params chat_params;
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
            flydelta_overlay);
        result.chat_params = chat_params;
        return ok;
    }

private:
    llama_model * model;
    const common_chat_templates * templates;
    std::vector<llama_adapter_lora *> adapters;
    std::vector<float> adapter_scales;
    common_flydelta_static_overlay flydelta_overlay;
};

} // namespace

std::unique_ptr<common_agent_inference> make_llama_cli_agent_inference(
    llama_model * model,
    const common_chat_templates * templates,
    const std::vector<llama_adapter_lora *> & adapters,
    const std::vector<float> & adapter_scales,
    const common_flydelta_static_overlay & flydelta_overlay) {
    return std::make_unique<llama_cli_agent_inference>(
        model, templates, adapters, adapter_scales, flydelta_overlay);
}
