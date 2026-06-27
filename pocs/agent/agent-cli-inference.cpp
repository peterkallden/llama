#include "agent-cli-inference.h"

#include "../memory/memory-cli-chat.h"

namespace {

class llama_cli_agent_inference final : public common_agent_inference {
public:
    llama_cli_agent_inference(llama_model * model, const common_chat_templates * templates)
        : model(model), templates(templates) {}

    bool infer(
            const common_agent_inference_request & request,
            common_agent_inference_result & result) override {
        result = {};
        return generate_chat_turn(
            model,
            templates,
            request.messages,
            request.tools,
            request.tool_choice,
            request.options,
            result.output,
            result.chat_params,
            result.n_decode,
            request.json_schema);
    }

private:
    llama_model * model;
    const common_chat_templates * templates;
};

} // namespace

std::unique_ptr<common_agent_inference> make_llama_cli_agent_inference(
    llama_model * model,
    const common_chat_templates * templates) {
    return std::make_unique<llama_cli_agent_inference>(model, templates);
}
