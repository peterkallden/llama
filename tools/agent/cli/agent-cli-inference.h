#pragma once

#include "agent/agent-inference.h"
#include "common.h"
#include "chat.h"
#include "llama.h"

#include <memory>
#include <vector>

struct server_context;

std::unique_ptr<common_agent_inference> make_llama_cli_agent_inference(
    llama_model * model,
    const common_chat_templates * templates,
    const std::vector<llama_adapter_lora *> & adapters = {},
    const std::vector<float> & adapter_scales = {});

std::unique_ptr<common_agent_inference> make_server_context_agent_inference(
    server_context & server,
    const common_params & params_base,
    const std::vector<llama_logit_bias> & logit_bias_eog,
    const common_chat_templates * templates,
    bool supports_image,
    bool supports_audio);
