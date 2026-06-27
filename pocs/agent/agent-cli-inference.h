#pragma once

#include "agent/agent-inference.h"
#include "chat.h"
#include "llama.h"

#include <memory>

std::unique_ptr<common_agent_inference> make_llama_cli_agent_inference(
    llama_model * model,
    const common_chat_templates * templates);
