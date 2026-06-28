#pragma once

#include "agent/agent-inference.h"
#include "chat.h"
#include "llama.h"

#include <memory>

struct server_context;

std::unique_ptr<common_agent_inference> make_llama_cli_agent_inference(
    llama_model * model,
    const common_chat_templates * templates);

std::unique_ptr<common_agent_inference> make_server_context_agent_inference(
    server_context & server,
    const common_chat_templates * templates);
