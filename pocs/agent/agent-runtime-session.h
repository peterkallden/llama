#pragma once

#include "../common/cli-config.h"

#include "agent-runtime-assembly.h"

#include "chat.h"
#include "llama.h"

#include <string>

struct common_agent_runtime_session {
    llama_model * model = nullptr;
    common_chat_templates_ptr chat_templates;
    common_agent_inference_session inference_session;
    bool initialized = false;
    agent_inference_backend initialized_backend = agent_inference_backend::cli;
    common_agent_inference_options initialized_options;

    common_agent_runtime_session() = default;
    common_agent_runtime_session(common_agent_runtime_session &&) = default;
    common_agent_runtime_session & operator=(common_agent_runtime_session && other);
    ~common_agent_runtime_session();

    void reset();
};

bool initialize_agent_runtime_session(
    const common_agent_inference_options & options,
    agent_inference_backend backend,
    bool memory_enabled,
    const std::string & fallback_reason,
    common_agent_runtime_session & session,
    std::string & error);
