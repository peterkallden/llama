#pragma once

#include "../common/cli-config.h"

#include "agent-runtime-assembly.h"

#include "chat.h"
#include "llama.h"

#include <string>

struct common_agent_model_load_key {
    std::string model;
    int n_gpu_layers = 0;
    bool fit_params = true;
};

struct common_agent_inference_context_key {
    agent_inference_backend backend = agent_inference_backend::cli;
    common_agent_model_load_key model_key;
};

struct common_agent_runtime_session {
    llama_model * model = nullptr;
    common_chat_templates_ptr chat_templates;
    common_agent_inference_session inference_session;
    bool model_loaded = false;
    agent_inference_backend loaded_model_backend = agent_inference_backend::cli;
    common_agent_model_load_key loaded_model_key;
    bool initialized = false;
    common_agent_inference_context_key initialized_context_key;

    common_agent_runtime_session() = default;
    common_agent_runtime_session(common_agent_runtime_session &&) = default;
    common_agent_runtime_session & operator=(common_agent_runtime_session && other);
    ~common_agent_runtime_session();

    void reset();
    void reset_inference_context();
    void reset_loaded_model();
};

bool initialize_agent_runtime_session(
    const common_agent_inference_options & options,
    agent_inference_backend backend,
    bool memory_enabled,
    const std::string & fallback_reason,
    common_agent_runtime_session & session,
    std::string & error);
