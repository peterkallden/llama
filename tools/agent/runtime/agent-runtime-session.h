#pragma once

#include "tools/agent/cli/agent-cli-options.h"

#include "../runtime/agent-runtime-assembly.h"

#include "chat.h"
#include "llama.h"

#include <memory>
#include <string>

class common_agent_server_context_host;

struct common_agent_model_load_key {
    std::string model;
    int n_gpu_layers = 0;
    bool fit_params = true;
    std::string mmproj;
};

struct common_agent_inference_context_key {
    agent_inference_backend backend = agent_inference_backend::cli;
    common_agent_model_load_key model_key;
};

struct common_agent_runtime_loaded_model_state {
    llama_model * model = nullptr;
    common_chat_templates_ptr chat_templates;
    std::shared_ptr<common_agent_server_context_host> server_context_host;
    bool loaded = false;
    agent_inference_backend backend = agent_inference_backend::cli;
    common_agent_model_load_key key;

    void reset();
};

struct common_agent_runtime_inference_context_state {
    common_agent_inference_session session;
    bool initialized = false;
    common_agent_inference_context_key key;

    void reset();
};

struct common_agent_runtime_session {
    common_agent_runtime_loaded_model_state loaded_model;
    common_agent_runtime_inference_context_state inference_context;

    common_agent_runtime_session() = default;
    common_agent_runtime_session(common_agent_runtime_session &&) = default;
    common_agent_runtime_session & operator=(common_agent_runtime_session && other);
    ~common_agent_runtime_session();

    const common_agent_inference_session * active_inference_session() const;
    common_agent_inference_session * active_inference_session();

    void reset();
};

bool initialize_agent_runtime_session(
    const common_agent_inference_options & options,
    agent_inference_backend backend,
    bool memory_enabled,
    const std::string & fallback_reason,
    common_agent_runtime_session & session,
    std::string & error);
