#pragma once

#include "../runtime/agent-runtime-assembly.h"
#include "agent/adaptation/adapter-registry.h"
#include "agent/runtime/agent-inference-contracts.h"

#include "chat.h"
#include "llama.h"

#include <memory>
#include <string>
#include <vector>

class common_agent_server_context_host;
struct common_agent_runtime_resident_model;

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
    std::string profile_id;
    std::string profile_cache_key;
    std::vector<llama_adapter_lora *> adapters;
    std::vector<float> adapter_scales;
    const common_chat_templates * chat_templates_view = nullptr;
    bool externally_owned = false;
    std::shared_ptr<void> residency_owner;

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

// Attach a model loaded by the process-wide residency manager.  The model
// resource remains owned by the manager; this session only owns its context
// and keeps the resource alive through residency_owner.
bool initialize_agent_runtime_session_from_resident_model(
    const common_agent_inference_options & options,
    agent_inference_backend backend,
    const std::shared_ptr<common_agent_runtime_resident_model> & resident_model,
    common_agent_runtime_session & session,
    std::string & error);

// Resolve and load an approved profile overlay into the already loaded base
// model.  The adapter root is host-owned; manifest paths remain relative and
// are never allowed to escape that root.  Changing the profile always
// recreates the inference context so KV/resident state cannot cross profiles.
bool apply_agent_runtime_model_profile(
    common_agent_runtime_session & session,
    const common_agent_model_profile & profile,
    const common_learning_adapter_registry & registry,
    const std::string & adapter_root,
    std::string & error);
