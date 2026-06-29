#include "agent-runtime-session.h"

#include "chat.h"

#include <cstdio>

common_agent_cli_runtime_session & common_agent_cli_runtime_session::operator=(common_agent_cli_runtime_session && other) {
    if (this != &other) {
        reset();
        model = other.model;
        other.model = nullptr;
        chat_templates = std::move(other.chat_templates);
        inference_session = std::move(other.inference_session);
    }
    return *this;
}

common_agent_cli_runtime_session::~common_agent_cli_runtime_session() {
    reset();
}

void common_agent_cli_runtime_session::reset() {
    inference_session = {};
    chat_templates.reset();
    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }
}

bool initialize_agent_cli_runtime_session(
    const common_agent_inference_options & options,
    agent_inference_backend backend,
    bool memory_enabled,
    const std::string & fallback_reason,
    common_agent_cli_runtime_session & session,
    std::string & error) {
    session.reset();

    if (backend == agent_inference_backend::cli) {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.n_gpu_layers;
        if (!memory_enabled) {
            fprintf(stderr,
                "debug: chat fallback active, loading %s without memory retrieval or episode recording (%s)\n",
                options.model.c_str(),
                fallback_reason.c_str());
        }
        session.model = llama_model_load_from_file(options.model.c_str(), model_params);
        if (session.model == nullptr) {
            error = "failed to load model: " + options.model;
            return false;
        }
        session.chat_templates = common_chat_templates_init(session.model, "");
    }

    if (!make_agent_inference_session(
            options,
            backend,
            session.model,
            session.chat_templates.get(),
            session.inference_session,
            error)) {
        session.reset();
        return false;
    }

    error.clear();
    return true;
}
