#include "agent-runtime-session.h"

#include "chat.h"

namespace {

common_agent_model_load_key make_agent_model_load_key(
        const common_agent_inference_options & options) {
    return {
        options.model,
        options.n_gpu_layers,
        options.fit_params,
    };
}

common_agent_inference_context_key make_agent_inference_context_key(
        const common_agent_inference_options & options,
        agent_inference_backend backend) {
    return {
        backend,
        make_agent_model_load_key(options),
        options.n_predict,
    };
}

bool common_agent_model_load_key_match(
        const common_agent_model_load_key & lhs,
        const common_agent_model_load_key & rhs) {
    return lhs.model == rhs.model &&
        lhs.n_gpu_layers == rhs.n_gpu_layers &&
        lhs.fit_params == rhs.fit_params;
}

bool common_agent_inference_context_key_match(
        const common_agent_inference_context_key & lhs,
        const common_agent_inference_context_key & rhs) {
    return lhs.backend == rhs.backend &&
        lhs.n_predict == rhs.n_predict &&
        common_agent_model_load_key_match(lhs.model_key, rhs.model_key);
}

} // namespace

common_agent_runtime_session & common_agent_runtime_session::operator=(common_agent_runtime_session && other) {
    if (this != &other) {
        reset();
        model = other.model;
        other.model = nullptr;
        chat_templates = std::move(other.chat_templates);
        inference_session = std::move(other.inference_session);
        model_loaded = other.model_loaded;
        loaded_model_backend = other.loaded_model_backend;
        loaded_model_key = std::move(other.loaded_model_key);
        initialized = other.initialized;
        initialized_context_key = std::move(other.initialized_context_key);
        other.model_loaded = false;
        other.initialized = false;
    }
    return *this;
}

common_agent_runtime_session::~common_agent_runtime_session() {
    reset();
}

void common_agent_runtime_session::reset_inference_context() {
    inference_session = {};
    initialized = false;
    initialized_context_key = {};
}

void common_agent_runtime_session::reset_loaded_model() {
    chat_templates.reset();
    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }
    model_loaded = false;
    loaded_model_backend = agent_inference_backend::cli;
    loaded_model_key = {};
}

void common_agent_runtime_session::reset() {
    reset_inference_context();
    reset_loaded_model();
}

bool initialize_agent_runtime_session(
    const common_agent_inference_options & options,
    agent_inference_backend backend,
    bool memory_enabled,
    const std::string & fallback_reason,
    common_agent_runtime_session & session,
    std::string & error) {
    (void) memory_enabled;
    (void) fallback_reason;

    const auto requested_model_key = make_agent_model_load_key(options);
    const auto requested_context_key = make_agent_inference_context_key(options, backend);

    if (session.initialized &&
            session.inference_session.inference &&
            common_agent_inference_context_key_match(session.initialized_context_key, requested_context_key)) {
        error.clear();
        return true;
    }

    const bool reuse_loaded_model =
        backend == agent_inference_backend::cli &&
        session.model_loaded &&
        session.loaded_model_backend == backend &&
        common_agent_model_load_key_match(session.loaded_model_key, requested_model_key);

    session.reset_inference_context();

    if (!reuse_loaded_model) {
        session.reset_loaded_model();
    }

    if (backend == agent_inference_backend::cli) {
        if (!session.model_loaded) {
            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers = options.n_gpu_layers;
            session.model = llama_model_load_from_file(options.model.c_str(), model_params);
            if (session.model == nullptr) {
                error = "failed to load model: " + options.model;
                return false;
            }
            session.chat_templates = common_chat_templates_init(session.model, "");
            session.model_loaded = true;
            session.loaded_model_backend = backend;
            session.loaded_model_key = requested_model_key;
        }
    } else {
        session.reset_loaded_model();
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

    session.initialized = true;
    session.initialized_context_key = requested_context_key;
    error.clear();
    return true;
}
