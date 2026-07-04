#include "agent-runtime-session.h"

#include "agent-cli-inference.h"
#include "agent-server-context-host.h"

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
        common_agent_model_load_key_match(lhs.model_key, rhs.model_key);
}

bool build_agent_inference_session(
        const common_agent_inference_options & options,
        agent_inference_backend backend,
        llama_model * model,
        const common_chat_templates * templates,
        common_agent_inference_session & session,
        std::string & error) {
    session = {};
    session.backend = backend;
    session.model = model;
    session.templates = templates;
    if (backend == agent_inference_backend::server_context) {
        return make_server_context_inference_session(options, session, error);
    }

    session.inference = make_llama_cli_agent_inference(model, templates);
    error.clear();
    return true;
}

} // namespace

void common_agent_runtime_loaded_model_state::reset() {
    chat_templates.reset();
    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }
    loaded = false;
    backend = agent_inference_backend::cli;
    key = {};
}

void common_agent_runtime_inference_context_state::reset() {
    session = {};
    initialized = false;
    key = {};
}

common_agent_runtime_session & common_agent_runtime_session::operator=(common_agent_runtime_session && other) {
    if (this != &other) {
        reset();
        loaded_model = std::move(other.loaded_model);
        inference_context = std::move(other.inference_context);
        other.loaded_model.loaded = false;
        other.loaded_model.model = nullptr;
        other.inference_context.initialized = false;
    }
    return *this;
}

common_agent_runtime_session::~common_agent_runtime_session() {
    reset();
}

const common_agent_inference_session * common_agent_runtime_session::active_inference_session() const {
    return inference_context.initialized ? &inference_context.session : nullptr;
}

common_agent_inference_session * common_agent_runtime_session::active_inference_session() {
    return inference_context.initialized ? &inference_context.session : nullptr;
}

void common_agent_runtime_session::reset() {
    inference_context.reset();
    loaded_model.reset();
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

    if (session.inference_context.initialized &&
            session.inference_context.session.inference &&
            common_agent_inference_context_key_match(session.inference_context.key, requested_context_key)) {
        error.clear();
        return true;
    }

    const bool reuse_loaded_model =
        backend == agent_inference_backend::cli &&
        session.loaded_model.loaded &&
        session.loaded_model.backend == backend &&
        common_agent_model_load_key_match(session.loaded_model.key, requested_model_key);

    session.inference_context.reset();

    if (!reuse_loaded_model) {
        session.loaded_model.reset();
    }

    if (backend == agent_inference_backend::cli) {
        if (!session.loaded_model.loaded) {
            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers = options.n_gpu_layers;
            session.loaded_model.model = llama_model_load_from_file(options.model.c_str(), model_params);
            if (session.loaded_model.model == nullptr) {
                error = "failed to load model: " + options.model;
                return false;
            }
            session.loaded_model.chat_templates = common_chat_templates_init(session.loaded_model.model, "");
            session.loaded_model.loaded = true;
            session.loaded_model.backend = backend;
            session.loaded_model.key = requested_model_key;
        }
    } else {
        session.loaded_model.reset();
    }

    if (!build_agent_inference_session(
            options,
            backend,
            session.loaded_model.model,
            session.loaded_model.chat_templates.get(),
            session.inference_context.session,
            error)) {
        session.reset();
        return false;
    }

    session.inference_context.initialized = true;
    session.inference_context.key = requested_context_key;
    error.clear();
    return true;
}
