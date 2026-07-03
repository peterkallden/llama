#include "agent-server-context-host.h"

#include "agent-cli-inference.h"

#include "log.h"
#include "server-context.h"

common_agent_server_context_load_key make_agent_server_context_load_key(
        const common_agent_inference_options & options) {
    return {
        options.model,
        options.n_gpu_layers,
        options.fit_params,
    };
}

common_params make_agent_server_context_load_params(
        const common_agent_inference_options & options) {
    common_params params = {};
    params.model.path = options.model;
    params.n_predict = options.n_predict;
    params.n_gpu_layers = options.n_gpu_layers;
    params.fit_params = options.fit_params;
    params.n_parallel = 1;
    params.n_sequences = 1;
    params.n_ctx = 0;
    params.verbosity = LOG_LEVEL_WARN;
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    return params;
}

common_agent_server_context_host::common_agent_server_context_host()
    : server_ptr(std::make_unique<server_context>()) {}

common_agent_server_context_host::~common_agent_server_context_host() {
    stop();
}

bool common_agent_server_context_host::start(
        const common_agent_inference_options & options,
        std::string & error) {
    stop();

    current_load_key = make_agent_server_context_load_key(options);
    params_base = make_agent_server_context_load_params(options);

    if (!server_ptr->load_model(params_base)) {
        error = "failed to load resident server_context model: " + options.model;
        return false;
    }

    loop = std::make_unique<std::thread>([this]() {
        server_ptr->start_loop();
    });
    running = true;
    error.clear();
    return true;
}

void common_agent_server_context_host::stop() {
    if (!running) {
        return;
    }
    server_ptr->terminate();
    if (loop && loop->joinable()) {
        loop->join();
    }
    loop.reset();
    running = false;
}

server_context & common_agent_server_context_host::server() {
    return *server_ptr;
}

const server_context & common_agent_server_context_host::server() const {
    return *server_ptr;
}

bool make_server_context_inference_session(
        const common_agent_inference_options & options,
        common_agent_inference_session & session,
        std::string & error) {
    auto host = std::make_shared<common_agent_server_context_host>();
    if (!host->start(options, error)) {
        return false;
    }

    session = {};
    session.backend = agent_inference_backend::server_context;
    auto meta = host->server().get_meta();
    session.templates = meta.chat_params.tmpls.get();
    session.inference = make_server_context_agent_inference(
        host->server(),
        host->params(),
        meta.logit_bias_eog,
        session.templates);
    session.keepalive = std::move(host);
    error.clear();
    return true;
}
