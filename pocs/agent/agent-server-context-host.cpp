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

common_agent_server_context_context_key make_agent_server_context_context_key(
        const common_agent_inference_options & options) {
    return {
        make_agent_server_context_load_key(options),
        1,
        1,
        0,
    };
}

common_agent_server_context_host_config make_agent_server_context_host_config(
        const common_agent_inference_options & options) {
    return {
        make_agent_server_context_context_key(options),
        options.n_predict,
        LOG_LEVEL_WARN,
    };
}

common_params make_agent_server_context_params(
        const common_agent_server_context_host_config & config) {
    common_params params = {};
    params.model.path = config.context_key.load_key.model;
    params.n_predict = config.n_predict;
    params.n_gpu_layers = config.context_key.load_key.n_gpu_layers;
    params.fit_params = config.context_key.load_key.fit_params;
    params.n_parallel = config.context_key.n_parallel;
    params.n_sequences = config.context_key.n_sequences;
    params.n_ctx = config.context_key.n_ctx;
    params.verbosity = config.verbosity;
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    return params;
}

common_params make_agent_server_context_params(
        const common_agent_inference_options & options) {
    return make_agent_server_context_params(make_agent_server_context_host_config(options));
}

common_agent_server_context_host::common_agent_server_context_host()
    : server_ptr(std::make_unique<server_context>()) {}

common_agent_server_context_host::~common_agent_server_context_host() {
    stop();
}

bool common_agent_server_context_host::start(
        const common_agent_server_context_host_config & config,
        std::string & error) {
    stop();

    current_config = config;
    current_context_key = config.context_key;
    current_load_key = config.context_key.load_key;
    params_base = make_agent_server_context_params(config);

    if (!server_ptr->load_model(params_base)) {
        error = "failed to load resident server_context model: " + current_load_key.model;
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
    if (!host->start(make_agent_server_context_host_config(options), error)) {
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
