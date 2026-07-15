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
        LOG_LEVEL_WARN,
    };
}

common_params make_agent_server_context_params(
        const common_agent_server_context_host_config & config) {
    common_params params = {};
    params.model.path = config.context_key.load_key.model;
    params.n_predict = -1;
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
    : instance(std::make_unique<common_agent_server_context_running_instance>()) {}

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
    instance = std::make_unique<common_agent_server_context_running_instance>();
    instance->server = std::make_unique<server_context>();
    instance->params = make_agent_server_context_params(config);

    if (!instance->server->load_model(instance->params)) {
        error = "failed to load resident server_context model: " + current_load_key.model;
        instance.reset();
        return false;
    }

    instance->loop = std::make_unique<std::thread>([this]() {
        instance->server->start_loop();
    });
    instance->running = true;
    error.clear();
    return true;
}

bool common_agent_server_context_host::build_inference_session(
        common_agent_inference_session & session,
        std::string & error) const {
    if (!instance || !instance->running || !instance->server) {
        error = "server_context host is not running";
        return false;
    }

    session = {};
    session.backend = agent_inference_backend::server_context;
    auto meta = instance->server->get_meta();
    session.templates = meta.chat_params.tmpls.get();
    session.inference = make_server_context_agent_inference(
        *instance->server,
        instance->params,
        meta.logit_bias_eog,
        session.templates);
    error.clear();
    return true;
}

void common_agent_server_context_host::stop() {
    if (!instance || !instance->running) {
        return;
    }
    instance->server->terminate();
    if (instance->loop && instance->loop->joinable()) {
        instance->loop->join();
    }
    instance.reset();
}

server_context & common_agent_server_context_host::server() {
    return *instance->server;
}

const server_context & common_agent_server_context_host::server() const {
    return *instance->server;
}
