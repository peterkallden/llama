#include "agent-server-context-host.h"

#include "../cli/agent-cli-inference.h"

#include "log.h"
#include "common.h"
#include "server-context.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <system_error>

namespace {

bool server_context_host_trace_enabled() {
    const char * value = std::getenv("LLAMA_AGENT_RESIDENT_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void server_context_host_trace(const char * event, const common_agent_server_context_host_config * config = nullptr) {
    if (!server_context_host_trace_enabled()) {
        return;
    }
    std::fprintf(stderr, "agent server_context host trace: event=%s", event);
    if (config != nullptr) {
        std::fprintf(stderr,
            " model=%s n_ctx=%d n_threads=%d n_parallel=%d n_sequences=%d fit_params=%s",
            config->context_key.load_key.model.c_str(),
            config->context_key.n_ctx,
            config->context_key.n_threads,
            config->context_key.n_parallel,
            config->context_key.n_sequences,
            config->context_key.load_key.fit_params ? "true" : "false");
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

void initialize_server_context_host_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        common_init();
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    });
}

} // namespace

common_agent_server_context_load_key make_agent_server_context_load_key(
        const common_agent_inference_options & options) {
    return {
        options.model,
        options.n_gpu_layers,
        options.fit_params,
        options.mmproj,
    };
}

common_agent_server_context_context_key make_agent_server_context_context_key(
        const common_agent_inference_options & options) {
    return {
        make_agent_server_context_load_key(options),
        1,
        1,
        static_cast<int>(options.context_size_tokens),
        options.n_threads,
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
    params.mmproj.path = config.context_key.load_key.mmproj;
    params.n_predict = -1;
    params.n_gpu_layers = config.context_key.load_key.n_gpu_layers;
    params.fit_params = config.context_key.load_key.fit_params;
    params.n_parallel = config.context_key.n_parallel;
    params.n_sequences = config.context_key.n_sequences;
    params.n_ctx = config.context_key.n_ctx;
    params.cpuparams.n_threads = config.context_key.n_threads;
    params.cpuparams_batch.n_threads = config.context_key.n_threads;
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
    server_context_host_trace("start-enter", &config);
    stop();
    initialize_server_context_host_once();

    if (config.context_key.load_key.model.empty()) {
        error = "resident server_context model path is empty";
        return false;
    }
    {
        std::error_code ec;
        const std::filesystem::path model_path(config.context_key.load_key.model);
        if (!std::filesystem::exists(model_path, ec)) {
            error = "resident server_context model does not exist: " + config.context_key.load_key.model;
            return false;
        }
        if (!std::filesystem::is_regular_file(model_path, ec)) {
            error = "resident server_context model is not a regular file: " + config.context_key.load_key.model;
            return false;
        }
    }
    if (!config.context_key.load_key.mmproj.empty()) {
        std::error_code ec;
        const std::filesystem::path mmproj_path(config.context_key.load_key.mmproj);
        if (!std::filesystem::exists(mmproj_path, ec)) {
            error = "resident server_context mmproj does not exist: " + config.context_key.load_key.mmproj;
            return false;
        }
        if (!std::filesystem::is_regular_file(mmproj_path, ec)) {
            error = "resident server_context mmproj is not a regular file: " + config.context_key.load_key.mmproj;
            return false;
        }
    }

    current_config = config;
    current_context_key = config.context_key;
    current_load_key = config.context_key.load_key;
    instance = std::make_unique<common_agent_server_context_running_instance>();
    instance->server = std::make_unique<server_context>();
    instance->params = make_agent_server_context_params(config);

    server_context_host_trace("before-load-model", &config);
    if (!instance->server->load_model(instance->params)) {
        error = "failed to load resident server_context model: " + current_load_key.model;
        instance.reset();
        return false;
    }
    server_context_host_trace("after-load-model", &config);

    instance->loop = std::make_unique<std::thread>([this]() {
        if (server_context_host_trace_enabled()) {
            std::fprintf(stderr, "agent server_context host trace: event=loop-enter\n");
            std::fflush(stderr);
        }
        instance->server->start_loop();
        if (server_context_host_trace_enabled()) {
            std::fprintf(stderr, "agent server_context host trace: event=loop-exit\n");
            std::fflush(stderr);
        }
    });
    instance->running = true;
    server_context_host_trace("start-exit", &config);
    error.clear();
    return true;
}

bool common_agent_server_context_host::build_inference_session(
        common_agent_inference_session & session,
        std::string & error) const {
    if (server_context_host_trace_enabled()) {
        std::fprintf(stderr, "agent server_context host trace: event=build-inference-session-enter\n");
        std::fflush(stderr);
    }
    if (!instance || !instance->running || !instance->server) {
        error = "server_context host is not running";
        return false;
    }

    session = {};
    session.backend = agent_inference_backend::server_context;
    auto meta = instance->server->get_meta();
    session.capabilities.image = meta.chat_params.allow_image;
    session.capabilities.audio = meta.chat_params.allow_audio;
    session.templates = meta.chat_params.tmpls.get();
    session.inference = make_server_context_agent_inference(
        *instance->server,
        instance->params,
        meta.logit_bias_eog,
        session.templates);
    if (server_context_host_trace_enabled()) {
        std::fprintf(stderr, "agent server_context host trace: event=build-inference-session-exit\n");
        std::fflush(stderr);
    }
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
