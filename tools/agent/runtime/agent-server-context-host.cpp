#include "agent-server-context-host.h"

#include "../cli/agent-cli-inference.h"
#include "agent/adaptation/flydelta/flydelta-activation.h"

#include "log.h"
#include "common.h"
#include "server-context.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <utility>

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
        session.templates,
        session.capabilities.image,
        session.capabilities.audio);
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

namespace {

bool run_server_flydelta_arm_batch(
        const std::shared_ptr<common_agent_server_context_host> & host,
        const common_agent_server_flydelta_binding & binding,
        const bool native_batch,
        const common_flydelta_arm_batch_request & request,
        common_flydelta_arm_batch_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!host || !binding.prepare_arm || !binding.finalize_arm) {
        error = "resident FlyDelta host binding is incomplete";
        return false;
    }
    if (!common_flydelta_arm_batch_request_validate(request, error)) {
        return false;
    }

    common_agent_inference_session session;
    if (!host->build_inference_session(session, error) || !session.inference) {
        if (error.empty()) error = "resident FlyDelta host could not build inference session";
        return false;
    }

    std::vector<common_agent_generation_request> generation_requests;
    generation_requests.reserve(request.arms.size());
    for (const auto & arm : request.arms) {
        common_agent_generation_request generation_request;
        if (!binding.prepare_arm(arm, generation_request, error)) {
            if (error.empty()) error = "resident FlyDelta host failed to prepare an arm";
            return false;
        }
        generation_requests.push_back(std::move(generation_request));
    }

    const auto generation_start = std::chrono::steady_clock::now();
    std::vector<common_agent_generation_result> generation_results;
    if (!session.inference->generate_batch(generation_requests, generation_results) ||
            generation_results.size() != request.arms.size()) {
        error = "resident FlyDelta host batch generation returned incomplete results";
        return false;
    }
    const float generation_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - generation_start).count());

    result.schema_version = request.schema_version;
    result.arms.reserve(request.arms.size());
    for (size_t index = 0; index < request.arms.size(); ++index) {
        common_flydelta_arm_result arm_result;
        if (!binding.finalize_arm(
                request.arms[index], generation_results[index], arm_result, error)) {
            if (error.empty()) error = "resident FlyDelta host failed to finalize an arm";
            return false;
        }
        if (arm_result.arm_id.empty()) {
            arm_result.arm_id = request.arms[index].arm_id;
        }
        if (arm_result.arm_id != request.arms[index].arm_id) {
            error = "resident FlyDelta host returned an arm identity mismatch";
            return false;
        }
        arm_result.execution_metrics.available = true;
        arm_result.execution_metrics.model_ms = generation_ms;
        arm_result.execution_metrics.generation_ms = generation_ms;
        arm_result.execution_metrics.batched_execution_used =
            request.arms.size() > 1 && native_batch;
        if (generation_requests[index].flydelta_activation &&
                generation_requests[index].flydelta_activation->sparse_overlay.enabled) {
            const auto & sparse = generation_requests[index].flydelta_activation->sparse_overlay;
            arm_result.execution_metrics.overlay_bytes_to_device =
                sparse.layer_indices.size() * sizeof(uint32_t) +
                sparse.data.size() * sizeof(float);
        }
        if (generation_results[index].flydelta_capture) {
            arm_result.execution_metrics.capture_bytes_to_host =
                generation_results[index].flydelta_capture->values.size() * sizeof(float);
        }
        if (arm_result.execution_metrics.execution_path ==
                common_flydelta_arm_execution_metrics::path::unknown) {
            arm_result.execution_metrics.execution_path =
                request.arms.size() > 1 && native_batch
                    ? common_flydelta_arm_execution_metrics::path::backend_batch
                    : request.arms.size() > 1
                        ? common_flydelta_arm_execution_metrics::path::scalar_fallback
                        : common_flydelta_arm_execution_metrics::path::scalar;
            if (request.arms.size() > 1 && !native_batch) {
                arm_result.execution_metrics.fallback_reason =
                    "per_sequence_cvec_batch_not_enabled";
            }
        }
        result.arms.push_back(std::move(arm_result));
    }
    return common_flydelta_arm_batch_result_validate(result, request, error);
}

} // namespace

std::shared_ptr<const common_flydelta_model_host>
common_agent_server_context_host_make_flydelta_model_host(
        std::shared_ptr<common_agent_server_context_host> host,
        common_agent_server_flydelta_binding binding,
        std::string & error) {
    error.clear();
    if (!host || !binding.prepare_arm || !binding.finalize_arm ||
            !binding.register_evaluator) {
        error = "resident FlyDelta host binding requires arm and evaluator callbacks";
        return {};
    }

    const bool native_batch = host->server().per_sequence_cvec_batch_enabled() &&
        host->context_key().n_parallel > 1;
    auto model_host = std::make_shared<common_flydelta_model_host>();
    model_host->capabilities = binding.primitives;
    model_host->capabilities.bounded_arm_batch = native_batch;
    model_host->run_bounded_arm_batch = [host, binding, native_batch](
            const common_flydelta_arm_batch_request & request,
            common_flydelta_arm_batch_result & result,
            std::string & callback_error) {
        return run_server_flydelta_arm_batch(
            host, binding, native_batch, request, result, callback_error);
    };
    model_host->run_bounded_arm = [host, binding](
            const common_flydelta_arm_request & request,
            common_flydelta_arm_result & result,
            std::string & callback_error) {
        common_flydelta_arm_batch_request batch;
        batch.arms.push_back(request);
        common_flydelta_arm_batch_result batch_result;
        if (!run_server_flydelta_arm_batch(
                host, binding, false, batch, batch_result, callback_error) ||
                batch_result.arms.size() != 1) {
            return false;
        }
        result = std::move(batch_result.arms.front());
        return true;
    };
    model_host->register_evaluator = std::move(binding.register_evaluator);
    error.clear();
    return model_host;
}
