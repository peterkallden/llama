#include "agent-runtime-assembly.h"

#include "../memory/memory-cli-memory.h"

#include "agent-cli-inference.h"
#include "agent-cli-runtime.h"

#include "common.h"
#include "server-context.h"

#include <thread>

namespace {

struct agent_resident_inference_host {
    server_context server;
    common_params params;
    std::thread loop;
    bool running = false;

    ~agent_resident_inference_host() {
        stop();
    }

    bool start(const common_agent_inference_options & options, std::string & error) {
        params = {};
        params.model.path = options.model;
        params.n_predict = options.n_predict;
        params.n_gpu_layers = options.n_gpu_layers;
        params.fit_params = options.fit_params;
        params.n_parallel = 1;
        params.n_sequences = 1;
        params.n_ctx = 0;
        postprocess_cpu_params(params.cpuparams, nullptr);
        postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

        if (!server.load_model(params)) {
            error = "failed to load resident server_context model: " + options.model;
            return false;
        }

        loop = std::thread([this]() {
            server.start_loop();
        });
        running = true;
        return true;
    }

    void stop() {
        if (!running) {
            return;
        }
        server.terminate();
        if (loop.joinable()) {
            loop.join();
        }
        running = false;
    }
};

} // namespace

common_agent_generation_config make_agent_generation_config(const args & options) {
    common_agent_generation_config config;
    config.n_predict = options.n_predict;
    return config;
}

common_agent_inference_options make_agent_inference_options(const args & options) {
    common_agent_inference_options config;
    config.model = options.model;
    config.n_predict = options.n_predict;
    config.n_gpu_layers = options.n_gpu_layers;
    config.fit_params = true;
    return config;
}

common_agent_runtime_config make_agent_runtime_config(const args & options) {
    common_agent_runtime_config config;
    config.generation_config = make_agent_generation_config(options);
    config.enable_memory_learning = options.memory_learn == "post-turn";
    config.memory_learning_config.min_confidence = options.memory_learn_min_confidence;
    config.memory_learning_config.min_expected_reuse = options.memory_learn_min_reuse;
    config.embed_memory = [&options](const std::string & text, std::vector<float> & embedding, std::string & error) {
        return ensure_memory_cli_embedding(options, text, embedding, "memory candidate", error);
    };
    return config;
}

bool parse_agent_inference_backend(const std::string & value, agent_inference_backend & backend) {
    if (value == "cli") {
        backend = agent_inference_backend::cli;
        return true;
    }
    if (value == "server-context") {
        backend = agent_inference_backend::server_context;
        return true;
    }
    return false;
}

bool make_agent_inference_session(
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
        auto host = std::make_shared<agent_resident_inference_host>();
        if (!host->start(options, error)) {
            return false;
        }
        auto meta = host->server.get_meta();
        session.templates = meta.chat_params.tmpls.get();
        session.inference = make_server_context_agent_inference(
            host->server,
            host->params,
            meta.logit_bias_eog,
            session.templates);
        session.keepalive = std::move(host);
        error.clear();
        return true;
    }

    session.inference = make_llama_cli_agent_inference(model, templates);
    error.clear();
    return true;
}

common_agent_runtime_assembly make_agent_runtime_assembly(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    common_agent_inference & inference,
    const common_agent_runtime_config & runtime_config,
    const std::vector<common_chat_tool> & tools,
    const common_tool_registry * tool_registry) {
    common_agent_runtime_assembly assembly;
    assembly.planner = make_llama_cli_planner(inference, runtime_config.generation_config, tools);
    assembly.executor = make_llama_cli_action_executor(inference, runtime_config.generation_config);
    assembly.reflector = make_llama_cli_reflection_engine(inference, runtime_config.generation_config);

    if (runtime_config.enable_memory_learning) {
        assembly.candidate_extractor = make_llama_cli_memory_candidate_extractor(inference, runtime_config.generation_config);
        assembly.memory_learner = std::make_unique<common_memory_post_turn_learner>(
            memory_store,
            *assembly.candidate_extractor,
            runtime_config.embed_memory,
            runtime_config.memory_learning_config);
    }

    assembly.runtime = std::make_unique<common_agent_runtime>(
        plan_store,
        *assembly.planner,
        *assembly.executor,
        *assembly.reflector,
        tool_registry,
        assembly.memory_learner.get());
    return assembly;
}
