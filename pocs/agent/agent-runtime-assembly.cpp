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
    std::thread loop;
    bool running = false;

    ~agent_resident_inference_host() {
        stop();
    }

    bool start(const args & options, std::string & error) {
        common_params params;
        params.model.path = options.model;
        params.n_predict = options.n_predict;
        params.n_gpu_layers = options.n_gpu_layers;
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
    const args & options,
    agent_inference_backend backend,
    llama_model * model,
    const common_chat_templates * templates,
    common_agent_inference_session & session,
    std::string & error) {
    session = {};
    if (backend == agent_inference_backend::server_context) {
        auto host = std::make_shared<agent_resident_inference_host>();
        if (!host->start(options, error)) {
            return false;
        }
        auto meta = host->server.get_meta();
        session.inference = make_server_context_agent_inference(
            host->server,
            meta.logit_bias_eog,
            meta.chat_params.tmpls.get());
        session.owner = std::move(host);
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
    const args & options,
    const std::vector<common_chat_tool> & tools,
    const common_tool_registry * tool_registry) {
    common_agent_runtime_assembly assembly;
    assembly.planner = make_llama_cli_planner(inference, options, tools);
    assembly.executor = make_llama_cli_action_executor(inference, options);
    assembly.reflector = make_llama_cli_reflection_engine(inference, options);

    if (options.memory_learn == "post-turn") {
        assembly.candidate_extractor = make_llama_cli_memory_candidate_extractor(inference, options);
        common_memory_learning_config learning_config;
        learning_config.min_confidence = options.memory_learn_min_confidence;
        learning_config.min_expected_reuse = options.memory_learn_min_reuse;
        assembly.memory_learner = std::make_unique<common_memory_post_turn_learner>(
            memory_store,
            *assembly.candidate_extractor,
            [&options](const std::string & text, std::vector<float> & embedding, std::string & embedding_error) {
                return ensure_memory_cli_embedding(options, text, embedding, "memory candidate", embedding_error);
            },
            learning_config);
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
