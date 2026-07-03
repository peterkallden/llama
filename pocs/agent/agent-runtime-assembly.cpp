#include "agent-runtime-assembly.h"

#include "../memory/memory-cli-memory.h"

#include "agent-cli-runtime.h"

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
