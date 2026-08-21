#include "agent-runtime-assembly.h"

#include "../cli/agent-cli-runtime.h"
#include "../tooling/agent-tool-runtime-adapter.h"

#include <algorithm>

common_agent_inference_options make_agent_inference_options(common_agent_inference_options config) {
    return config;
}

common_agent_runtime_config make_agent_runtime_config(common_agent_runtime_build_config build_config) {
    common_agent_runtime_config config;
    config.generation_config = std::move(build_config.generation_config);
    config.context_budgets = build_config.context_budgets;
    config.generation_config.context_budgets = config.context_budgets;
    config.max_continuations = build_config.max_continuations;
    config.context_token_estimator = std::move(build_config.context_token_estimator);
    config.enable_memory_learning = build_config.enable_memory_learning;
    config.memory_learning_config = std::move(build_config.memory_learning_config);
    config.embed_memory = std::move(build_config.embed_memory);
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
    agent_tool_view * tool_view) {
    common_agent_runtime_assembly assembly;
    assembly.planner = make_llama_cli_planner(inference, runtime_config.generation_config, tools);
    assembly.executor = make_llama_cli_action_executor(inference, runtime_config.generation_config);
    assembly.reflector = make_llama_cli_reflection_engine(
        inference, runtime_config.generation_config, tools);

    if (runtime_config.enable_memory_learning) {
        assembly.candidate_extractor = make_llama_cli_memory_candidate_extractor(inference, runtime_config.generation_config);
        assembly.memory_learner = std::make_unique<common_memory_post_turn_learner>(
            memory_store,
            *assembly.candidate_extractor,
            runtime_config.embed_memory,
            runtime_config.memory_learning_config);
    }

    if (tool_view != nullptr) {
        assembly.tool_runtime = make_provider_agent_tool_runtime(*tool_view);
    }

    assembly.runtime = std::make_unique<common_agent_runtime>(
        plan_store,
        *assembly.planner,
        *assembly.executor,
        *assembly.reflector,
        assembly.tool_runtime.get(),
        assembly.memory_learner.get(),
        nullptr,
        runtime_config.context_budgets,
        runtime_config.generation_config.context_size_tokens,
        static_cast<size_t>(std::max(0, runtime_config.generation_config.n_predict)),
        runtime_config.context_token_estimator);
    return assembly;
}
