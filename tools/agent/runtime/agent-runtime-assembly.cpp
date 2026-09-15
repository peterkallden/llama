#include "agent-runtime-assembly.h"

#include "../cli/agent-cli-runtime.h"
#include "../tooling/agent-tool-runtime-adapter.h"
#include "../adaptation/agent-learning-lifecycle-store.h"

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
    config.enable_adaptation_capture = build_config.enable_adaptation_capture;
    config.adaptation_config = std::move(build_config.adaptation_config);
    config.adaptation_transaction_backend = std::move(build_config.adaptation_transaction_backend);
    config.adaptation_transaction_path = std::move(build_config.adaptation_transaction_path);
    config.enable_flydelta_candidate_lifecycle = build_config.enable_flydelta_candidate_lifecycle;
    config.flydelta_lifecycle_backend = std::move(build_config.flydelta_lifecycle_backend);
    config.flydelta_lifecycle_path = std::move(build_config.flydelta_lifecycle_path);
    config.enable_flydelta_capture_candidates = build_config.enable_flydelta_capture_candidates;
    config.flydelta_model_profile_fingerprint = std::move(build_config.flydelta_model_profile_fingerprint);
    config.flydelta_capture_layout_revision = std::move(build_config.flydelta_capture_layout_revision);
    config.flydelta_max_capture_candidates = build_config.flydelta_max_capture_candidates;
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

    if (runtime_config.enable_adaptation_capture) {
        std::string store_error;
        assembly.adaptation_store = make_agent_learning_transaction_store(
            runtime_config.adaptation_transaction_backend,
            runtime_config.adaptation_transaction_path,
            store_error);
        if (!assembly.adaptation_store) assembly.adaptation_error = std::move(store_error);
        if (assembly.adaptation_store) {
            auto adaptation_config = runtime_config.adaptation_config;
            if (runtime_config.enable_flydelta_capture_candidates) {
                assembly.flydelta_capture_collector = std::make_unique<common_flydelta_capture_candidate_collector>(
                    runtime_config.flydelta_model_profile_fingerprint,
                    runtime_config.flydelta_capture_layout_revision,
                    runtime_config.flydelta_max_capture_candidates);
                if (runtime_config.enable_flydelta_candidate_lifecycle) {
                    std::string lifecycle_error;
                    assembly.flydelta_lifecycle_store = make_agent_learning_lifecycle_store(
                        runtime_config.flydelta_lifecycle_backend,
                        runtime_config.flydelta_lifecycle_path,
                        lifecycle_error);
                    if (!assembly.flydelta_lifecycle_store) {
                        assembly.adaptation_error = std::move(lifecycle_error);
                    }
                }
                if (assembly.flydelta_lifecycle_store) {
                    assembly.flydelta_runtime_candidate_observer =
                        std::make_unique<common_flydelta_runtime_candidate_observer>(
                            *assembly.flydelta_capture_collector,
                            assembly.flydelta_lifecycle_store.get());
                    adaptation_config.source_observer =
                        assembly.flydelta_runtime_candidate_observer->source_observer();
                } else {
                    adaptation_config.source_observer = assembly.flydelta_capture_collector->source_observer();
                }
            }
            assembly.adaptation_observer = std::make_unique<common_learning_transaction_observer>(
                *assembly.adaptation_store, std::move(adaptation_config));
        }
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
    if (assembly.adaptation_observer) {
        assembly.runtime->set_adaptation_observer(assembly.adaptation_observer.get());
    }
    return assembly;
}
