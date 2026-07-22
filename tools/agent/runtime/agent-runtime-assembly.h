#pragma once

#include "../tooling/agent-tool-provider.h"
#include "agent/agent-inference.h"
#include "agent/agent-context-budgets.h"
#include "agent/agent-runtime.h"
#include "agent/memory-learning.h"

#include "chat.h"
#include "llama.h"
#include "memory/memory-store.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

enum class agent_inference_backend {
    cli,
    server_context,
};

bool parse_agent_inference_backend(const std::string & value, agent_inference_backend & backend);

struct common_agent_generation_config {
    int n_predict = 0;
    common_agent_context_budget_config context_budgets;
};

struct common_agent_inference_options {
    std::string model;
    int n_predict = -1;
    int n_gpu_layers = 0;
    bool fit_params = true;
};

struct common_agent_runtime_config {
    common_agent_generation_config generation_config;
    common_agent_context_budget_config context_budgets;
    bool enable_memory_learning = false;
    common_memory_learning_config memory_learning_config;
    std::function<bool(const std::string & text, std::vector<float> & embedding, std::string & error)> embed_memory;
};

struct common_agent_runtime_build_config {
    common_agent_generation_config generation_config;
    common_agent_context_budget_config context_budgets;
    bool enable_memory_learning = false;
    common_memory_learning_config memory_learning_config;
    std::function<bool(const std::string & text, std::vector<float> & embedding, std::string & error)> embed_memory;
};

common_agent_inference_options make_agent_inference_options(
    common_agent_inference_options config);

common_agent_runtime_config make_agent_runtime_config(
    common_agent_runtime_build_config config);

struct common_agent_inference_session {
    agent_inference_backend backend = agent_inference_backend::cli;
    std::shared_ptr<void> keepalive;
    llama_model * model = nullptr;
    const common_chat_templates * templates = nullptr;
    std::unique_ptr<common_agent_inference> inference;
};

struct common_agent_runtime_assembly {
    std::unique_ptr<common_planner> planner;
    std::unique_ptr<common_action_executor> executor;
    std::unique_ptr<common_reflection_engine> reflector;
    std::unique_ptr<common_memory_candidate_extractor> candidate_extractor;
    std::unique_ptr<common_memory_post_turn_learner> memory_learner;
    std::unique_ptr<common_agent_tool_runtime> tool_runtime;
    std::unique_ptr<common_agent_runtime> runtime;
};

common_agent_runtime_assembly make_agent_runtime_assembly(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    common_agent_inference & inference,
    const common_agent_runtime_config & runtime_config,
    const std::vector<common_chat_tool> & tools,
    agent_tool_view * tool_view);
