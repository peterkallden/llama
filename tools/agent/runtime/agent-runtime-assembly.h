#pragma once

#include "../tooling/agent-tool-provider.h"
#include "agent/agent-inference.h"
#include "agent/agent-context-budgets.h"
#include "agent/agent-runtime.h"
#include "agent/learning/memory-learning.h"
#include "agent/runtime/agent-inference-contracts.h"

#include "chat.h"
#include "llama.h"
#include "memory/memory-store.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct common_agent_inference_capabilities {
    bool text = true;
    bool image = false;
    bool audio = false;
};

bool parse_agent_inference_backend(const std::string & value, agent_inference_backend & backend);

struct common_agent_generation_config {
    int n_predict = 0;
    int n_threads = 2;
    bool generation_trace = false;
    size_t context_size_tokens = 0;
    common_agent_context_budget_config context_budgets;
    // Optional host-side preflight that narrows the model-facing tool view by
    // generated family before the planner sees individual tool contracts.
    bool enable_tool_family_routing = false;
};

struct common_agent_runtime_config {
    common_agent_generation_config generation_config;
    common_agent_context_budget_config context_budgets;
    bool enable_memory_learning = false;
    common_memory_learning_config memory_learning_config;
    std::function<bool(const std::string & text, std::vector<float> & embedding, std::string & error)> embed_memory;
    // Bounded internal inference slices after a generation limit. Zero
    // disables automatic continuation for compatibility.
    size_t max_continuations = 2;
    common_agent_context_token_estimator context_token_estimator;
};

struct common_agent_runtime_build_config {
    common_agent_generation_config generation_config;
    common_agent_context_budget_config context_budgets;
    bool enable_memory_learning = false;
    common_memory_learning_config memory_learning_config;
    std::function<bool(const std::string & text, std::vector<float> & embedding, std::string & error)> embed_memory;
    size_t max_continuations = 2;
    common_agent_context_token_estimator context_token_estimator;
};

common_agent_inference_options make_agent_inference_options(
    common_agent_inference_options config);

common_agent_runtime_config make_agent_runtime_config(
    common_agent_runtime_build_config config);

struct common_agent_inference_session {
    agent_inference_backend backend = agent_inference_backend::cli;
    common_agent_inference_capabilities capabilities;
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
