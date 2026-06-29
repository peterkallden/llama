#pragma once

#include "../common/cli-config.h"

#include "agent/agent-inference.h"
#include "agent/agent-runtime.h"
#include "agent/memory-learning.h"

#include "chat.h"
#include "llama.h"
#include "memory/memory-store.h"

#include <memory>
#include <string>
#include <vector>

enum class agent_inference_backend {
    cli,
    server_context,
};

bool parse_agent_inference_backend(const std::string & value, agent_inference_backend & backend);

struct common_agent_inference_session {
    std::shared_ptr<void> owner;
    std::unique_ptr<common_agent_inference> inference;
};

bool make_agent_inference_session(
    const args & options,
    agent_inference_backend backend,
    llama_model * model,
    const common_chat_templates * templates,
    common_agent_inference_session & session,
    std::string & error);

struct common_agent_runtime_assembly {
    std::unique_ptr<common_planner> planner;
    std::unique_ptr<common_action_executor> executor;
    std::unique_ptr<common_reflection_engine> reflector;
    std::unique_ptr<common_memory_candidate_extractor> candidate_extractor;
    std::unique_ptr<common_memory_post_turn_learner> memory_learner;
    std::unique_ptr<common_agent_runtime> runtime;
};

common_agent_runtime_assembly make_agent_runtime_assembly(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    common_agent_inference & inference,
    const args & options,
    const std::vector<common_chat_tool> & tools,
    const common_tool_registry * tool_registry);
