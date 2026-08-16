#pragma once

#include "tools/agent/cli/agent-cli-options.h"

#include "../runtime/agent-runtime-chat-driver.h"
#include "../runtime/agent-runtime-execution.h"

#include <string>

enum class common_agent_runtime_host_mode {
    chat,
    agent,
};

struct common_agent_runtime_turn_request {
    common_agent_request request;
    common_agent_scope scope;
    common_agent_inference_options inference_options;
    common_agent_runtime_policy policy;
    common_agent_runtime_config runtime_config;
    common_agent_orchestration_config orchestration_config;
    common_agent_generation_options generation_options;
    common_memory_scope memory_scope = common_memory_scope::session;
    bool memory_enabled = false;
    std::string fallback_reason;
    common_agent_runtime_execution_control execution_control;
};

struct common_agent_runtime_resident_request_config {
    std::string prompt;
    std::string session_id;
    std::string namespace_id;
    std::string project_id;
    std::optional<common_memory_policy_pack> policy_pack;
    std::string model;
    int n_predict = 0;
    int n_gpu_layers = 0;
    bool fit_params = true;
    std::string inference_backend = "server-context";
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;
    int n_threads = 2;
    size_t context_size_tokens = 0;
    std::string mmproj;
};
