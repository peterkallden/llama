#pragma once

#include "memory/memory-retrieval.h"

#ifdef LLAMA_MEMORY_POC_USE_AGENT_TOOLS
#include "plan/plan-store.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct args {
    std::string command;
    std::string backend = "auto";
    std::string memory_db;
    std::string id;
    std::string kind = "episode";
    std::string content;
    std::string query;
    std::string from;
    std::string relation;
    std::string to;
    std::string model;
    std::string embedding_model;
    std::string prompt;
    std::string memory_scope = "session";
    std::string memory_namespace = "local";
    std::string memory_session = "default";
    std::string memory_project;
    std::string memory_turn;
    std::vector<float> embedding;
    float importance = 0.5f;
    float confidence = 0.5f;
    float weight = 1.0f;
    size_t limit = 8;
    size_t memory_token_budget = 768;
    size_t max_tool_rounds = 1;
    int n_predict = 128;
    int n_gpu_layers = 99;
    bool record_episode = false;
    bool enable_memory_search_tool = false;
    bool enable_memory_remember_tool = false;
    std::string tool_profile;
    std::string mcp_tool_command;
    std::vector<std::string> mcp_tool_args;
    std::string mcp_tool_server_name = "mcp";
    std::string mcp_tool_prefix;
    std::string resource_blob_backend = "auto";
    std::string resource_blob_root;
    std::string resource_metadata_backend = "auto";
    std::string resource_metadata_db;
    std::string planning_mode = "off";
    std::string reflection_mode = "off";
    std::string agent_profile = "default";
    std::string plan_scope = "turn";
    std::string plan_backend = "auto";
    std::string plan_db;
    std::string plan_id;
    std::string agent_plan = "off";
    std::string repository_root;
    std::string agent_bootstrap = "none";
    std::string agent_import;
    std::string agent_export;
    std::string agent_blueprint;
    bool plan_show_summary = false;
    bool agent_trace = false;
    bool memory_global_opt_in = false;
    bool tool_profile_explicit = false;
    bool planning_mode_explicit = false;
    bool reflection_mode_explicit = false;
    bool memory_learn_explicit = false;
    bool agent_profile_explicit = false;
    std::string memory_learn = "off";
    bool memory_learn_show_candidate = false;
    float memory_learn_min_confidence = 0.75f;
    float memory_learn_min_reuse = 0.65f;
    std::string agent_inference_backend = "cli";
    size_t turn_timeout_ms = 0;
    uint32_t inference_step_timeout_ms = 0;
    uint32_t tool_timeout_ms = 1000;
    uint32_t mcp_connect_timeout_ms = 0;
    uint32_t mcp_request_timeout_ms = 0;
    uint32_t mcp_shutdown_timeout_ms = 0;
};

bool resolve_agent_profile(args & a, std::string & error);
std::unique_ptr<common_memory_store> make_memory_store(const args & a, std::string & error);
bool open_memory_store(common_memory_store & store, const args & a, std::string & error);

#ifdef LLAMA_MEMORY_POC_USE_AGENT_TOOLS
std::unique_ptr<common_plan_store> make_plan_store(const args & a, std::string & error);
#endif
