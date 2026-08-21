#pragma once

#include "memory/memory-retrieval.h"
#include "agent/tool-catalog.h"
#include "agent/agent-context-budgets.h"
#include "agent/sandbox-host-config.h"

#ifdef LLAMA_MEMORY_POC_USE_AGENT_TOOLS
#include "plan/plan-store.h"
#endif

#include <cstdint>
#include <memory>
#include <map>
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
    std::string mmproj;
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
    size_t max_tool_rounds = 16;
    int n_predict = 128;
    int n_threads = 2;
    // Leave enough room for the compact plan, verified observations and a
    // bounded reflection JSON response in the default CLI/daemon path.
    int context_size = 3072;
    common_agent_context_budget_config context_budgets;
    size_t max_continuations = 2;
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
    std::map<std::string, agent_resource_processor_execution_policy> resource_processor_policies;
    std::vector<std::string> resource_paths;
    std::string resource_mime_type;
    bool agent_runtime = false;
    std::string thinking_mode = "reflective";
    int max_reflection_rounds = 1;
    int max_plan_revisions = 0;
    size_t max_research_iterations = 0;
    std::string agent_profile = "default";
    std::string plan_scope = "turn";
    std::string plan_backend = "auto";
    std::string plan_db;
    std::string data_backend = "auto";
    std::string data_db;
    std::string plan_id;
    std::string agent_plan = "off";
    std::string repository_root;
    std::string agent_bootstrap = "none";
    std::string agent_import;
    std::string agent_export;
    std::string agent_blueprint;
    bool plan_show_summary = false;
    bool include_summary = false;
    bool agent_trace = false;
    bool generation_trace = false;
    bool require_tool_execution = false;
    bool memory_global_opt_in = false;
    bool tool_profile_explicit = false;
    std::map<std::string, std::vector<std::string>> tool_capabilities;
    std::map<std::string, common_tool_profile> tool_profiles;
    common_agent_sandbox_host_config sandbox;
    bool thinking_mode_explicit = false;
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
