#pragma once

#include "../common/cli-config.h"
#include "agent-host-mcp-provider-config.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

struct daemon_options;

struct agent_host_config {
    int schema_version = 1;

    std::string model_backend = "server-context";
    std::string model_path;
    std::string embedding_model;

    int runtime_context_size = 0;
    int n_predict = 64;
    int n_gpu_layers = 0;
    std::string default_mode = "chat";
    std::string planning_mode = "off";
    std::string reflection_mode = "off";
    std::string memory_learn = "off";
    std::string agent_plan = "off";
    bool memory_learn_show_candidate = false;
    float memory_learn_min_confidence = 0.75f;
    float memory_learn_min_reuse = 0.65f;
    bool plan_show_summary = false;
    bool agent_trace = false;

    std::string memory_backend = "auto";
    std::string memory_db;
    std::string plan_backend = "auto";
    std::string plan_db;

    std::string resource_blob_backend = "auto";
    std::string resource_blob_root;
    std::string resource_metadata_backend = "auto";
    std::string resource_metadata_db;

    std::string tool_profile;
    std::string repository_root;
    std::vector<agent_host_mcp_provider_config> mcp_providers;

    size_t queue_capacity = 8;
    size_t max_turn_seconds = 0;
    size_t max_tool_rounds = 0;
};

bool parse_agent_host_config_json(
    const nlohmann::ordered_json & value,
    agent_host_config & config,
    std::string & error);

bool load_agent_host_config(
    const std::string & path,
    agent_host_config & config,
    std::string & error);

nlohmann::ordered_json agent_host_config_to_json(
    const agent_host_config & config);

bool validate_agent_host_config(
    const agent_host_config & config,
    std::string & error);

void apply_agent_host_config_to_daemon_options(
    const agent_host_config & config,
    daemon_options & options);

void apply_agent_host_config_to_args(
    const agent_host_config & config,
    args & options);

bool select_agent_host_stdio_mcp_provider(
    const agent_host_config & config,
    agent_host_mcp_provider_config & provider,
    std::string & error);
