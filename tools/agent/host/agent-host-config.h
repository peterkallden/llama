#pragma once

#include "tools/agent/cli/agent-cli-options.h"
#include "../host/agent-host-mcp-provider-config.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/sandbox/sandbox-host-config.h"
#include "agent/sandbox/sandbox-policy.h"
#include "agent/workspace-contract.h"
#include "agent-diagnostics-config.h"
#include "agent-host-openapi-provider-config.h"
#include "resource/resource-contract.h"
#include "agent/adaptation/learning-cause-classifier.h"
#include "agent/runtime/model-catalog.h"

#include <cstdint>
#include <nlohmann/json.hpp>

#include <string>
#include <map>
#include <set>
#include <vector>

struct daemon_options;

using agent_host_sandbox_config = common_agent_sandbox_host_config;

struct agent_host_config {
    int schema_version = 1;

    std::string model_backend = "server-context";
    std::string model_path;
    std::string mmproj_path;
    std::string embedding_model;
    // Optional catalog profile override. Empty means models.routing.default_profile.
    std::string model_profile;
    // Optional catalog-based model configuration. The legacy model fields
    // remain the effective runtime input until profile routing is enabled.
    common_agent_model_catalog model_catalog;

    int runtime_context_size = 3072;
    int n_predict = 64;
    int n_threads = 2;
    common_agent_context_budget_config context_budgets;
    size_t max_continuations = 2;
    int n_gpu_layers = 0;
    std::string default_mode = "chat";
    std::string thinking_mode = "reflective";
    int max_reflection_rounds = 1;
    int max_plan_revisions = 0;
    size_t max_research_iterations = 0;
    std::string memory_learn = "off";
    std::string agent_plan = "off";
    std::string agent_blueprint = "off";
    bool memory_learn_show_candidate = false;
    float memory_learn_min_confidence = 0.75f;
    float memory_learn_min_reuse = 0.65f;
    bool plan_show_summary = false;
    bool agent_trace = false;
    bool adaptation_capture = false;
    bool adaptation_collection_allowed = false;
    size_t adaptation_max_evidence = 16;
    std::string adaptation_transaction_backend = "auto";
    std::string adaptation_transaction_path;
    std::set<std::string> adaptation_stable_model_facing_tools;

    std::string memory_backend = "auto";
    std::string memory_db;
    std::string plan_backend = "auto";
    std::string plan_db;
    std::string data_backend = "auto";
    std::string data_db;

    std::string resource_blob_backend = "auto";
    std::string resource_blob_root;
    std::string resource_metadata_backend = "auto";
    std::string resource_metadata_db;
    std::map<std::string, agent_resource_processor_execution_policy> resource_processor_policies;

    std::string tool_profile;
    std::map<std::string, std::vector<std::string>> tool_capabilities;
    std::map<std::string, std::string> tool_family_descriptions;
    std::map<std::string, common_tool_profile> tool_profiles;
    agent_host_sandbox_config sandbox;
    std::string repository_root;
    agent_host_diagnostics_config diagnostics;
    std::vector<agent_host_mcp_provider_config> mcp_providers;
    std::vector<agent_host_openapi_provider_config> openapi_providers;
    // Optional directory of provider-object JSON fragments. The loader
    // expands these into the provider vectors before validation/runtime use.
    std::string tools_include_dir;
    // Provider ids loaded from tools_include_dir are omitted by the canonical
    // serializer so an effective config cannot accidentally duplicate them.
    std::set<std::string> included_provider_ids;
    bool inbound_mcp_enabled = false;
    std::string inbound_mcp_listen_address = "127.0.0.1";
    int inbound_mcp_port = 0;
    std::string inbound_mcp_path = "/mcp";
    std::string inbound_mcp_allowed_origin;
    size_t inbound_mcp_max_body_bytes = 1024 * 1024;
    size_t inbound_mcp_max_result_bytes = 1024 * 1024;
    bool inbound_mcp_agent_tools_enabled = false;
    size_t inbound_mcp_max_delegation_depth = 1;
    std::vector<agent_host_mcp_inbound_token_config> inbound_mcp_tokens;
    std::string inbound_mcp_authorization_mode = "opaque";
    std::string inbound_mcp_jwt_issuer;
    std::string inbound_mcp_jwt_audience;
    std::string inbound_mcp_jwt_jwks_uri;
    std::vector<std::string> inbound_mcp_jwt_allowed_algorithms = {"RS256"};
    std::vector<std::string> inbound_mcp_jwt_required_scopes;
    std::string inbound_mcp_jwt_tool_profile;
    std::vector<std::string> inbound_mcp_jwt_allowed_tools;
    bool inbound_mcp_jwt_allow_writes = false;
    bool inbound_mcp_jwt_allow_admin = false;
    bool jsonl_tcp_enabled = false;
    std::string jsonl_tcp_listen_address = "127.0.0.1";
    int jsonl_tcp_port = 0;
    size_t jsonl_tcp_max_line_bytes = 1024 * 1024;
    size_t jsonl_tcp_idle_timeout_seconds = 300;
    bool jsonl_unix_socket_enabled = false;
    std::string jsonl_unix_socket_path;
    int jsonl_unix_socket_mode = 0660;

    size_t queue_capacity = 8;
    size_t worker_count = 1;
    size_t inference_max_active = 1;
    size_t max_turn_seconds = 0;
    size_t turn_timeout_ms = 0;
    uint32_t inference_step_timeout_ms = 0;
    uint32_t tool_timeout_ms = 1000;
    uint32_t mcp_connect_timeout_ms = 0;
    uint32_t mcp_request_timeout_ms = 0;
    uint32_t mcp_shutdown_timeout_ms = 0;
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

// Resolve an implicit host configuration path. An explicit path has highest
// priority; an empty result means that no implicit configuration exists.
bool resolve_agent_host_config_path(
    const std::string & explicit_path,
    std::string & path,
    std::string & error);

nlohmann::ordered_json agent_host_config_to_json(
    const agent_host_config & config);

bool validate_agent_host_config(
        const agent_host_config & config,
        std::string & error);

// Resolve the configured generation profile without loading it. Legacy
// model.path configurations return false with a descriptive unavailable error.
bool resolve_agent_host_model_selection(
        const agent_host_config & config,
        common_agent_model_selection & selection,
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
