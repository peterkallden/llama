#pragma once

#include "tools/agent/cli/agent-cli-options.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/data-store.h"
#include "agent/adaptation/learning-cause-classifier.h"

#include "../host/agent-host-mcp-provider-config.h"
#include "../host/agent-host-openapi-provider-config.h"
#include "../runtime/agent-plan-orchestration.h"
#include "../runtime/agent-runtime-assembly.h"
#include "../runtime/agent-runtime-execution.h"
#include "../runtime/agent-runtime-resident.h"
#include "agent-daemon-service.h"
#include "../host/agent-diagnostics-config.h"
#include "../../../common/agent/protocol/agent-jsonl.h"

#include <cstdint>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <memory>
#include <map>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

struct daemon_options {
    std::string config_path;
    std::string model;
    std::string mmproj;
    std::string embedding_model;
    std::string backend = "auto";
    std::string memory_db;
    std::string plan_backend = "auto";
    std::string plan_db;
    std::string data_backend = "auto";
    std::string data_db;
    std::string default_mode = "chat";
    std::string thinking_mode = "reflective";
    int max_reflection_rounds = 1;
    int max_plan_revisions = 0;
    size_t max_research_iterations = 0;
    int n_predict = 64;
    int n_threads = 2;
    int context_size = 3072;
    common_agent_context_budget_config context_budgets;
    size_t max_continuations = 2;
    int n_gpu_layers = 0;
    std::string memory_learn = "off";
    std::string agent_plan = "off";
    std::string agent_blueprint = "off";
    std::string tool_profile;
    std::map<std::string, std::vector<std::string>> tool_capabilities;
    std::map<std::string, std::string> tool_family_descriptions;
    std::map<std::string, common_tool_profile> tool_profiles;
    common_agent_sandbox_host_config sandbox;
    std::string repository_root;
    agent_host_diagnostics_config diagnostics;
    std::string mcp_tool_command;
    std::vector<std::string> mcp_tool_args;
    std::string mcp_tool_server_name = "mcp";
    std::string mcp_tool_prefix;
    std::vector<agent_host_mcp_provider_config> mcp_providers;
    std::vector<agent_host_openapi_provider_config> openapi_providers;
    std::string resource_blob_backend = "auto";
    std::string resource_blob_root;
    std::string resource_metadata_backend = "auto";
    std::string resource_metadata_db;
    std::map<std::string, agent_resource_processor_execution_policy> resource_processor_policies;
    bool memory_learn_show_candidate = false;
    float memory_learn_min_confidence = 0.75f;
    float memory_learn_min_reuse = 0.65f;
    bool plan_show_summary = false;
    bool agent_trace = false;
    bool adaptation_capture = false;
    bool adaptation_collection_allowed = false;
    size_t adaptation_max_evidence = 16;
    std::string adaptation_transaction_path;
    std::set<std::string> adaptation_stable_model_facing_tools;
    size_t max_tool_rounds = 0;
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
    bool http_enabled = false;
    bool http_agent_tools_enabled = false;
    size_t http_max_delegation_depth = 1;
    std::string http_listen_address = "127.0.0.1";
    int http_port = 0;
    std::string http_path = "/mcp";
    std::string http_allowed_origin;
    std::string http_token_env;
    std::string http_bearer_token;
    std::vector<agent_host_mcp_inbound_token_config> http_token_profiles;
    std::string http_authorization_mode = "opaque";
    std::string http_jwt_issuer;
    std::string http_jwt_audience;
    std::string http_jwt_jwks_uri;
    std::vector<std::string> http_jwt_allowed_algorithms = {"RS256"};
    std::vector<std::string> http_jwt_required_scopes;
    std::string http_jwt_tool_profile;
    std::vector<std::string> http_jwt_allowed_tools;
    bool http_jwt_allow_writes = false;
    bool http_jwt_allow_admin = false;
    size_t http_max_body_bytes = 1024 * 1024;
    size_t http_max_result_bytes = 1024 * 1024;
    bool tcp_enabled = false;
    std::string tcp_listen_address = "127.0.0.1";
    int tcp_port = 0;
    size_t tcp_max_line_bytes = 1024 * 1024;
    size_t tcp_idle_timeout_seconds = 300;
    bool unix_socket_enabled = false;
    std::string unix_socket_path;
    int unix_socket_mode = 0660;
};

class common_agent_daemon_dispatcher;

struct agent_daemon_foreground_request {
    common_agent_daemon_command command;
};

struct agent_daemon_foreground_response {
    common_agent_daemon_command_result result;
    bool shutdown_after = false;
};

using agent_daemon_jsonl_stream = common_agent_jsonl_stream;

bool parse_mode(
    const std::string & value,
    common_agent_runtime_host_mode & mode);

bool parse_agent_daemon_args(int argc, char ** argv, daemon_options & options);
void print_agent_daemon_usage(const char * argv0);

bool initialize_agent_daemon_environment(
    const daemon_options & options,
    common_agent_daemon_runtime & runtime,
    std::string & error);

void configure_agent_daemon_provider_probe(
    const daemon_options & options,
    common_agent_daemon_runtime & runtime);

bool resolve_agent_daemon_tooling(
    const daemon_options & options,
    const common_agent_runtime_resident_runtime * runtime,
    const common_agent_runtime_session_host_turn_request & request,
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    agent_resource_store * resource_store,
    common_agent_runtime_tooling & tooling,
    std::string & error,
    common_agent_data_store * data_store = nullptr,
    std::optional<bool> allow_policy_gated_writes = std::nullopt);

bool parse_agent_daemon_command(
    const nlohmann::ordered_json & parsed,
    const daemon_options & options,
    common_agent_runtime_host_mode default_mode,
    common_agent_daemon_command & command,
    std::string & error);

nlohmann::ordered_json make_agent_daemon_ready_response(const daemon_options & options);
nlohmann::ordered_json make_agent_daemon_error_response(const std::string & error);
nlohmann::ordered_json make_agent_daemon_command_response(
    const common_agent_daemon_command_result & result);

// JSONL keeps lifecycle events on their separate event-message channel. The
// terminal response is deliberately a result-only projection.
nlohmann::ordered_json make_agent_daemon_jsonl_command_response(
    const common_agent_daemon_command_result & result);

bool parse_agent_daemon_foreground_request(
    const nlohmann::ordered_json & parsed,
    const daemon_options & options,
    common_agent_runtime_host_mode default_mode,
    agent_daemon_foreground_request & request,
    std::string & error);

bool execute_agent_daemon_foreground_request(
    const agent_daemon_foreground_request & request,
    common_agent_daemon_dispatcher & dispatcher,
    agent_daemon_foreground_response & response,
    std::string & error);

bool run_agent_daemon_jsonl_adapter(
    FILE * input,
    FILE * output,
    const daemon_options & options,
    const std::shared_ptr<common_agent_daemon_config_store> & config_store,
    common_agent_daemon_dispatcher & dispatcher,
    std::string & error);

bool run_agent_daemon_jsonl_stream(
    agent_daemon_jsonl_stream & stream,
    const daemon_options & options,
    const std::shared_ptr<common_agent_daemon_config_store> & config_store,
    common_agent_daemon_dispatcher & dispatcher,
    const std::function<bool(
        nlohmann::ordered_json &,
        std::string &)> & prepare_request,
    std::string & error);
