#pragma once

#include "tools/agent/cli/agent-cli-options.h"

#include "../host/agent-host-mcp-provider-config.h"
#include "../host/agent-host-openapi-provider-config.h"
#include "../resource/agent-resource-store.h"
#include "../resource/agent-resource-processing-service.h"
#include "agent/sandbox/sandbox-host-config.h"
#include "agent/sandbox/sandbox-contract.h"
#include "agent/data-store.h"
#include "../host/agent-diagnostics-config.h"
#include "agent/workspace-manager.h"
#include "../runtime/agent-runtime-host.h"
#include "../runtime/agent-runtime-tooling.h"
#include "../tooling/agent-tool-provider.h"

#include <memory>
#include <map>
#include <string>
#include <vector>

struct common_agent_cli_tool_selection {
    common_agent_runtime_tooling tooling;
    std::unique_ptr<agent_tool_view> tool_view;
    std::vector<std::unique_ptr<agent_mcp_tool_client>> mcp_clients;
    std::vector<std::unique_ptr<agent_tool_provider>> openapi_providers;
    std::unique_ptr<agent_embedding_provider> embedding_provider;
    std::unique_ptr<agent_resource_store> owned_resource_store;
    std::vector<std::shared_ptr<agent_resource_processor>> resource_processors;
    std::shared_ptr<agent_resource_processor_registry> resource_processor_registry;
    std::shared_ptr<agent_resource_processing_service> resource_processing_service;
    std::unique_ptr<common_agent_data_store> owned_data_store;
    std::shared_ptr<common_agent_sandbox_runtime> sandbox_runtime;
    std::shared_ptr<common_agent_workspace_manager> workspace_manager;
};

struct agent_host_stdio_mcp_provider_request {
    std::string server_name = "mcp";
    std::string transport = "stdio";
    std::vector<std::string> command_line;
    std::string url;
    std::string bearer_token;
    std::vector<std::string> allowed_tools;
    uint32_t connect_timeout_ms = 5000;
    uint32_t request_timeout_ms = 30000;
    uint32_t shutdown_timeout_ms = 2000;
    size_t max_result_bytes = 1024 * 1024;
    std::string exposed_name_prefix;
};

struct agent_host_tool_selection_request {
    agent_tool_context tool_context;
    std::string repository_root;
    agent_resource_store_config resource_store_config;
    common_agent_data_store_config data_store_config;
    common_agent_data_store * data_store = nullptr;
    std::vector<agent_host_stdio_mcp_provider_request> mcp_providers;
    std::vector<agent_host_openapi_provider_config> openapi_providers;
    std::map<std::string, std::vector<std::string>> tool_capabilities;
    std::map<std::string, std::string> tool_family_descriptions;
    std::map<std::string, common_tool_profile> tool_profiles;
    common_agent_sandbox_host_config sandbox;
    std::map<std::string, agent_resource_processor_execution_policy> resource_processor_policies;
    agent_host_diagnostics_config diagnostics;
};

bool has_enabled_stdio_mcp_provider(
    const std::vector<agent_host_mcp_provider_config> & providers);

bool has_enabled_mcp_provider(
    const std::vector<agent_host_mcp_provider_config> & providers);

void append_configured_stdio_mcp_providers(
    const std::vector<agent_host_mcp_provider_config> & configured_providers,
    std::vector<agent_host_stdio_mcp_provider_request> & request_providers);

void append_legacy_stdio_mcp_provider(
    const std::string & command,
    const std::vector<std::string> & args,
    const std::string & server_name,
    const std::string & prefix,
    std::vector<agent_host_stdio_mcp_provider_request> & request_providers);

bool resolve_agent_host_tool_selection(
    common_memory_store & store,
    common_plan_store * plan_store,
    agent_resource_store * resource_store,
    std::string * current_plan_id,
    const std::string & tool_profile,
    const agent_host_tool_selection_request & request,
    const common_memory_query & query,
    agent_embedding_provider * embedding_provider,
    common_agent_cli_tool_selection & selection,
    std::string & error);

common_agent_runtime_host_post_run make_agent_cli_runtime_post_run(
    common_memory_store & store,
    const args & options,
    bool memory_enabled);

bool resolve_agent_cli_tool_selection(
    common_memory_store & store,
    common_plan_store * plan_store,
    agent_resource_store * resource_store,
    std::string * current_plan_id,
    const args & options,
    const common_memory_query & query,
    bool memory_enabled,
    common_agent_cli_tool_selection & selection,
    std::string & error);

common_agent_runtime_turn_request make_agent_cli_runtime_turn_request(
    const args & options,
    const common_agent_scope & scope,
    const common_agent_orchestration_config & orchestration_config,
    common_memory_scope memory_scope,
    bool memory_enabled,
    const std::string & fallback_reason,
    agent_embedding_provider * embedding_provider = nullptr,
    common_agent_request request = {},
    common_agent_generation_options generation_options = {},
    std::vector<common_agent_input_resource> input_resources = {});

common_agent_runtime_host_inputs make_agent_cli_runtime_host_chat_inputs(
    common_memory_store & store,
    args & options,
    const std::vector<common_chat_msg> & messages,
    common_memory_scope memory_scope,
    const std::vector<common_memory_hit> & memories,
    bool memory_enabled,
    const std::string & fallback_reason,
    const common_agent_runtime_tooling & tooling,
    agent_embedding_provider * embedding_provider,
    common_agent_runtime_host_post_run post_run,
    std::vector<common_agent_input_resource> input_resources = {});

common_agent_runtime_host_inputs make_agent_cli_runtime_host_agent_inputs(
    common_memory_store & store,
    common_plan_store & plan_store,
    args & options,
    common_agent_scope & scope,
    std::string & current_plan_id,
    const std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
    const common_agent_orchestration_config & orchestration_config,
    common_memory_scope memory_scope,
    const std::vector<common_memory_hit> & memories,
    bool memory_enabled,
    const std::string & fallback_reason,
    const common_agent_runtime_tooling & tooling,
    agent_embedding_provider * embedding_provider,
    common_agent_runtime_host_post_run post_run,
    std::vector<common_agent_input_resource> input_resources = {});

int finish_agent_cli_runtime_result(const common_agent_result & result);
