#pragma once

#include "../common/cli-config.h"

#include "agent-host-mcp-provider-config.h"
#include "agent-resource-store.h"
#include "agent-tool-provider.h"
#include "agent-runtime-host.h"
#include "agent-runtime-tooling.h"

#include <memory>
#include <string>
#include <vector>

struct common_agent_cli_tool_selection {
    common_agent_runtime_tooling tooling;
    std::unique_ptr<agent_tool_view> tool_view;
    std::vector<std::unique_ptr<agent_mcp_tool_client>> mcp_clients;
    std::unique_ptr<agent_embedding_provider> embedding_provider;
    std::unique_ptr<agent_resource_store> owned_resource_store;
};

struct agent_host_stdio_mcp_provider_request {
    std::string server_name = "mcp";
    std::vector<std::string> command_line;
    std::string exposed_name_prefix;
};

struct agent_host_tool_selection_request {
    agent_tool_context tool_context;
    std::string repository_root;
    agent_resource_store_config resource_store_config;
    std::vector<agent_host_stdio_mcp_provider_request> mcp_providers;
    std::string mcp_tool_command;
    std::vector<std::string> mcp_tool_args;
    std::string mcp_tool_server_name = "mcp";
    std::string mcp_tool_prefix;
};

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
    common_agent_generation_options generation_options = {});

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
    common_agent_runtime_host_post_run post_run);

common_agent_runtime_host_inputs make_agent_cli_runtime_host_mini_inputs(
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
    common_agent_runtime_host_post_run post_run);

int finish_agent_cli_runtime_result(const common_agent_result & result);
