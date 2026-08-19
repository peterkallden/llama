#pragma once

#include "agent/agent-inference.h"
#include "agent/agent-bootstrap.h"
#include "agent/learning/blueprint-selector.h"
#include "tools/agent/cli/agent-cli-options.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

struct common_agent_generation_config;
class agent_tool_view;

bool parse_plan_scope(const std::string & value, common_plan_scope & scope);

bool load_bootstrap_file(
    const std::string & path,
    common_agent_bootstrap_package & package,
    std::string & error);

bool export_agent_package(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const common_agent_scope & scope,
    const std::string & output_path,
    std::string & error);

std::unique_ptr<common_blueprint_selector> make_llama_cli_blueprint_selector(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config);

struct common_agent_plan_selection_result {
    std::optional<std::string> plan_id;
    float confidence = 0.0f;
    std::string reason;
    std::optional<common_agent_generated_text_result> generation;
};

common_agent_plan_selection_result select_llama_cli_plan_result(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const common_agent_request & request,
    const std::vector<common_plan_state> & candidates,
    std::string & error);

std::optional<std::string> select_llama_cli_plan(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const common_agent_request & request,
    const std::vector<common_plan_state> & candidates,
    std::string & error);

struct common_agent_blueprint_binding_result {
    bool applied = false;
    size_t bound_steps = 0;
    std::string reason;
    std::optional<common_agent_generated_text_result> generation;
};

common_agent_blueprint_binding_result bind_llama_cli_blueprint_tools_result(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    agent_tool_view & tool_view,
    const common_agent_request & request,
    common_plan_store & store,
    const std::string & plan_id,
    std::string & error);

bool bind_llama_cli_blueprint_tools(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    agent_tool_view & tool_view,
    const common_agent_request & request,
    common_plan_store & store,
    const std::string & plan_id,
    std::string & error);
