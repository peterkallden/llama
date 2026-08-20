#pragma once

#include "agent/agent-inference.h"
#include "agent/learning/blueprint-selector.h"
#include "tools/agent/cli/agent-cli-options.h"
#include "tools/agent/runtime/agent-runtime-package-io.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

struct common_agent_generation_config;
class agent_tool_view;

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
