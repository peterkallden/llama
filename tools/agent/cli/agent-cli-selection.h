#pragma once

#include "agent/agent-inference.h"
#include "agent/learning/blueprint-selector.h"
#include "tools/agent/cli/agent-cli-options.h"
#include "tools/agent/runtime/agent-runtime-package-io.h"
#include "tools/agent/runtime/agent-route-compiler.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

struct common_agent_generation_config;
class agent_tool_view;

struct common_agent_workflow_selection_result {
    std::optional<common_plan_workflow_binding> binding;
    std::string reason;
    std::optional<common_agent_generated_text_result> generation;
};

struct common_agent_workflow_action_selection_result {
    std::optional<common_agent_tool_argument_binding> action;
    std::string reason;
    std::optional<common_agent_generated_text_result> generation;
};

// Select one exact tool and schema-valid argument object inside a compiled
// workflow route. The host later owns plan IDs, dependencies and materializing
// any deterministic A* prefix.
common_agent_workflow_action_selection_result select_llama_cli_workflow_action(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const common_agent_request & request,
    const std::vector<common_chat_tool> & tools,
    const std::vector<common_agent_tool_argument_binding> & fixed_bindings,
    const std::vector<std::string> & host_resolved_dataset_tools,
    common_agent_tool_output_format output_format,
    std::string & error);

// Choose one exact entity id from a host-recorded observation.  The caller
// validates the result against the same observation before binding it.
bool select_llama_cli_workflow_candidate(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const common_agent_request & request,
    const common_plan_observation & observation,
    const common_plan_workflow_transition & transition,
    std::string & candidate_id,
    std::string & error);

struct common_agent_route_selection_result {
    std::optional<std::string> route_id;
    float confidence = 0.0f;
    std::string reason;
    std::optional<common_agent_generated_text_result> generation;
};

common_agent_route_selection_result select_llama_cli_route(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const common_agent_request & request,
    const common_agent_route_catalog & catalog,
    std::string & error);

common_agent_workflow_selection_result select_llama_cli_blueprint_workflow(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const common_agent_request & request,
    const std::vector<common_plan_workflow_binding> & bindings,
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
