#pragma once

#include "agent/agent-inference.h"
#include "agent/agent-bootstrap.h"
#include "agent/blueprint-selector.h"
#include "agent/tool-registry.h"
#include "common/cli-config.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

bool parse_plan_scope(const std::string & value, common_plan_scope & scope);

bool load_bootstrap_file(
    const std::string & path,
    common_agent_bootstrap_package & package,
    std::string & error);

bool export_agent_package(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const args & a,
    std::string & error);

std::unique_ptr<common_blueprint_selector> make_llama_cli_blueprint_selector(
    common_agent_inference & inference,
    const args & options);

std::optional<std::string> select_llama_cli_plan(
    common_agent_inference & inference,
    const args & options,
    const common_agent_request & request,
    const std::vector<common_plan_state> & candidates,
    std::string & error);

bool bind_llama_cli_blueprint_tools(
    common_agent_inference & inference,
    const args & options,
    const common_tool_registry & registry,
    const common_agent_request & request,
    common_plan_store & store,
    const std::string & plan_id,
    std::string & error);
