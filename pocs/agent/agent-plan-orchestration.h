#pragma once

#include "../common/cli-config.h"

#include "agent/agent-contract.h"
#include "agent/agent-inference.h"
#include "agent/blueprint-selector.h"
#include "agent/tool-registry.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <string>
#include <vector>

struct common_agent_generation_config;

bool maybe_install_agent_bootstrap(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const args & options,
    const common_agent_scope & scope,
    std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
    std::string & error);

bool maybe_export_agent_package(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const args & options,
    bool & exported,
    std::string & error);

bool maybe_auto_select_plan(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const args & options,
    args & mutable_options,
    const common_agent_scope & scope,
    common_plan_store & plan_store,
    std::string & error);

bool maybe_auto_select_blueprint(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const args & options,
    args & mutable_options,
    const common_agent_scope & scope,
    common_plan_store & plan_store,
    const std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
    bool profile_tools_active,
    const common_tool_registry * tool_registry,
    std::string & error);
