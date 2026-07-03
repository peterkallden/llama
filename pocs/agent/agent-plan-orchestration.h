#pragma once

#include "../common/cli-config.h"

#include "agent/agent-contract.h"
#include "agent/agent-inference.h"
#include "agent/blueprint-selector.h"
#include "agent/tool-registry.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <functional>
#include <string>
#include <vector>

struct common_agent_generation_config;
class agent_tool_view;

struct common_agent_orchestration_config {
    std::string prompt;
    std::string agent_plan = "off";
    std::string agent_blueprint;
    std::string agent_bootstrap = "none";
    std::string agent_import;
    std::string agent_export;
};

common_agent_orchestration_config make_agent_orchestration_config(const args & options);

struct common_agent_bootstrap_runtime_config {
    std::function<bool(const std::string & text, std::vector<float> & embedding, std::string & error)> embed_procedure;
};

common_agent_bootstrap_runtime_config make_agent_bootstrap_runtime_config(const args & options);

bool maybe_install_agent_bootstrap(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const common_agent_orchestration_config & config,
    const common_agent_bootstrap_runtime_config & runtime_config,
    const common_agent_scope & scope,
    std::string & current_plan_id,
    std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
    std::string & error);

bool maybe_export_agent_package(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    const common_agent_orchestration_config & config,
    const common_agent_scope & scope,
    bool & exported,
    std::string & error);

bool maybe_auto_select_plan(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const common_agent_orchestration_config & config,
    std::string & current_plan_id,
    const common_agent_scope & scope,
    common_plan_store & plan_store,
    std::string & error);

bool maybe_auto_select_blueprint(
    common_agent_inference & inference,
    const common_agent_generation_config & generation_config,
    const common_agent_orchestration_config & config,
    std::string & current_plan_id,
    const common_agent_scope & scope,
    common_plan_store & plan_store,
    const std::vector<common_blueprint_candidate> & installed_blueprint_candidates,
    bool profile_tools_active,
    agent_tool_view * tool_view,
    std::string & error);
