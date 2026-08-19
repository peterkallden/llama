#pragma once

#include "../runtime/agent-runtime-tooling.h"

#include "agent/agent-contract.h"
#include "agent/agent-inference.h"
#include "agent/learning/blueprint-selector.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <functional>
#include <string>
#include <vector>

struct common_agent_generation_config;

struct common_agent_orchestration_config {
    std::string prompt;
    std::string agent_plan = "off";
    std::string agent_blueprint;
    std::string agent_bootstrap = "none";
    std::string agent_import;
    std::string agent_export;
};

struct common_agent_orchestration_build_config {
    std::string prompt;
    std::string agent_plan = "off";
    std::string agent_blueprint;
    std::string agent_bootstrap = "none";
    std::string agent_import;
    std::string agent_export;
};

common_agent_orchestration_config make_agent_orchestration_config(
    common_agent_orchestration_build_config config);

struct common_agent_bootstrap_runtime_config {
    std::function<bool(const std::string & text, std::vector<float> & embedding, std::string & error)> embed_procedure;
};

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

struct common_agent_orchestration_runtime_context {
    common_agent_inference & inference;
    const common_agent_generation_config & generation_config;
    const common_agent_orchestration_config & config;
    std::string & current_plan_id;
    const common_agent_scope & scope;
    common_plan_store & plan_store;
    const std::vector<common_blueprint_candidate> & installed_blueprint_candidates;
    const std::optional<common_memory_policy_pack> * policy_pack = nullptr;
    const common_agent_runtime_tooling * tooling = nullptr;
    std::vector<common_agent_event> & pre_turn_events;
    std::vector<common_runtime_trace_entry> & pre_turn_trace;
};

bool maybe_auto_select_plan(
    const common_agent_orchestration_runtime_context & context,
    std::string & error);

bool maybe_auto_select_blueprint(
    const common_agent_orchestration_runtime_context & context,
    std::string & error);
