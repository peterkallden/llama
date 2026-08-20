#pragma once

#include "tools/agent/cli/agent-cli-options.h"

#include "../runtime/agent-plan-orchestration.h"
#include "tools/agent/cli/agent-cli-scope.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"
#include "agent/agent-scope.h"
#include "resource/resource-contract.h"

#include <memory>
#include <string>
#include <vector>

struct common_agent_cli_run_setup {
    bool bootstrap_enabled = false;
    std::unique_ptr<common_plan_store> plan_store;
    std::vector<common_blueprint_candidate> installed_blueprint_candidates;
    common_plan_scope requested_plan_scope = common_plan_scope::turn;
    common_agent_scope agent_scope;
    std::string active_plan_id;
    common_agent_orchestration_config orchestration_config;
    common_agent_bootstrap_runtime_config bootstrap_runtime_config;
};

bool prepare_agent_cli_args(args & options, std::string & error);

class agent_resource_store;

bool import_agent_cli_resources(
    const args & options,
    const common_agent_scope & scope,
    agent_resource_store & resource_store,
    std::vector<common_agent_input_resource> & out,
    std::string & error);

bool import_agent_cli_text_resources(
    const args & options,
    const common_agent_scope & scope,
    agent_resource_store & resource_store,
    std::vector<common_agent_input_resource> & out,
    std::string & error);

bool prepare_agent_cli_run_setup(
    common_memory_store & memory_store,
    args & options,
    common_agent_cli_run_setup & setup,
    bool & exported,
    std::string & error);
