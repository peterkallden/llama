#pragma once

#include "agent/agent-bootstrap.h"
#include "tools/agent/cli/agent-cli-options.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"

#include <string>

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
