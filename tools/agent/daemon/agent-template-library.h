#pragma once
#include "agent/template-library.h"
#include "agent/agent-bootstrap.h"
#include "resource/resource-contract.h"

// Installs immutable definitions and publishes an internal resource receipt only
// after verifying the complete package. Historical receipts authorize no library
// unless its stable identity and owner remain in the current host configuration.
bool load_agent_template_libraries(
    const std::vector<common_agent_template_library_config> & libraries,
    common_memory_store & memories, common_plan_store & plans,
    agent_resource_store & resources, common_agent_bootstrap_embedder embed,
    common_agent_template_library_snapshot & snapshot,
    std::vector<common_blueprint_candidate> & candidates, std::string & error);
