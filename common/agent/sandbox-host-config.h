#pragma once

#include "sandbox-policy.h"
#include "workspace-contract.h"

#include <map>
#include <string>

// Backend selection and workspace policy owned by the host. This neutral
// shape is carried through CLI, daemon and MCP host options.
struct common_agent_sandbox_host_config {
    std::string backend = "none";
    std::string docker_executable = "docker";
    std::string docker_default_image;
    common_agent_workspace_roots workspace;
    common_agent_sandbox_policy defaults;
    std::map<std::string, common_agent_sandbox_policy> classes;
};
