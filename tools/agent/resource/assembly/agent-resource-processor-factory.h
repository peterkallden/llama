#pragma once

#include "tools/agent/resource/agent-resource-processing-service.h"
#include "tools/agent/resource/dispatch/agent-resource-processor-dispatch.h"
#include "agent/sandbox-runtime.h"
#include "agent/sandbox-policy.h"
#include "agent/workspace-manager.h"

#include <functional>
#include <map>
#include <memory>
#include <string>

// Inputs needed to assemble one operation-scoped resource-processing provider.
// Selection stays in dispatch; this structure only carries already-normalized
// host-owned policy and backend capabilities into the assembly seam.
struct agent_resource_processing_assembly_request {
    std::map<std::string, agent_resource_processor_execution_policy> policies;
    std::map<std::string, common_agent_sandbox_policy> sandbox_classes;
    common_agent_sandbox_policy sandbox_defaults;
    std::shared_ptr<common_agent_sandbox_runtime> docker_runtime;
    std::shared_ptr<common_agent_sandbox_runtime> kubernetes_runtime;
    std::shared_ptr<common_agent_sandbox_runtime> local_runtime;
    std::shared_ptr<common_agent_workspace_manager> workspace_manager;
    agent_resource_store * resource_store = nullptr;
    std::string sandbox_backend;
};

agent_resource_processing_provider_factory make_agent_resource_processing_provider_factory(
        agent_resource_processing_assembly_request request);
