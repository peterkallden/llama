#pragma once

#include "agent-host-config.h"
#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/sandbox/sandbox-runtime.h"
#include "agent/workspace-manager.h"

#include <functional>
#include <memory>

class common_agent_sandbox_docker_runtime;
class common_agent_sandbox_kubernetes_runtime;
class common_agent_sandbox_lxc_runtime;

struct agent_host_sandbox_assembly_request {
    common_agent_sandbox_host_config config;
    common_agent_scope scope;
    agent_resource_store * resource_store = nullptr;
};

struct agent_host_sandbox_assembly {
    std::shared_ptr<common_agent_sandbox_docker_runtime> docker_runtime;
    std::shared_ptr<common_agent_sandbox_kubernetes_runtime> kubernetes_runtime;
    std::shared_ptr<common_agent_sandbox_lxc_runtime> lxc_runtime;
    std::shared_ptr<common_agent_workspace_manager> workspace_manager;
    std::function<common_tool_execution_result(common_agent_sandbox_request)> execute;
};

agent_host_sandbox_assembly make_agent_host_sandbox_assembly(
        agent_host_sandbox_assembly_request request);
