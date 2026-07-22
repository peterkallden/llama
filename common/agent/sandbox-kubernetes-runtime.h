#pragma once

#include "sandbox-contract.h"

#include <string>
#include <utility>

struct common_agent_kubernetes_sandbox_config {
    std::string executable = "kubectl";
    std::string namespace_name = "default";
    std::string service_account;
    std::string runtime_class;
    std::string storage_class;
    std::string workspace_storage_size = "4Gi";
    std::string artifact_storage_size = "1Gi";
    bool cleanup = true;
};

// Kubernetes Job-backed implementation of the host-owned sandbox contract.
// Workspaces are materialized into project- or session-scoped PVCs; Jobs use
// operation-specific subPaths so the host does not need to be node-visible.
class common_agent_sandbox_kubernetes_runtime final : public common_agent_sandbox_runtime {
public:
    explicit common_agent_sandbox_kubernetes_runtime(common_agent_kubernetes_sandbox_config config = {})
        : config(std::move(config)) {}

    bool execute(
            const common_agent_sandbox_request & request,
            common_agent_sandbox_result & result,
            std::string & error) override;

private:
    common_agent_kubernetes_sandbox_config config;
};
