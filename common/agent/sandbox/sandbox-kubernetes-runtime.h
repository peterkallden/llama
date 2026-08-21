#pragma once

#include "sandbox-contract.h"

#include <string>
#include <utility>

struct common_agent_kubernetes_sandbox_config {
    std::string executable = "kubectl";
    std::string kubeconfig;
    std::string context;
    bool insecure_skip_tls_verify = false;
    std::string namespace_name = "default";
    std::string service_account;
    std::string runtime_class;
    std::string storage_class;
    std::string workspace_storage_size = "4Gi";
    std::string artifact_storage_size = "1Gi";
    std::string staging_image = "alpine:3.20";
    std::string pvc_retention = "project";
    uint32_t staging_timeout_ms = 120000;
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
