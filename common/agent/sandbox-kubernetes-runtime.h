#pragma once

#include "sandbox-contract.h"

#include <string>
#include <utility>

struct common_agent_kubernetes_sandbox_config {
    std::string executable = "kubectl";
    std::string namespace_name = "default";
    std::string service_account;
    std::string runtime_class;
    bool cleanup = true;
};

// Kubernetes Job-backed implementation of the host-owned sandbox contract.
// The first slice uses hostPath mounts, making it suitable for a local
// Kubernetes installation where the workspace is visible to the worker node.
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
