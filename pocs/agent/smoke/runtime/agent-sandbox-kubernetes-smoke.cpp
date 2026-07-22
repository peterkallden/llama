#include "agent/sandbox-kubernetes-runtime.h"

#include <cstdio>

int main() {
    common_agent_sandbox_kubernetes_runtime runtime({"kubectl", "llama-agent", {}, {}, true});
    common_agent_sandbox_request request;
    request.operation_id = "kubernetes-contract-smoke";
    request.command.program = "sh";
    request.network = common_agent_sandbox_network_scope::none;

    common_agent_sandbox_result result;
    std::string error;
    if (runtime.execute(request, result, error) ||
            error.find("requires an image") == std::string::npos) {
        std::fprintf(stderr, "Kubernetes missing-image contract was not rejected\n");
        return 1;
    }

    request.image = "llama-agent-test:latest";
    request.network = common_agent_sandbox_network_scope::package_registry;
    error.clear();
    if (runtime.execute(request, result, error) ||
            error.find("network=none") == std::string::npos) {
        std::fprintf(stderr, "Kubernetes network policy contract was not rejected\n");
        return 1;
    }

    std::printf("kubernetes_sandbox_contract=ok\n");
    return 0;
}
