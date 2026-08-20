#include "agent/sandbox/sandbox-kubernetes-runtime.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

int main() {
    const bool insecure_skip_tls_verify = std::getenv("LLAMA_AGENT_KUBERNETES_INSECURE_SKIP_TLS_VERIFY") != nullptr;
    common_agent_sandbox_kubernetes_runtime runtime({"kubectl", {}, {}, insecure_skip_tls_verify, "llama-agent", {}, {}, {}, "4Gi", "1Gi", "alpine:3.20", "project", 120000, true});
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

    const char * enabled = std::getenv("LLAMA_AGENT_KUBERNETES_SMOKE");
    if (enabled == nullptr || std::string(enabled) != "1") {
        std::printf("kubernetes_sandbox_contract=ok\n");
        std::printf("kubernetes_backend_smoke=skipped\n");
        return 0;
    }

    const auto root = std::filesystem::temp_directory_path() / "llama-agent-kubernetes-smoke";
    std::filesystem::create_directories(root / "source");
    std::filesystem::create_directories(root / "writable");
    std::filesystem::create_directories(root / "artifacts");
    const char * image = std::getenv("LLAMA_AGENT_KUBERNETES_IMAGE");
    const char * namespace_name = std::getenv("LLAMA_AGENT_KUBERNETES_NAMESPACE");
    common_agent_sandbox_kubernetes_runtime backend({
        "kubectl",
        {}, {}, insecure_skip_tls_verify,
        namespace_name == nullptr ? "default" : namespace_name,
        {}, {}, {}, "4Gi", "1Gi", "alpine:3.20", "project", 120000, true});
    request.operation_id = "kubernetes-smoke-1";
    request.image = image == nullptr ? "alpine:3.20" : image;
    request.command.arguments = {"-c", "printf kubernetes-ok > /workspace/artifacts/result.txt; printf kubernetes-ok"};
    request.network = common_agent_sandbox_network_scope::none;
    request.workspace.source_path = (root / "source").string();
    request.workspace.writable_path = (root / "writable").string();
    request.workspace.artifact_path = (root / "artifacts").string();
    request.limits.timeout_ms = 30000;
    request.limits.cpu_count = 1;
    request.limits.max_output_bytes = 4096;
    request.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;
    error.clear();
    if (!backend.execute(request, result, error) ||
            result.status != common_agent_sandbox_status::completed ||
            result.exit_code != 0 ||
            result.stdout_excerpt.find("kubernetes-ok") == std::string::npos ||
            result.artifacts.empty()) {
        std::fprintf(stderr, "Kubernetes sandbox smoke failed: %s\n", (error.empty() ? result.error : error).c_str());
        return 1;
    }

    std::printf("kubernetes_sandbox_contract=ok\n");
    std::printf("kubernetes_backend_smoke=passed\n");
    return 0;
}
