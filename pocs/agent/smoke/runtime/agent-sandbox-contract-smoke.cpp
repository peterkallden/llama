#include "tools/agent/tooling/agent-sandbox-helper.h"
#include "agent/sandbox/sandbox-kubernetes-runtime.h"
#include "agent/sandbox/sandbox-lxc-runtime.h"

#include <cstdio>
#include <filesystem>

int main() {
    common_agent_sandbox_capabilities docker_capabilities;
    docker_capabilities.process_isolation = true;
    docker_capabilities.filesystem_readonly = true;
    docker_capabilities.filesystem_workspace_write = true;
    docker_capabilities.filesystem_artifact_write = true;
    docker_capabilities.network_none = true;
    docker_capabilities.cpu_limit = true;
    docker_capabilities.memory_limit = true;
    docker_capabilities.process_limit = true;
    docker_capabilities.artifact_collection = true;
    if (!docker_capabilities.process_isolation ||
            !docker_capabilities.filesystem_artifact_write ||
            !docker_capabilities.network_none ||
            docker_capabilities.network_package_registry) {
        std::fprintf(stderr, "Docker capability declaration is inconsistent\n");
        return 1;
    }

    common_agent_sandbox_request capability_request;
    capability_request.operation_id = "capability-check";
    capability_request.command.program = "test-program";
    capability_request.limits.cpu_count = 1;
    capability_request.network = common_agent_sandbox_network_scope::none;
    capability_request.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;
    std::string capability_error;
    if (!common_agent_sandbox_validate_capabilities(
            capability_request, docker_capabilities, capability_error)) {
        std::fprintf(stderr, "Docker capability check unexpectedly failed: %s\n",
            capability_error.c_str());
        return 1;
    }
    capability_request.network = common_agent_sandbox_network_scope::package_registry;
    if (common_agent_sandbox_validate_capabilities(
            capability_request, docker_capabilities, capability_error)) {
        std::fprintf(stderr, "Unsupported Docker network scope was accepted\n");
        return 1;
    }

    common_agent_sandbox_kubernetes_runtime kubernetes_runtime;
    const auto kubernetes_capabilities = kubernetes_runtime.capabilities();
    if (!kubernetes_capabilities.process_isolation ||
            !kubernetes_capabilities.network_none ||
            !kubernetes_capabilities.cpu_limit ||
            !kubernetes_capabilities.memory_limit ||
            kubernetes_capabilities.process_limit ||
            kubernetes_capabilities.network_package_registry) {
        std::fprintf(stderr, "Kubernetes capability declaration is inconsistent\n");
        return 1;
    }

    common_agent_sandbox_lxc_runtime lxc_without_profile({
        "lxc", "ubuntu:24.04", "none", "", "none", true,
    });
    const auto lxc_without_profile_capabilities = lxc_without_profile.capabilities();
    if (!lxc_without_profile_capabilities.process_isolation ||
            !lxc_without_profile_capabilities.cpu_limit ||
            !lxc_without_profile_capabilities.memory_limit ||
            !lxc_without_profile_capabilities.process_limit ||
            lxc_without_profile_capabilities.network_none ||
            lxc_without_profile_capabilities.network_allowlisted) {
        std::fprintf(stderr, "LXC capability declaration overstates an unconfigured profile\n");
        return 1;
    }

    common_agent_sandbox_lxc_runtime lxc_allowlisted({
        "lxc", "ubuntu:24.04", "profile", "agent-network-allowlisted", "allowlisted", true,
    });
    const auto lxc_allowlisted_capabilities = lxc_allowlisted.capabilities();
    if (!lxc_allowlisted_capabilities.network_allowlisted ||
            lxc_allowlisted_capabilities.network_none ||
            lxc_allowlisted_capabilities.network_research_web) {
        std::fprintf(stderr, "LXC network profile capability declaration is inconsistent\n");
        return 1;
    }
    capability_request.network = common_agent_sandbox_network_scope::allowlisted;
    capability_request.limits.memory_bytes = 1024 * 1024;
    capability_request.limits.process_count = 8;
    if (!common_agent_sandbox_validate_capabilities(
            capability_request, lxc_allowlisted_capabilities, capability_error)) {
        std::fprintf(stderr, "LXC resource/network capability check unexpectedly failed: %s\n",
            capability_error.c_str());
        return 1;
    }

    common_agent_sandbox_local_test_runtime runtime;
    common_agent_sandbox_policy policy;
    policy.execution_class = "readonly-analysis";
    policy.image = "test-sandbox-image:latest";
    policy.limits.timeout_ms = 60000;
    policy.limits.cpu_count = 1;
    policy.limits.max_output_bytes = 65536;
    policy.network = common_agent_sandbox_network_scope::none;
    policy.filesystem = common_agent_sandbox_filesystem_scope::readonly;

    common_agent_sandbox_tool_helper helper(runtime, policy);
    common_agent_sandbox_request request;
    request.operation_id = "sandbox-smoke-1";
    request.execution_class = "readonly-analysis";
    request.command.program = "test-program";
    request.limits.timeout_ms = 1000;
    request.limits.cpu_count = 1;
    request.limits.max_output_bytes = 4096;

    const auto result = helper.run(request);
    if (!result.ok || result.output.find("local-test/sandbox-smoke-1") == std::string::npos ||
            runtime.last_request.command.program != "test-program") {
        std::fprintf(stderr, "sandbox local helper did not return the expected result\n");
        return 1;
    }
    if (runtime.last_request.image != "test-sandbox-image:latest") {
        std::fprintf(stderr, "sandbox policy image was not applied by the helper\n");
        return 1;
    }

    request.network = common_agent_sandbox_network_scope::research_web;
    const auto rejected = helper.run(request);
    if (rejected.ok || rejected.failure_class != common_tool_failure_class::policy ||
            rejected.failure_code != "sandbox.policy_rejected") {
        std::fprintf(stderr, "sandbox policy did not reject an excessive network scope\n");
        return 1;
    }

    common_agent_sandbox_unavailable_runtime unavailable_runtime;
    common_agent_sandbox_tool_helper unavailable_helper(unavailable_runtime, policy);
    request.network = common_agent_sandbox_network_scope::none;
    const auto unavailable = unavailable_helper.run(request);
    if (unavailable.ok || unavailable.failure_code != "sandbox.backend_unavailable" ||
            unavailable.failure_class != common_tool_failure_class::execution) {
        std::fprintf(stderr, "no-backend runtime did not return the expected result\n");
        return 1;
    }

    common_agent_workspace_manager workspace_manager({
        (std::filesystem::temp_directory_path() / "llama-agent-sandbox-contract-workspaces").string(),
        (std::filesystem::temp_directory_path() / "llama-agent-sandbox-contract-artifacts").string(),
    });
    common_agent_sandbox_local_test_runtime workspace_runtime;
    common_agent_sandbox_tool_helper workspace_helper(workspace_runtime, policy);
    workspace_helper.set_workspace_manager(&workspace_manager);
    common_agent_workspace_context context;
    context.workspace_id = "developer:project";
    context.namespace_id = "local";
    context.session_id = "session-1";
    context.turn_id = "turn-1";
    common_agent_workspace_operation operation;
    const auto workspace_result = workspace_helper.run_for_workspace(
        context, "build/op-1", request, &operation);
    if (!workspace_result.ok || operation.source_path.empty() ||
            workspace_runtime.last_request.workspace.source_path != operation.source_path ||
            workspace_runtime.last_request.workspace.writable_path != operation.writable_path ||
            workspace_runtime.last_request.workspace.artifact_path != operation.artifact_path) {
        std::fprintf(stderr, "workspace-aware sandbox helper did not prepare the operation\n");
        return 1;
    }

    common_agent_sandbox_result raw_result;
    std::string raw_error;
    if (!workspace_helper.run_raw_for_workspace(
            context, "raw/op-1", request, raw_result, raw_error) ||
            !raw_error.empty() ||
            raw_result.status != common_agent_sandbox_status::completed ||
            raw_result.backend_execution_id != "local-test/raw/op-1") {
        std::fprintf(stderr, "workspace-aware sandbox helper did not expose the raw result\n");
        return 1;
    }

    std::printf("sandbox_status=ok\n");
    std::printf("sandbox_backend=%s\n", runtime.last_request.command.program.c_str());
    return 0;
}
