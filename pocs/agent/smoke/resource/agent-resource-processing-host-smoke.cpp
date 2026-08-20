#include "tools/agent/resource/agent-resource-processing-host.h"

#include <cassert>
#include <filesystem>

int main() {
    common_agent_sandbox_local_test_runtime runtime;
    common_agent_sandbox_policy policy;
    policy.execution_class = "resource.processor.test";
    policy.limits.timeout_ms = 1000;
    policy.limits.max_output_bytes = 4096;
    policy.network = common_agent_sandbox_network_scope::none;
    policy.filesystem = common_agent_sandbox_filesystem_scope::artifact_write;

    common_agent_workspace_manager workspace_manager({
        (std::filesystem::temp_directory_path() / "llama-agent-processing-host-workspaces").string(),
        (std::filesystem::temp_directory_path() / "llama-agent-processing-host-artifacts").string(),
    });
    agent_sandbox_resource_processing_host host(runtime, policy);
    host.set_workspace_manager(&workspace_manager);

    agent_resource_processing_execution_context context;
    context.operation_id = "processor-host-smoke";
    context.workspace.workspace_id = "resource:project";
    context.workspace.namespace_id = "local";
    context.workspace.session_id = "session-1";
    context.workspace.turn_id = "turn-1";

    common_agent_sandbox_request request;
    request.execution_class = "resource.processor.test";
    request.command.program = "test-processor";
    request.limits.timeout_ms = 500;
    request.limits.max_output_bytes = 1024;

    common_agent_sandbox_result result;
    std::vector<agent_resource_processing_host_artifact> artifacts;
    std::string error;
    assert(host.execute(context, request, result, artifacts, error));
    assert(error.empty());
    assert(result.status == common_agent_sandbox_status::completed);
    assert(result.backend_execution_id == "local-test/processor-host-smoke");

    request.artifacts.paths = {"missing.png"};
    artifacts.clear();
    error.clear();
    assert(!host.execute(context, request, result, artifacts, error));
    assert(error.find("artifact was not produced") != std::string::npos);
    return 0;
}
