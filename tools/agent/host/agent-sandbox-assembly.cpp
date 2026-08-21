#include "agent-sandbox-assembly.h"

#include "agent/sandbox/sandbox-docker-runtime.h"
#include "agent/sandbox/sandbox-kubernetes-runtime.h"
#include "agent/sandbox/sandbox-local-runtime.h"
#include "agent/sandbox/sandbox-policy.h"
#include "tools/agent/tooling/agent-sandbox-helper.h"

#include <utility>

namespace {

common_agent_sandbox_policy resolve_policy(
        const common_agent_sandbox_host_config & config,
        const std::string & execution_class) {
    common_agent_sandbox_policy policy = config.defaults;
    const auto it = config.classes.find(execution_class);
    if (it != config.classes.end()) policy = it->second;
    policy.execution_class = execution_class;
    return policy;
}

common_agent_workspace_context make_workspace_context(
        const common_agent_scope & scope,
        const common_agent_sandbox_request & request) {
    common_agent_workspace_context context;
    context.workspace_id = scope.project_id.empty()
        ? "session:" + scope.session_id
        : "project:" + scope.project_id;
    context.project_id = scope.project_id;
    context.namespace_id = scope.namespace_id;
    context.session_id = scope.session_id;
    context.turn_id = scope.turn_id;
    context.input_resources = request.workspace.input_resources;
    return context;
}

} // namespace

agent_host_sandbox_assembly make_agent_host_sandbox_assembly(
        agent_host_sandbox_assembly_request request) {
    agent_host_sandbox_assembly result;
    const auto config = std::move(request.config);
    const auto scope = std::move(request.scope);
    auto * resource_store = request.resource_store;

    result.workspace_manager = std::make_shared<common_agent_workspace_manager>(config.workspace);
    result.docker_runtime = std::make_shared<common_agent_sandbox_docker_runtime>(
        common_agent_docker_sandbox_config{
            config.docker_executable,
            config.docker_default_image,
        });
    result.kubernetes_runtime = std::make_shared<common_agent_sandbox_kubernetes_runtime>(
        common_agent_kubernetes_sandbox_config{
            config.kubernetes_executable,
            config.kubernetes_kubeconfig,
            config.kubernetes_context,
            config.kubernetes_insecure_skip_tls_verify,
            config.kubernetes_namespace,
            config.kubernetes_service_account,
            config.kubernetes_runtime_class,
            config.kubernetes_storage_class,
            config.kubernetes_workspace_storage_size,
            config.kubernetes_artifact_storage_size,
            config.kubernetes_staging_image,
            config.kubernetes_pvc_retention,
            config.kubernetes_staging_timeout_ms,
            config.kubernetes_cleanup,
        });

    if (config.backend == "docker" || config.backend == "kubernetes") {
        const auto backend = config.backend;
        const auto docker_runtime = result.docker_runtime;
        const auto kubernetes_runtime = result.kubernetes_runtime;
        const auto workspace_manager = result.workspace_manager;
        const auto policies = config.classes;
        const auto defaults = config.defaults;
        result.execute = [backend, docker_runtime, kubernetes_runtime, workspace_manager,
                policies, defaults, scope, resource_store](common_agent_sandbox_request sandbox_request) {
            auto policy = defaults;
            const auto it = policies.find(sandbox_request.execution_class);
            if (it != policies.end()) policy = it->second;
            policy.execution_class = sandbox_request.execution_class;
            common_agent_sandbox_runtime & runtime = backend == "docker"
                ? static_cast<common_agent_sandbox_runtime &>(*docker_runtime)
                : static_cast<common_agent_sandbox_runtime &>(*kubernetes_runtime);
            common_agent_sandbox_tool_helper helper(runtime, policy);
            helper.set_workspace_manager(workspace_manager.get());
            helper.set_resource_store(resource_store, {
                scope.namespace_id,
                scope.session_id,
                scope.project_id,
                scope.turn_id,
            });
            return helper.run_for_workspace(
                make_workspace_context(scope, sandbox_request),
                sandbox_request.operation_id,
                std::move(sandbox_request));
        };
        return result;
    }

    auto unavailable_runtime = std::make_shared<common_agent_sandbox_unavailable_runtime>();
    const auto policies = config.classes;
    const auto defaults = config.defaults;
    result.execute = [unavailable_runtime, policies, defaults](common_agent_sandbox_request sandbox_request) {
        auto policy = defaults;
        const auto it = policies.find(sandbox_request.execution_class);
        if (it != policies.end()) policy = it->second;
        policy.execution_class = sandbox_request.execution_class;
        common_agent_sandbox_tool_helper helper(*unavailable_runtime, policy);
        return helper.run(sandbox_request);
    };
    return result;
}
