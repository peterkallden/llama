#pragma once

#include "agent/sandbox/sandbox-contract.h"
#include "agent/sandbox/sandbox-policy.h"
#include "agent/sandbox/sandbox-runtime.h"
#include "agent/workspace-manager.h"
#include "agent/tooling/contracts/tool-runtime-contract.h"

#include <nlohmann/json.hpp>

#include <utility>

class common_agent_sandbox_tool_helper {
public:
    common_agent_sandbox_tool_helper(
            common_agent_sandbox_runtime & runtime,
            common_agent_sandbox_policy policy)
        : runtime(runtime), policy(std::move(policy)) {}

    common_tool_execution_result run(
            const common_agent_sandbox_request & request) {
        std::string error;
        if (!policy.validate(request, error)) {
            return common_tool_execution_result::failure(
                "sandbox.policy_rejected",
                common_tool_failure_class::policy,
                false,
                "The sandbox request was rejected by host policy.",
                error);
        }

        auto resolved_request = request;
        if (resolved_request.image.empty()) resolved_request.image = policy.image;
        common_agent_sandbox_result result;
        if (!runtime.execute(resolved_request, result, error)) {
            return common_tool_execution_result::failure(
                "sandbox.execution_failed",
                common_tool_failure_class::execution,
                false,
                "The sandbox backend failed to execute the request.",
                error);
        }
        if (result.status != common_agent_sandbox_status::completed) {
            if (result.status == common_agent_sandbox_status::backend_unavailable) {
                return common_tool_execution_result::failure(
                    "sandbox.backend_unavailable",
                    common_tool_failure_class::execution,
                    false,
                    "No sandbox execution backend is configured for this host.",
                    result.error);
            }
            const auto failure_class = result.status == common_agent_sandbox_status::timed_out
                ? common_tool_failure_class::timeout
                : common_tool_failure_class::execution;
            return common_tool_execution_result::failure(
                result.status == common_agent_sandbox_status::timed_out
                    ? "sandbox.timed_out"
                    : "sandbox.execution_failed",
                failure_class,
                false,
                "The sandbox did not complete successfully.",
                result.error);
        }

        nlohmann::ordered_json output = {
            {"status", common_agent_sandbox_status_name(result.status)},
            {"exit_code", result.exit_code},
            {"stdout", result.stdout_excerpt},
            {"stderr", result.stderr_excerpt},
            {"backend_execution_id", result.backend_execution_id},
        };
        return common_tool_execution_result::success(
            output.dump(),
            "Sandbox execution completed.",
            result.artifacts);
    }

    common_tool_execution_result run_for_workspace(
            const common_agent_workspace_context & context,
            const std::string & operation_id,
            common_agent_sandbox_request request,
            common_agent_workspace_operation * operation_out = nullptr) {
        if (workspace_manager == nullptr) {
            return common_tool_execution_result::failure(
                "sandbox.workspace_manager_unavailable",
                common_tool_failure_class::execution,
                false,
                "The host has not configured a workspace manager.",
                "sandbox workspace manager is required for workspace execution");
        }
        common_agent_workspace_operation operation;
        std::string error;
        if (!prepare_workspace_request(
                context, operation_id, request, operation, operation_out, error)) {
            const bool materialization_failed =
                error.rfind("sandbox resource materialization failed:", 0) == 0;
            return common_tool_execution_result::failure(
                materialization_failed
                    ? "sandbox.resource_materialization_failed"
                    : "sandbox.workspace_setup_failed",
                common_tool_failure_class::execution,
                false,
                materialization_failed
                    ? "A sandbox input resource could not be materialized."
                    : "The sandbox workspace could not be prepared.",
                error);
        }
        return run(request);
    }

    bool run_raw_for_workspace(
            const common_agent_workspace_context & context,
            const std::string & operation_id,
            common_agent_sandbox_request request,
            common_agent_sandbox_result & result,
            std::string & error,
            common_agent_workspace_operation * operation_out = nullptr) {
        common_agent_workspace_operation operation;
        if (!prepare_workspace_request(
                context, operation_id, request, operation, operation_out, error)) {
            return false;
        }
        return run_raw(request, result, error);
    }

    bool run_raw(
            const common_agent_sandbox_request & request,
            common_agent_sandbox_result & result,
            std::string & error) {
        if (!policy.validate(request, error)) return false;
        auto resolved_request = request;
        if (resolved_request.image.empty()) resolved_request.image = policy.image;
        if (!runtime.execute(resolved_request, result, error)) return false;
        return true;
    }

    void set_workspace_manager(common_agent_workspace_manager * manager) {
        workspace_manager = manager;
    }

    void set_resource_store(
            agent_resource_store * store,
            agent_resource_read_authority authority = {}) {
        resource_store = store;
        resource_authority = std::move(authority);
    }

private:
    bool prepare_workspace_request(
            const common_agent_workspace_context & context,
            const std::string & operation_id,
            common_agent_sandbox_request & request,
            common_agent_workspace_operation & operation,
            common_agent_workspace_operation * operation_out,
            std::string & error) {
        if (workspace_manager == nullptr) {
            error = "sandbox workspace manager is required for workspace execution";
            return false;
        }
        if (!workspace_manager->create_operation(context, operation_id, operation, error)) {
            return false;
        }
        request.operation_id = operation.operation_id;
        request.project_id = context.project_id;
        request.workspace_id = context.workspace_id;
        request.workspace.source_path = operation.source_path;
        request.workspace.writable_path = operation.writable_path;
        request.workspace.artifact_path = operation.artifact_path;
        if (resource_store != nullptr) {
            for (size_t index = 0; index < request.workspace.input_resources.size(); ++index) {
                const auto & resource = request.workspace.input_resources[index];
                const auto file_name = resource.name.empty()
                    ? "input-" + std::to_string(index + 1) + ".txt"
                    : resource.name;
                std::string materialized_path;
                if (!workspace_manager->materialize_resource(
                        operation,
                        resource,
                        *resource_store,
                        resource_authority,
                        file_name,
                        request.artifacts.max_bytes,
                        materialized_path,
                        error)) {
                    error = "sandbox resource materialization failed: " + error;
                    return false;
                }
            }
        }
        if (operation_out != nullptr) *operation_out = operation;
        return true;
    }

    common_agent_sandbox_runtime & runtime;
    common_agent_sandbox_policy policy;
    common_agent_workspace_manager * workspace_manager = nullptr;
    agent_resource_store * resource_store = nullptr;
    agent_resource_read_authority resource_authority;
};

class common_agent_sandbox_local_test_runtime final : public common_agent_sandbox_runtime {
public:
    bool execute(
            const common_agent_sandbox_request & request,
            common_agent_sandbox_result & result,
            std::string & error) override {
        if (request.command.program.empty()) {
            error = "local test sandbox requires a declared program";
            return false;
        }
        last_request = request;
        result = {};
        result.status = common_agent_sandbox_status::completed;
        result.exit_code = 0;
        result.stdout_excerpt = "local sandbox test backend";
        result.backend_execution_id = "local-test/" + request.operation_id;
        error.clear();
        return true;
    }

    common_agent_sandbox_request last_request;
};
