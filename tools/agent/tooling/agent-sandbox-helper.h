#pragma once

#include "agent/sandbox-contract.h"
#include "agent/sandbox-policy.h"
#include "agent/tool-runtime-contract.h"

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

        common_agent_sandbox_result result;
        if (!runtime.execute(request, result, error)) {
            return common_tool_execution_result::failure(
                "sandbox.execution_failed",
                common_tool_failure_class::execution,
                false,
                "The sandbox backend failed to execute the request.",
                error);
        }
        if (result.status != common_agent_sandbox_status::completed) {
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

private:
    common_agent_sandbox_runtime & runtime;
    common_agent_sandbox_policy policy;
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
