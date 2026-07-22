#pragma once

#include "sandbox-contract.h"

#include <string>

struct common_agent_sandbox_policy {
    std::string execution_class;
    // Host-owned image selection. Requests may leave image empty and inherit
    // the policy image, then the backend default image.
    std::string image;
    common_agent_sandbox_limits limits;
    common_agent_sandbox_network_scope network = common_agent_sandbox_network_scope::none;
    common_agent_sandbox_filesystem_scope filesystem = common_agent_sandbox_filesystem_scope::readonly;
    bool allow_artifacts = true;

    bool validate(
            const common_agent_sandbox_request & request,
            std::string & error) const {
        if (execution_class.empty() || request.execution_class != execution_class) {
            error = "sandbox execution class is not allowed";
            return false;
        }
        if (request.operation_id.empty() || request.command.program.empty()) {
            error = "sandbox request requires an operation id and program";
            return false;
        }
        if (request.limits.timeout_ms == 0 || request.limits.timeout_ms > limits.timeout_ms) {
            error = "sandbox timeout exceeds policy";
            return false;
        }
        if (limits.memory_bytes != 0 &&
                (request.limits.memory_bytes == 0 || request.limits.memory_bytes > limits.memory_bytes)) {
            error = "sandbox memory limit exceeds policy";
            return false;
        }
        if (request.limits.cpu_count == 0 || request.limits.cpu_count > limits.cpu_count) {
            error = "sandbox CPU limit exceeds policy";
            return false;
        }
        if (limits.max_output_bytes != 0 &&
                (request.limits.max_output_bytes == 0 || request.limits.max_output_bytes > limits.max_output_bytes)) {
            error = "sandbox output limit exceeds policy";
            return false;
        }
        if (static_cast<int>(request.network) > static_cast<int>(network)) {
            error = "sandbox network scope exceeds policy";
            return false;
        }
        if (static_cast<int>(request.filesystem) > static_cast<int>(filesystem)) {
            error = "sandbox filesystem scope exceeds policy";
            return false;
        }
        if (!allow_artifacts && request.artifacts.collect) {
            error = "sandbox artifacts are not allowed by policy";
            return false;
        }
        error.clear();
        return true;
    }
};
