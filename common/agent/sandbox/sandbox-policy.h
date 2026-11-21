#pragma once

#include "sandbox-contract.h"

#include <string>

inline bool common_agent_sandbox_network_scope_within(
        common_agent_sandbox_network_scope requested,
        common_agent_sandbox_network_scope allowed) {
    switch (allowed) {
        case common_agent_sandbox_network_scope::none:
            return requested == common_agent_sandbox_network_scope::none;
        case common_agent_sandbox_network_scope::dns_only:
            return requested == common_agent_sandbox_network_scope::none ||
                requested == common_agent_sandbox_network_scope::dns_only;
        case common_agent_sandbox_network_scope::allowlisted:
            return requested == common_agent_sandbox_network_scope::none ||
                requested == common_agent_sandbox_network_scope::dns_only ||
                requested == common_agent_sandbox_network_scope::allowlisted;
        case common_agent_sandbox_network_scope::package_registry:
            return requested == common_agent_sandbox_network_scope::none ||
                requested == common_agent_sandbox_network_scope::dns_only ||
                requested == common_agent_sandbox_network_scope::allowlisted ||
                requested == common_agent_sandbox_network_scope::package_registry;
        case common_agent_sandbox_network_scope::research_web:
            return true;
    }
    return false;
}

inline bool common_agent_sandbox_filesystem_scope_within(
        common_agent_sandbox_filesystem_scope requested,
        common_agent_sandbox_filesystem_scope allowed) {
    switch (allowed) {
        case common_agent_sandbox_filesystem_scope::readonly:
            return requested == common_agent_sandbox_filesystem_scope::readonly;
        case common_agent_sandbox_filesystem_scope::workspace_write:
            return requested == common_agent_sandbox_filesystem_scope::readonly ||
                requested == common_agent_sandbox_filesystem_scope::workspace_write;
        case common_agent_sandbox_filesystem_scope::artifact_write:
            return true;
    }
    return false;
}

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
        if (!common_agent_sandbox_network_scope_within(request.network, network)) {
            error = "sandbox network scope exceeds policy";
            return false;
        }
        if (!common_agent_sandbox_filesystem_scope_within(request.filesystem, filesystem)) {
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
