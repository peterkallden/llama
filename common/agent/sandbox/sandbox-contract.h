#pragma once

#include "resource/resource-contract.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class common_agent_sandbox_status {
    backend_unavailable,
    completed,
    failed,
    cancelled,
    timed_out,
};

inline const char * common_agent_sandbox_status_name(common_agent_sandbox_status status) {
    switch (status) {
        case common_agent_sandbox_status::backend_unavailable: return "backend_unavailable";
        case common_agent_sandbox_status::completed: return "completed";
        case common_agent_sandbox_status::failed: return "failed";
        case common_agent_sandbox_status::cancelled: return "cancelled";
        case common_agent_sandbox_status::timed_out: return "timed_out";
    }
    return "failed";
}

enum class common_agent_sandbox_network_scope {
    none,
    dns_only,
    allowlisted,
    package_registry,
    research_web,
};

enum class common_agent_sandbox_filesystem_scope {
    readonly,
    workspace_write,
    artifact_write,
};

// Backend capabilities are explicit rather than inferred from enum ordering.
// A backend advertises what it can enforce; policy resolution must fail closed
// when a requested scope is not represented here.
struct common_agent_sandbox_capabilities {
    bool process_isolation = false;
    bool filesystem_readonly = false;
    bool filesystem_workspace_write = false;
    bool filesystem_artifact_write = false;
    bool network_none = false;
    bool network_dns_only = false;
    bool network_allowlisted = false;
    bool network_package_registry = false;
    bool network_research_web = false;
    bool cpu_limit = false;
    bool memory_limit = false;
    bool process_limit = false;
    bool artifact_collection = false;
};

inline bool common_agent_sandbox_supports_network(
        const common_agent_sandbox_capabilities & capabilities,
        common_agent_sandbox_network_scope scope) {
    switch (scope) {
        case common_agent_sandbox_network_scope::none: return capabilities.network_none;
        case common_agent_sandbox_network_scope::dns_only: return capabilities.network_dns_only;
        case common_agent_sandbox_network_scope::allowlisted: return capabilities.network_allowlisted;
        case common_agent_sandbox_network_scope::package_registry: return capabilities.network_package_registry;
        case common_agent_sandbox_network_scope::research_web: return capabilities.network_research_web;
    }
    return false;
}

inline bool common_agent_sandbox_supports_filesystem(
        const common_agent_sandbox_capabilities & capabilities,
        common_agent_sandbox_filesystem_scope scope) {
    switch (scope) {
        case common_agent_sandbox_filesystem_scope::readonly: return capabilities.filesystem_readonly;
        case common_agent_sandbox_filesystem_scope::workspace_write: return capabilities.filesystem_workspace_write;
        case common_agent_sandbox_filesystem_scope::artifact_write: return capabilities.filesystem_artifact_write;
    }
    return false;
}

struct common_agent_sandbox_limits {
    uint32_t timeout_ms = 60000;
    size_t memory_bytes = 0;
    uint32_t cpu_count = 1;
    uint32_t process_count = 0;
    size_t max_output_bytes = 65536;
};

struct common_agent_sandbox_command {
    std::string program;
    std::vector<std::string> arguments;
    std::string working_directory;
};

struct common_agent_sandbox_workspace {
    std::string source_path;
    std::string writable_path;
    std::string artifact_path;
    std::vector<common_runtime_resource_ref> input_resources;
};

struct common_agent_sandbox_artifact_policy {
    bool collect = true;
    size_t max_bytes = 16 * 1024 * 1024;
    std::vector<std::string> paths;
};

struct common_agent_sandbox_request {
    std::string operation_id;
    std::string project_id;
    std::string workspace_id;
    std::string execution_class;
    std::string image;
    common_agent_sandbox_command command;
    common_agent_sandbox_workspace workspace;
    common_agent_sandbox_limits limits;
    common_agent_sandbox_network_scope network = common_agent_sandbox_network_scope::none;
    common_agent_sandbox_filesystem_scope filesystem = common_agent_sandbox_filesystem_scope::readonly;
    common_agent_sandbox_artifact_policy artifacts;
};

inline bool common_agent_sandbox_validate_capabilities(
        const common_agent_sandbox_request & request,
        const common_agent_sandbox_capabilities & capabilities,
        std::string & error) {
    if (!capabilities.process_isolation) {
        error = "sandbox backend does not provide process isolation";
        return false;
    }
    if (!common_agent_sandbox_supports_network(capabilities, request.network)) {
        error = "sandbox backend does not support the requested network scope";
        return false;
    }
    if (!common_agent_sandbox_supports_filesystem(capabilities, request.filesystem)) {
        error = "sandbox backend does not support the requested filesystem scope";
        return false;
    }
    if (request.limits.cpu_count != 0 && !capabilities.cpu_limit) {
        error = "sandbox backend cannot enforce CPU limits";
        return false;
    }
    if (request.limits.memory_bytes != 0 && !capabilities.memory_limit) {
        error = "sandbox backend cannot enforce memory limits";
        return false;
    }
    if (request.limits.process_count != 0 && !capabilities.process_limit) {
        error = "sandbox backend cannot enforce process limits";
        return false;
    }
    if (request.artifacts.collect && !capabilities.artifact_collection) {
        error = "sandbox backend cannot collect artifacts";
        return false;
    }
    error.clear();
    return true;
}

struct common_agent_sandbox_usage {
    uint64_t wall_time_ms = 0;
    uint64_t cpu_time_ms = 0;
    size_t peak_memory_bytes = 0;
    size_t output_bytes = 0;
};

struct common_agent_sandbox_result {
    common_agent_sandbox_status status = common_agent_sandbox_status::failed;
    int exit_code = -1;
    std::string stdout_excerpt;
    std::string stderr_excerpt;
    std::vector<common_runtime_resource_ref> artifacts;
    common_agent_sandbox_usage usage;
    std::string backend_execution_id;
    std::string error;
};

class common_agent_sandbox_runtime {
public:
    virtual ~common_agent_sandbox_runtime() = default;

    virtual common_agent_sandbox_capabilities capabilities() const {
        return {};
    }

    virtual bool execute(
        const common_agent_sandbox_request & request,
        common_agent_sandbox_result & result,
        std::string & error) = 0;
};
