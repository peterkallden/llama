#pragma once

#include "agent/sandbox/sandbox-contract.h"
#include "tools/agent/tooling/agent-sandbox-helper.h"
#include "agent/workspace-contract.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <utility>

struct agent_resource_processing_host_artifact {
    std::string name;
    std::string mime_type;
    std::string bytes;
    common_runtime_resource_ref resource;
};

// Host-owned execution context for one resource-processing operation. It is
// deliberately operation-scoped; it is not a second queue or a processor
// lifetime store.
struct agent_resource_processing_execution_context {
    common_agent_workspace_context workspace;
    std::string operation_id;
};

class agent_resource_processing_host {
public:
    virtual ~agent_resource_processing_host() = default;

    virtual bool execute(
            const agent_resource_processing_execution_context & context,
            common_agent_sandbox_request request,
            common_agent_sandbox_result & result,
            std::vector<agent_resource_processing_host_artifact> & artifacts,
            std::string & error) = 0;
};

class agent_sandbox_resource_processing_host final : public agent_resource_processing_host {
public:
    agent_sandbox_resource_processing_host(
            common_agent_sandbox_runtime & runtime,
            common_agent_sandbox_policy policy)
        : helper(runtime, std::move(policy)) {}

    bool execute(
            const agent_resource_processing_execution_context & context,
            common_agent_sandbox_request request,
            common_agent_sandbox_result & result,
            std::vector<agent_resource_processing_host_artifact> & artifacts,
            std::string & error) override {
        if (context.operation_id.empty()) {
            error = "resource processing execution requires an operation id";
            return false;
        }
        request.operation_id = context.operation_id;
        common_agent_workspace_operation operation;
        if (!helper.run_raw_for_workspace(
                context.workspace,
                context.operation_id,
                request,
                result,
                error,
                &operation)) {
            return false;
        }
        if (result.status != common_agent_sandbox_status::completed ||
                !request.artifacts.collect) {
            return true;
        }
        for (const auto & artifact_name : request.artifacts.paths) {
            const std::filesystem::path relative(artifact_name);
            if (relative.empty() || relative.is_absolute() ||
                    std::find(relative.begin(), relative.end(), "..") != relative.end()) {
                error = "resource processor returned an unsafe artifact path";
                return false;
            }
            const auto path = std::filesystem::path(operation.artifact_path) / relative;
            std::ifstream input(path, std::ios::binary);
            if (!input) {
                error = "resource processor artifact was not produced: " + artifact_name;
                return false;
            }
            std::string bytes(
                (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (request.artifacts.max_bytes > 0 &&
                    bytes.size() > request.artifacts.max_bytes) {
                error = "resource processor artifact exceeded its configured limit";
                return false;
            }
            artifacts.push_back({
                artifact_name,
                artifact_name.size() >= 4 && artifact_name.substr(artifact_name.size() - 4) == ".png"
                    ? "image/png"
                    : "application/octet-stream",
                std::move(bytes),
                {},
            });
        }
        return true;
    }

    void set_workspace_manager(common_agent_workspace_manager * manager) {
        helper.set_workspace_manager(manager);
    }

    void set_resource_store(
            agent_resource_store * store,
            agent_resource_read_authority authority = {}) {
        helper.set_resource_store(store, std::move(authority));
    }

private:
    common_agent_sandbox_tool_helper helper;
};
