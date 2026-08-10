#pragma once

#include "agent/sandbox-contract.h"
#include "tools/agent/tooling/agent-sandbox-helper.h"
#include "agent/workspace-contract.h"

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
            std::string & error) override {
        if (context.operation_id.empty()) {
            error = "resource processing execution requires an operation id";
            return false;
        }
        request.operation_id = context.operation_id;
        return helper.run_raw_for_workspace(
            context.workspace, context.operation_id, std::move(request), result, error);
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
