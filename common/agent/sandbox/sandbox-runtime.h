#pragma once

#include "sandbox-contract.h"

// Backend-neutral runtime used when no execution backend has been configured.
// It never falls back to an unsandboxed host process.
class common_agent_sandbox_unavailable_runtime final : public common_agent_sandbox_runtime {
public:
    bool execute(
            const common_agent_sandbox_request & request,
            common_agent_sandbox_result & result,
            std::string & error) override {
        result = {};
        result.status = common_agent_sandbox_status::backend_unavailable;
        result.backend_execution_id = request.operation_id.empty()
            ? "no-backend"
            : "no-backend/" + request.operation_id;
        result.error = "no sandbox execution backend is configured";
        error.clear();
        return true;
    }
};
