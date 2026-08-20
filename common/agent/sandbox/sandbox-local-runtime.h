#pragma once

#include "sandbox-contract.h"

class common_agent_sandbox_local_runtime final : public common_agent_sandbox_runtime {
public:
    bool execute(
            const common_agent_sandbox_request & request,
            common_agent_sandbox_result & result,
            std::string & error) override;
};
