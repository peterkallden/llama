#pragma once

#include "sandbox-contract.h"

#include <string>
#include <utility>

struct common_agent_docker_sandbox_config {
    std::string executable = "docker";
    std::string default_image;
};

// Docker-backed implementation of the host-owned sandbox contract. The
// request remains semantic: Docker flags and mount locations are derived here
// and are never supplied by the model.
class common_agent_sandbox_docker_runtime final : public common_agent_sandbox_runtime {
public:
    explicit common_agent_sandbox_docker_runtime(common_agent_docker_sandbox_config config = {})
        : config(std::move(config)) {}

    bool execute(
            const common_agent_sandbox_request & request,
            common_agent_sandbox_result & result,
            std::string & error) override;

private:
    common_agent_docker_sandbox_config config;
};
