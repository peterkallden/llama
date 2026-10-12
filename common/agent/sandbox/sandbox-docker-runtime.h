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

    common_agent_sandbox_capabilities capabilities() const override {
        common_agent_sandbox_capabilities result;
        result.process_isolation = true;
        result.filesystem_readonly = true;
        result.filesystem_workspace_write = true;
        result.filesystem_artifact_write = true;
        result.network_none = true;
        result.cpu_limit = true;
        result.memory_limit = true;
        result.process_limit = true;
        result.artifact_collection = true;
        return result;
    }

    bool execute(
            const common_agent_sandbox_request & request,
            common_agent_sandbox_result & result,
            std::string & error) override;

private:
    common_agent_docker_sandbox_config config;
};
