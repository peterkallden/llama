#pragma once

#include "sandbox-contract.h"

#include <string>
#include <utility>

struct common_agent_lxc_sandbox_config {
    std::string executable = "lxc";
    std::string default_image = "ubuntu:24.04";
    std::string network_mode = "none";
    std::string network_profile;
    // The operator declares which network scope the named profile actually
    // enforces. An arbitrary profile name must not imply every network scope.
    std::string network_profile_scope = "none";
    bool cleanup = true;
};

// LXC/Incus-compatible runtime. The host owns the container name, mounts and
// command translation; the model only supplies the semantic sandbox request.
// Network capabilities are advertised only for an explicitly configured
// profile/mode and unsupported scopes fail closed.
class common_agent_sandbox_lxc_runtime final : public common_agent_sandbox_runtime {
public:
    explicit common_agent_sandbox_lxc_runtime(common_agent_lxc_sandbox_config config = {})
        : config(std::move(config)) {}

    common_agent_sandbox_capabilities capabilities() const override;

    bool execute(
            const common_agent_sandbox_request & request,
            common_agent_sandbox_result & result,
            std::string & error) override;

private:
    common_agent_lxc_sandbox_config config;
};
