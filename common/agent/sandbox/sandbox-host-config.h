#pragma once

#include "sandbox-policy.h"
#include "sandbox-contract.h"
#include "agent/workspace-contract.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct common_agent_development_target_profile {
    std::string build_program;
    std::vector<std::string> build_arguments;
    std::string test_program;
    std::vector<std::string> test_arguments;
    std::string working_directory = "/workspace/source";
    std::map<std::string, std::vector<std::string>> configurations;
    std::map<std::string, std::vector<std::string>> test_filters;
    uint32_t max_timeout_ms = 1200000;
};

inline bool common_agent_sandbox_resolve_development_target(
        common_agent_sandbox_request & request,
        const std::map<std::string, common_agent_development_target_profile> & profiles,
        std::string & error) {
    const bool is_build = request.command.program == "agent.development.build";
    const bool is_test = request.command.program == "agent.development.test";
    if (!is_build && !is_test) { error.clear(); return true; }
    if (request.command.arguments.empty()) { error = "development target is missing"; return false; }
    const auto profile_it = profiles.find(request.command.arguments.front());
    if (profile_it == profiles.end()) { error = "development target is not configured"; return false; }
    const auto & profile = profile_it->second;
    const auto & program = is_test ? profile.test_program : profile.build_program;
    if (program.empty()) { error = "development operation is not configured for this target"; return false; }
    std::vector<std::string> arguments = is_test ? profile.test_arguments : profile.build_arguments;
    bool saw_configuration = false;
    bool saw_filter = false;
    for (size_t index = 1; index < request.command.arguments.size(); ++index) {
        const auto & option = request.command.arguments[index];
        constexpr std::string_view configuration_prefix = "--configuration=";
        constexpr std::string_view filter_prefix = "--filter=";
        if (option.rfind(configuration_prefix.data(), 0) == 0) {
            const auto name = option.substr(configuration_prefix.size());
            const auto found = profile.configurations.find(name);
            if (saw_configuration || found == profile.configurations.end()) {
                error = "development configuration is not enabled for this target";
                return false;
            }
            saw_configuration = true;
            arguments.insert(arguments.end(), found->second.begin(), found->second.end());
        } else if (option.rfind(filter_prefix.data(), 0) == 0 && is_test) {
            const auto name = option.substr(filter_prefix.size());
            const auto found = profile.test_filters.find(name);
            if (saw_filter || found == profile.test_filters.end()) {
                error = "development test filter is not enabled for this target";
                return false;
            }
            saw_filter = true;
            arguments.insert(arguments.end(), found->second.begin(), found->second.end());
        } else {
            error = "development request contains an unsupported argument";
            return false;
        }
    }
    request.command.program = program;
    request.command.arguments = std::move(arguments);
    request.command.working_directory = profile.working_directory;
    request.limits.timeout_ms = std::min(request.limits.timeout_ms, profile.max_timeout_ms);
    error.clear();
    return true;
}

// Backend selection and workspace policy owned by the host. This neutral
// shape is carried through CLI, daemon and MCP host options.
struct common_agent_sandbox_host_config {
    std::string backend = "none";
    std::string docker_executable = "docker";
    std::string docker_default_image;
    std::string lxc_executable = "lxc";
    std::string lxc_default_image = "ubuntu:24.04";
    std::string lxc_network_mode = "none";
    std::string lxc_network_profile;
    std::string lxc_network_profile_scope = "none";
    bool lxc_cleanup = true;
    std::string kubernetes_executable = "kubectl";
    std::string kubernetes_kubeconfig;
    std::string kubernetes_context;
    bool kubernetes_insecure_skip_tls_verify = false;
    std::string kubernetes_namespace = "default";
    std::string kubernetes_service_account;
    std::string kubernetes_runtime_class;
    std::string kubernetes_storage_class;
    std::string kubernetes_workspace_storage_size = "4Gi";
    std::string kubernetes_artifact_storage_size = "1Gi";
    std::string kubernetes_staging_image = "alpine:3.20";
    std::string kubernetes_pvc_retention = "project";
    uint32_t kubernetes_staging_timeout_ms = 120000;
    bool kubernetes_cleanup = true;
    common_agent_workspace_roots workspace;
    common_agent_sandbox_policy defaults;
    std::map<std::string, common_agent_sandbox_policy> classes;
    // Keys are host-approved logical targets; commands and argument vectors
    // are operator-owned and never accepted from model tool arguments.
    std::map<std::string, common_agent_development_target_profile> development_targets;
};
