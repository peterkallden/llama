#pragma once

#include "sandbox-policy.h"
#include "agent/workspace-contract.h"

#include <cstdint>
#include <map>
#include <string>

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
};
