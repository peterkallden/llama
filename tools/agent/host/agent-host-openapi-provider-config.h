#pragma once

#include <cstdint>
#include <cstddef>
#include <map>
#include <string>

#include "agent-host-provider-auth-config.h"

// Host-owned policy for one OpenAPI provider. Secrets are referenced by
// environment-variable name only; secret values never belong in this object
// or in the model-facing tool catalog.
struct agent_host_openapi_operation_policy {
    bool enabled = true;
    std::string access;
};

struct agent_host_openapi_provider_config {
    std::string type = "openapi";
    std::string id;
    bool enabled = true;
    bool required = false;

    // The first implementation accepts a local JSON document. The provider
    // may later gain a separately approved remote spec source.
    std::string spec_path;
    // Runtime-only origin for resolving relative spec_path values.
    std::string source_directory;
    std::string base_url;
    std::string prefix;

    // access is the global upper bound; exposure controls catalog selection.
    std::string access = "read_only";
    std::string exposure = "auto";
    std::map<std::string, agent_host_openapi_operation_policy> operations;

    agent_host_provider_auth_config auth;
    bool allow_private_network = false;
    uint32_t connect_timeout_ms = 5000;
    uint32_t request_timeout_ms = 30000;
    size_t max_result_bytes = 1024 * 1024;
};
