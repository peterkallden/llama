#pragma once

#include <string>

// Host-owned authentication selection for an outbound provider. The provider
// contract contains references to secrets, never the secret values themselves.
// OpenAPI security schemes supply placement details; MCP uses the selected
// mechanism directly until protocol-level OAuth discovery is added.
struct agent_host_provider_auth_config {
    std::string type = "none";
    std::string token_env;
};
