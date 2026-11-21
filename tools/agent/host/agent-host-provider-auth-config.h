#pragma once

#include <string>
#include <vector>

// Host-owned authentication selection for an outbound provider. The provider
// contract contains references to secrets, never the secret values themselves.
// OpenAPI security schemes supply placement details; MCP uses the selected
// mechanism directly until protocol-level OAuth discovery is added.
struct agent_host_provider_auth_config {
    std::string type = "none";
    // Optional OpenAPI securitySchemes key. When omitted, the host may use
    // the only compatible scheme declared by the contract.
    std::string scheme;
    std::string token_url;
    std::vector<std::string> scopes;
    std::string token_env;
    std::string username_env;
    std::string password_env;
    std::string client_id_env;
    std::string client_secret_env;
    std::string client_cert_path_env;
    std::string client_key_path_env;
    std::string ca_cert_path_env;
};
