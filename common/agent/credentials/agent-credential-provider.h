#pragma once

#include <string>

// Host-owned credential lookup. Portable agent configuration carries only a
// reference; the provider decides whether the secret comes from Android
// Keystore, a desktop keyring, or an explicitly configured development store.
class common_agent_credential_provider {
public:
    virtual ~common_agent_credential_provider() = default;

    virtual bool resolve(
        const std::string & credential_ref,
        std::string & secret,
        std::string & error) = 0;
};
