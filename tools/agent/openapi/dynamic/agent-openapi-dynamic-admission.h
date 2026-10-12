#pragma once

#include "../agent-openapi-catalog.h"

#include <nlohmann/json.hpp>

#include <map>
#include <mutex>
#include <string>
#include <vector>

struct agent_openapi_dynamic_registration {
    agent_host_openapi_provider_config config;
    nlohmann::json document;
    agent_openapi_catalog catalog;
    std::string source_resource_uri;
    std::string source_content_hash;
};

// Session-owned, in-memory registry. It stores validated parsed specs and
// their filtered catalogs, never credentials or caller-selected endpoints.
class agent_openapi_dynamic_registry {
public:
    bool add(agent_openapi_dynamic_registration registration, std::string & error);
    std::vector<agent_openapi_dynamic_registration> snapshot() const;
    bool contains(const std::string & provider_id) const;
    void clear();

private:
    mutable std::mutex mutex_;
    std::map<std::string, agent_openapi_dynamic_registration> registrations_;
};

// Build a conservative, anonymous, HTTPS/read-only registration from an
// already materialized OpenAPI resource. No network access occurs here.
bool admit_agent_dynamic_openapi_document(
    const nlohmann::json & document,
    const std::string & source_resource_uri,
    const std::string & source_content_hash,
    agent_openapi_dynamic_registration & registration,
    std::string & error);
