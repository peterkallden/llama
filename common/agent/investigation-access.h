#pragma once

#include "agent/data-store.h"
#include "agent/investigation-sources.h"
#include "resource/resource-contract.h"

// Shared host authorization for tool calls and selected chat context. The
// interfaces own scope resolution; the domain only enumerates dependencies.
inline bool common_investigation_authorize_search_sources(
        const common_investigation_case_revision & value, const common_agent_scope & scope,
        const common_agent_data_store & data_store, std::string & error) {
    for (const auto & provider : common_investigation_search_providers(value))
        if (!data_store.authorize_investigation_source(provider, scope, error)) return false;
    error.clear();
    return true;
}

inline bool common_investigation_authorize_sources(
        const common_investigation_case_revision & value, const common_agent_scope & scope,
        const common_agent_data_store & data_store, const agent_resource_store * resource_store,
        const agent_resource_read_authority & authority, std::string & error) {
    if (scope.namespace_id != authority.namespace_id || scope.project_id != authority.project_id ||
            (scope.project_id.empty() && scope.session_id != authority.session_id)) {
        error = "investigation authority does not match its host scope";
        return false;
    }
    if (!common_investigation_authorize_search_sources(value, scope, data_store, error)) return false;
    for (const auto & source : value.sources) {
        if (source.kind == common_investigation_source_kind::search_connector) continue;
        if (!resource_store) { error = "investigation resource store is unavailable"; return false; }
        const auto verify = [&](const std::string & uri, const std::string & expected_hash) {
            agent_resource_descriptor descriptor;
            if (!resource_store->stat(uri, authority, descriptor, error)) return false;
            if (!expected_hash.empty() && expected_hash != "sha256:" + descriptor.sha256) {
                error = "investigation resource failed its retained content hash check";
                return false;
            }
            return true;
        };
        if (!verify(source.resource_uri, source.content_hash)) return false;
        if (!source.representation_uri.empty() &&
                !verify(source.representation_uri, source.representation_hash)) return false;
    }
    error.clear();
    return true;
}
