#pragma once

#include "resource-contract.h"

#include <cstddef>
#include <string>
#include <utility>

struct agent_resource_retention_target {
    common_runtime_resource_scope scope = common_runtime_resource_scope::session;
    std::string namespace_id;
    std::string session_id;
    std::string project_id;
    std::string turn_id;
    std::string source_tool;
    size_t max_bytes = 0;
};

// Retain an authorized resource at the requested lifetime. Broader-scoped
// resources remain referenced directly; narrower resources are copied with
// immutable lineage to the requested scope.
inline bool retain_agent_resource(
        agent_resource_store & store,
        const common_runtime_resource_ref & source,
        const agent_resource_read_authority & authority,
        const agent_resource_retention_target & target,
        agent_resource_descriptor & retained,
        bool & copied,
        std::string & error) {
    const auto scope_rank = [](common_runtime_resource_scope scope) {
        switch (scope) {
            case common_runtime_resource_scope::turn: return 0;
            case common_runtime_resource_scope::session: return 1;
            case common_runtime_resource_scope::project: return 2;
        }
        return -1;
    };
    retained = {};
    copied = false;
    if (source.uri.empty() || target.namespace_id.empty() || target.max_bytes == 0 ||
            target.namespace_id != authority.namespace_id ||
            scope_rank(target.scope) < 0 ||
            (target.scope == common_runtime_resource_scope::session && target.session_id.empty()) ||
            (target.scope == common_runtime_resource_scope::project && target.project_id.empty()) ||
            (target.scope == common_runtime_resource_scope::turn &&
                (target.session_id != authority.session_id || target.turn_id != authority.turn_id)) ||
            (target.scope == common_runtime_resource_scope::session && target.session_id != authority.session_id) ||
            (target.scope == common_runtime_resource_scope::project && target.project_id != authority.project_id)) {
        error = "resource retention request has an invalid source, target scope or byte limit";
        return false;
    }
    agent_resource_descriptor original;
    if (!store.stat(source.uri, authority, original, error)) return false;
    if (original.scope != source.scope || (!source.mime_type.empty() && original.mime_type != source.mime_type) ||
            (source.size_bytes != 0 && original.size_bytes != source.size_bytes)) {
        error = "resource reference metadata does not match its host resource";
        return false;
    }
    if (scope_rank(original.scope) >= scope_rank(target.scope)) {
        retained = std::move(original);
        error.clear();
        return true;
    }
    if (original.size_bytes > target.max_bytes) {
        error = "resource exceeds the configured retention byte limit";
        return false;
    }
    std::string bytes;
    if (!store.read_bytes(original.uri, authority, original.size_bytes + 1, bytes, error)) return false;
    if (bytes.size() != original.size_bytes) {
        error = "resource size changed while it was prepared for retention";
        return false;
    }
    agent_resource_put_request request;
    request.name = original.name;
    request.description = original.description;
    request.mime_type = original.mime_type;
    request.bytes = std::move(bytes);
    request.bytes_are_authoritative = true;
    request.scope = target.scope;
    request.namespace_id = target.namespace_id;
    request.session_id = target.session_id;
    request.project_id = target.project_id;
    request.turn_id = target.turn_id;
    request.source_provider = original.source_provider;
    request.source_tool = target.source_tool;
    request.created_at = original.created_at;
    request.metadata = original.metadata;
    request.metadata.content_hash = original.sha256.empty() ? std::string{} : "sha256:" + original.sha256;
    request.lineage.parent_uri = original.uri;
    request.lineage.derivation = target.scope == common_runtime_resource_scope::project
        ? "research-source-project-promotion:v1" : "research-source-chat-retention:v1";
    if (!store.put_bytes(request, retained, error) || retained.uri.empty()) {
        if (error.empty()) error = "resource store returned an empty retained resource reference";
        return false;
    }
    copied = true;
    error.clear();
    return true;
}
