#pragma once

#include "runtime/runtime-state.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class agent_resource_store;

enum class common_runtime_resource_scope {
    turn,
    session,
    project,
};

inline const char * common_runtime_resource_scope_name(common_runtime_resource_scope scope) {
    switch (scope) {
        case common_runtime_resource_scope::turn:    return "turn";
        case common_runtime_resource_scope::session: return "session";
        case common_runtime_resource_scope::project: return "project";
    }
    return "turn";
}

enum class agent_resource_blob_backend {
    auto_,
    in_memory,
    fs,
    s3,
};

inline const char * agent_resource_blob_backend_name(agent_resource_blob_backend backend) {
    switch (backend) {
        case agent_resource_blob_backend::auto_:     return "auto";
        case agent_resource_blob_backend::in_memory: return "in-memory";
        case agent_resource_blob_backend::fs:        return "fs";
        case agent_resource_blob_backend::s3:        return "s3";
    }
    return "auto";
}

inline bool parse_agent_resource_blob_backend(
        const std::string & value,
        agent_resource_blob_backend & backend) {
    if (value == "auto") {
        backend = agent_resource_blob_backend::auto_;
        return true;
    }
    if (value == "in-memory") {
        backend = agent_resource_blob_backend::in_memory;
        return true;
    }
    if (value == "fs") {
        backend = agent_resource_blob_backend::fs;
        return true;
    }
    if (value == "s3") {
        backend = agent_resource_blob_backend::s3;
        return true;
    }
    return false;
}

enum class agent_resource_metadata_backend {
    auto_,
    in_memory,
    cozo,
};

inline const char * agent_resource_metadata_backend_name(agent_resource_metadata_backend backend) {
    switch (backend) {
        case agent_resource_metadata_backend::auto_:     return "auto";
        case agent_resource_metadata_backend::in_memory: return "in-memory";
        case agent_resource_metadata_backend::cozo:      return "cozo";
    }
    return "auto";
}

inline bool parse_agent_resource_metadata_backend(
        const std::string & value,
        agent_resource_metadata_backend & backend) {
    if (value == "auto") {
        backend = agent_resource_metadata_backend::auto_;
        return true;
    }
    if (value == "in-memory") {
        backend = agent_resource_metadata_backend::in_memory;
        return true;
    }
    if (value == "cozo") {
        backend = agent_resource_metadata_backend::cozo;
        return true;
    }
    return false;
}

struct agent_resource_store_config {
    std::string blob_backend = "auto";
    std::string blob_root;
    std::string metadata_backend = "auto";
    std::string metadata_db;
};

struct common_runtime_resource_metadata {
    std::string purpose;
    std::string content_summary;
    std::string usage_hint;
    std::string limitations;
    std::vector<std::string> keywords;
    std::vector<std::string> entities;
};

struct common_runtime_resource_ref {
    std::string uri;
    std::string name;
    std::string description;
    std::string mime_type;
    size_t size_bytes = 0;
    common_runtime_resource_scope scope = common_runtime_resource_scope::turn;
    common_runtime_resource_metadata metadata;
};

struct agent_blob_descriptor {
    std::string sha256;
    size_t size_bytes = 0;
};

struct agent_resource_put_request {
    std::string name;
    std::string description;
    std::string mime_type = "text/plain";
    std::string text;

    common_runtime_resource_scope scope = common_runtime_resource_scope::turn;

    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;
    std::string tool_call_id;

    std::string source_provider;
    std::string source_tool;

    int64_t created_at = 0;
    int64_t expires_at = 0;
    common_runtime_resource_metadata metadata;
};

struct agent_resource_descriptor : common_runtime_resource_ref {
    std::string resource_id;
    std::string sha256;

    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;
    std::string tool_call_id;

    std::string source_provider;
    std::string source_tool;

    int64_t created_at = 0;
    int64_t expires_at = 0;
};

inline common_agent_state_descriptor describe_agent_resource(
        const agent_resource_descriptor & resource) {
    common_agent_state_descriptor descriptor;
    descriptor.state_id = resource.resource_id.empty() ? resource.uri : resource.resource_id;
    descriptor.state_type = "resource";
    descriptor.state_class = common_agent_state_class::durable_domain;
    descriptor.lifetime = resource.scope == common_runtime_resource_scope::turn
        ? common_agent_state_lifetime::turn
        : resource.scope == common_runtime_resource_scope::session
            ? common_agent_state_lifetime::session
            : common_agent_state_lifetime::project;
    descriptor.persistence = resource.scope == common_runtime_resource_scope::turn
        ? common_agent_state_persistence::none
        : common_agent_state_persistence::checkpointable;
    descriptor.identity.namespace_id = resource.namespace_id;
    descriptor.identity.project_id = resource.project_id;
    descriptor.identity.session_id = resource.session_id;
    descriptor.identity.turn_id = resource.turn_id;
    descriptor.owner = "agent_resource_store";
    descriptor.source_of_truth = "resource metadata and blob store";
    return descriptor;
}

struct agent_resource_read_authority {
    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;
    int64_t now = 0;
};

struct agent_resource_runtime {
    agent_resource_store * store = nullptr;
    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;
};

inline agent_resource_read_authority make_agent_resource_read_authority(
        const agent_resource_runtime & runtime,
        int64_t now = 0) {
    agent_resource_read_authority authority;
    authority.namespace_id = runtime.namespace_id;
    authority.session_id = runtime.session_id;
    authority.project_id = runtime.project_id;
    authority.turn_id = runtime.turn_id;
    authority.now = now;
    return authority;
}

inline void apply_agent_resource_runtime(
        const agent_resource_runtime & runtime,
        agent_resource_put_request & request) {
    request.namespace_id = runtime.namespace_id;
    request.session_id = runtime.session_id;
    request.project_id = runtime.project_id;
    request.turn_id = runtime.turn_id;
}

class agent_blob_store {
public:
    virtual ~agent_blob_store() = default;

    virtual bool put_bytes(
        const std::string & bytes,
        agent_blob_descriptor & out,
        std::string & error) = 0;

    virtual bool get_bytes(
        const std::string & sha256,
        size_t max_bytes,
        std::string & out,
        std::string & error) const = 0;

    virtual bool exists_sha256(const std::string & sha256) const = 0;
};

class agent_resource_store {
public:
    virtual ~agent_resource_store() = default;

    virtual bool put_text(
        const agent_resource_put_request & request,
        agent_resource_descriptor & out,
        std::string & error) = 0;

    virtual bool read_text(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t max_bytes,
        std::string & out,
        std::string & error) const = 0;

    virtual bool stat(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        agent_resource_descriptor & out,
        std::string & error) const = 0;

    virtual bool list(
        const agent_resource_read_authority & authority,
        std::vector<agent_resource_descriptor> & out,
        std::string & error) const = 0;
};
