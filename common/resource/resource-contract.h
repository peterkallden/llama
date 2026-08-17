#pragma once

#include "runtime/runtime-state.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class agent_resource_store;
struct agent_resource_processing_binding_request;
class agent_resource_processing_provider;

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

    // Host-owned identity for a reusable derived representation. This is
    // provenance metadata, not a second resource authority.
    std::string processing_cache_key;

    // Language hints are metadata about the resource representation. A
    // declared value may come from ingestion/user context; a resolved value
    // is host-accepted processor metadata and must retain its source.
    std::string declared_language;
    std::string resolved_language;
    double language_confidence = 0.0;
    std::string language_source;
};

// Derived resources, such as bounded chunks, retain explicit lineage to the
// original resource.  Lineage is metadata, not a second source of truth.
struct common_runtime_resource_lineage {
    std::string parent_uri;
    size_t chunk_index = 0;
    size_t chunk_count = 0;
    size_t byte_offset = 0;
    size_t byte_length = 0;
    size_t overlap_bytes = 0;
    std::string derivation;
};

inline bool common_runtime_resource_lineage_is_valid(
        const common_runtime_resource_lineage & lineage,
        std::string & error) {
    if (lineage.parent_uri.empty()) {
        error.clear();
        return true;
    }
    if (lineage.chunk_count == 0) {
        error = "resource lineage requires a non-zero chunk_count";
        return false;
    }
    if (lineage.chunk_index >= lineage.chunk_count) {
        error = "resource lineage chunk_index is outside chunk_count";
        return false;
    }
    if (lineage.byte_length == 0) {
        error = "resource lineage requires a non-zero byte_length";
        return false;
    }
    if (lineage.derivation.empty()) {
        error = "resource lineage requires a derivation label";
        return false;
    }
    error.clear();
    return true;
}

struct common_runtime_resource_ref {
    std::string uri;
    std::string name;
    std::string description;
    std::string mime_type;
    size_t size_bytes = 0;
    common_runtime_resource_scope scope = common_runtime_resource_scope::turn;
    common_runtime_resource_metadata metadata;
    common_runtime_resource_lineage lineage;
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
    common_runtime_resource_lineage lineage;

    // Payload bytes are opaque to the resource store. Text callers continue
    // to use `text`; byte-oriented callers use `bytes`, including embedded NULs.
    // Kept at the end so existing aggregate initializers remain valid.
    std::string bytes;
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
    std::function<std::shared_ptr<agent_resource_processing_provider>(
        const agent_resource_processing_binding_request &)> processing_provider_factory;
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

struct common_runtime_resource_media_type {
    std::string declared_type;
    std::string resolved_type;
    bool content_verified = false;
};

inline std::string common_normalize_resource_media_type(std::string value) {
    const auto semicolon = value.find(';');
    if (semicolon != std::string::npos) value.resize(semicolon);
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
    value = value.substr(begin, end - begin);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline bool common_resource_media_type_is_text_like(const std::string & media_type) {
    const auto normalized = common_normalize_resource_media_type(media_type);
    if (normalized.empty()) return false;
    if (normalized.rfind("text/", 0) == 0) return true;
    if (normalized == "application/json" ||
            normalized == "application/ld+json" ||
            normalized == "application/xml" ||
            normalized == "application/xhtml+xml" ||
            normalized == "application/javascript" ||
            normalized == "application/ecmascript" ||
            normalized == "application/x-ndjson" ||
            normalized == "application/yaml" ||
            normalized == "application/x-yaml") {
        return true;
    }
    return normalized.size() > 5 && (
        normalized.compare(normalized.size() - 5, 5, "+json") == 0 ||
        normalized.compare(normalized.size() - 4, 4, "+xml") == 0);
}

struct agent_resource_byte_range {
    size_t offset = 0;
    size_t max_bytes = 0;
};

struct agent_resource_processing_limits {
    size_t max_source_bytes = 0;
    size_t max_output_bytes = 0;
    size_t max_generated_resources = 0;
    size_t max_duration_ms = 0;
    size_t max_pages = 0;
    size_t max_page_bytes = 0;
};

enum class agent_resource_processing_purpose {
    normalization,
    artifact_generation,
    preview,
};

// Host-owned execution policy for an external resource processor. This does
// not make the processor a model-selected tool; it only selects where the
// host may run its typed processor request.
struct agent_resource_processor_execution_policy {
    std::string execution = "local_preferred";
    std::string backend = "auto";
    std::string executable;
    // Optional typed helper/script path for processors that use an external
    // implementation. It is host configuration, never model input.
    std::string script;
    std::string image;
    std::string expected_version;
};

struct agent_resource_processing_request {
    common_runtime_resource_ref source;
    agent_resource_read_authority authority;
    common_runtime_resource_media_type media_type;
    std::string target_representation = "text";
    std::string target_media_type;
    agent_resource_processing_purpose purpose = agent_resource_processing_purpose::normalization;
    std::optional<size_t> page;
    std::optional<agent_resource_byte_range> range;
    agent_resource_processing_limits limits;

    // The host may provide a bounded source slice to a local processor. The
    // authoritative source remains in the resource store; processors must
    // not treat this field as a second source of truth.
    std::string source_bytes;
};

struct agent_resource_processing_output {
    std::string name;
    std::string description;
    std::string mime_type;
    std::string bytes;
    common_runtime_resource_metadata metadata;
    common_runtime_resource_lineage lineage;
};

struct agent_resource_processing_result {
    bool success = false;
    std::vector<common_runtime_resource_ref> resources;
    std::vector<agent_resource_processing_output> outputs;
    std::string failure_code;
    std::string safe_summary;
    std::string processor_id;
};

// Narrow host/runtime seam for consumers such as resource_read. The full
// processing service may add event emission and other host concerns, but the
// common tool contract only needs a semantic representation request.
struct agent_resource_processing_binding_request {
    std::string source_uri;
    // Host-owned operation identity. It is derived from the active tool turn;
    // model arguments never choose the workspace or execution backend.
    std::string operation_id;
    agent_resource_read_authority authority;
    common_runtime_resource_media_type media_type;
    std::string target_representation = "text";
    std::string target_media_type;
    agent_resource_processing_purpose purpose = agent_resource_processing_purpose::normalization;
    std::optional<size_t> page;
    std::optional<agent_resource_byte_range> range;
    agent_resource_processing_limits limits;
};

class agent_resource_processing_provider {
public:
    virtual ~agent_resource_processing_provider() = default;

    virtual agent_resource_processing_result process(
        const agent_resource_processing_binding_request & request) const = 0;
};

// Per-read construction seam for processors that need an operation-scoped
// workspace or sandbox host. The returned provider is short-lived and remains
// behind the semantic resource-processing contract.
using agent_resource_processing_provider_factory = std::function<
    std::shared_ptr<agent_resource_processing_provider>(
        const agent_resource_processing_binding_request & request)>;

class agent_resource_processor {
public:
    virtual ~agent_resource_processor() = default;

    virtual std::string id() const = 0;

    // Includes implementation/options identity when a processor has
    // operation-bound typed options. The default is sufficient for fixed
    // processors whose id already contains their version.
    virtual std::string cache_key() const { return id(); }

    struct support {
        bool supported = false;
        int priority = 0;
        bool lossy = false;
        bool requires_sandbox = false;
    };

    virtual bool supports(
            const std::string & mime_type,
            const std::string & target_representation) const = 0;

    virtual support supports(
            const agent_resource_processing_request & request) const {
        const std::string & resolved_type = request.media_type.resolved_type.empty()
            ? request.source.mime_type
            : request.media_type.resolved_type;
        return {
            supports(resolved_type, request.target_representation),
            0,
            false,
            false,
        };
    }

    virtual agent_resource_processing_result process(
        const agent_resource_processing_request & request) const = 0;
};

class agent_resource_processor_registry {
public:
    bool add(const agent_resource_processor & processor, std::string & error) {
        const auto processor_id = processor.id();
        if (processor_id.empty()) {
            error = "resource processor id is required";
            return false;
        }
        for (const auto * existing : processors_) {
            if (existing != nullptr && existing->id() == processor_id) {
                error = "duplicate resource processor id";
                return false;
            }
        }
        processors_.push_back(&processor);
        error.clear();
        return true;
    }

    const agent_resource_processor * resolve(
            const std::string & mime_type,
            const std::string & target_representation) const {
        for (const auto * processor : processors_) {
            if (processor != nullptr && processor->supports(mime_type, target_representation)) {
                return processor;
            }
        }
        return nullptr;
    }

    const agent_resource_processor * resolve(
            const agent_resource_processing_request & request) const {
        const agent_resource_processor * selected = nullptr;
        agent_resource_processor::support selected_support;
        for (const auto * processor : processors_) {
            if (processor == nullptr) continue;
            const auto candidate = processor->supports(request);
            if (!candidate.supported ||
                    (selected != nullptr && candidate.priority <= selected_support.priority)) {
                continue;
            }
            selected = processor;
            selected_support = candidate;
        }
        return selected;
    }

    size_t size() const {
        return processors_.size();
    }

private:
    std::vector<const agent_resource_processor *> processors_;
};

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

    virtual bool get_bytes_range(
        const std::string & sha256,
        size_t offset,
        size_t max_bytes,
        std::string & out,
        std::string & error) const {
        if (offset != 0) {
            error = "blob range reads are not supported by this backend";
            return false;
        }
        return get_bytes(sha256, max_bytes, out, error);
    }

    virtual bool exists_sha256(const std::string & sha256) const = 0;
};

class agent_resource_store {
public:
    virtual ~agent_resource_store() = default;

    // The generic resource boundary is byte-oriented.  The default adapter
    // keeps older text-only test stores source-compatible until they migrate.
    virtual bool put_bytes(
        const agent_resource_put_request & request,
        agent_resource_descriptor & out,
        std::string & error) {
        agent_resource_put_request text_request = request;
        text_request.text = request.bytes;
        return put_text(text_request, out, error);
    }

    virtual bool read_bytes(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t max_bytes,
        std::string & out,
        std::string & error) const {
        return read_text(uri, authority, max_bytes, out, error);
    }

    virtual bool read_bytes_range(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t offset,
        size_t max_bytes,
        std::string & out,
        std::string & error) const {
        return read_text_range(uri, authority, offset, max_bytes, out, error);
    }

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

    virtual bool read_text_range(
        const std::string & uri,
        const agent_resource_read_authority & authority,
        size_t offset,
        size_t max_bytes,
        std::string & out,
        std::string & error) const {
        if (offset != 0) {
            error = "resource range reads are not supported by this store";
            return false;
        }
        return read_text(uri, authority, max_bytes, out, error);
    }

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
