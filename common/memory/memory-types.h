#pragma once

#include <cstdint>
#include <ctime>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

enum class common_memory_kind {
    episode,
    fact,
    observation,
    reflection,
    procedure,
    constraint,
    decision,
    goal,
    preference,
};

enum class common_memory_scope {
    turn,
    session,
    project,
    global,
};

struct common_memory_record {
    std::string id;
    common_memory_kind kind = common_memory_kind::episode;

    std::string content;
    std::string summary;

    std::vector<float> embedding;

    float importance = 0.5f;
    float confidence = 0.5f;

    int64_t created_at = 0;
    int64_t accessed_at = 0;
    uint64_t access_count = 0;

    // Scope is a first-class access boundary, never model-controlled metadata.
    common_memory_scope scope = common_memory_scope::session;
    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;

    std::unordered_map<std::string, std::string> metadata;
};

struct common_memory_temporal_fields {
    std::optional<int64_t> observed_at;
    std::optional<int64_t> valid_from;
    std::optional<int64_t> valid_to;
};

inline bool common_memory_temporal_fields_read(
        const common_memory_record & record,
        common_memory_temporal_fields & fields,
        std::string & error) {
    fields = {};
    auto read = [&](const char * key, std::optional<int64_t> & output) {
        const auto found = record.metadata.find(key);
        if (found == record.metadata.end()) return true;
        try {
            size_t used = 0;
            const auto value = std::stoll(found->second, &used);
            if (used != found->second.size()) return false;
            output = value;
            return true;
        } catch (...) { return false; }
    };
    if (!read("agent.temporal.v1.observed_at", fields.observed_at) ||
        !read("agent.temporal.v1.valid_from", fields.valid_from) ||
        !read("agent.temporal.v1.valid_to", fields.valid_to)) {
        error = "memory temporal metadata must contain integer Unix timestamps";
        return false;
    }
    if (fields.valid_from && fields.valid_to && *fields.valid_from >= *fields.valid_to) {
        error = "memory validity interval must satisfy valid_from < valid_to";
        return false;
    }
    error.clear();
    return true;
}

inline bool common_memory_temporal_fields_write(
        common_memory_record & record,
        const common_memory_temporal_fields & fields,
        std::string & error) {
    if (fields.valid_from && fields.valid_to && *fields.valid_from >= *fields.valid_to) {
        error = "memory validity interval must satisfy valid_from < valid_to";
        return false;
    }
    auto write = [&](const char * key, const std::optional<int64_t> & value) {
        if (value) record.metadata[key] = std::to_string(*value);
        else record.metadata.erase(key);
    };
    write("agent.temporal.v1.observed_at", fields.observed_at);
    write("agent.temporal.v1.valid_from", fields.valid_from);
    write("agent.temporal.v1.valid_to", fields.valid_to);
    error.clear();
    return true;
}

struct common_memory_relation {
    std::string from;
    std::string relation;
    std::string to;
    float weight = 0.0f;
    int64_t created_at = 0;
};

struct common_memory_query {
    std::string text;
    std::vector<float> embedding;

    std::optional<common_memory_kind> kind;

    common_memory_scope scope = common_memory_scope::session;
    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;
    bool global_opt_in = false;

    size_t limit = 8;
    float minimum_score = 0.0f;
    size_t graph_depth = 1;
    size_t token_budget = 1024;
    std::optional<int64_t> as_of;
    bool include_history = false;
};

struct common_memory_hit {
    common_memory_record memory;

    float semantic_score = 0.0f;
    float graph_score = 0.0f;
    float recency_score = 0.0f;
    float final_score = 0.0f;

    std::string provenance;
};

const char * common_memory_kind_name(common_memory_kind kind);
bool common_memory_kind_parse(const std::string & value, common_memory_kind & out);
const char * common_memory_scope_name(common_memory_scope scope);
bool common_memory_scope_parse(const std::string & value, common_memory_scope & out);
bool common_memory_scope_matches(const common_memory_record & record, const common_memory_query & query);
float common_memory_cosine_similarity(const std::vector<float> & a, const std::vector<float> & b);
