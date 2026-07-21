#pragma once

#include <cstdint>
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
