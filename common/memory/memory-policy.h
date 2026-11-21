#pragma once

#include "memory/memory-retrieval.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct common_memory_remember_request {
    common_memory_kind kind = common_memory_kind::fact;
    std::string content;
    std::string rationale;
    float importance = 0.5f;
    float confidence = 0.5f;
    std::string source_role = "assistant";
    std::string source_turn_id;
    common_memory_scope scope = common_memory_scope::session;
    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;
    bool global_opt_in = false;
};

enum class common_memory_remember_decision {
    accept,
    reject,
    duplicate,
    conflict,
};

struct common_memory_remember_result {
    common_memory_remember_decision decision = common_memory_remember_decision::reject;
    std::string reason;
    std::optional<common_memory_record> record;
    std::vector<common_memory_hit> related_hits;
};

const char * common_memory_remember_decision_name(common_memory_remember_decision decision);

common_memory_remember_result common_memory_evaluate_remember_request(
    common_memory_store & store,
    const common_memory_remember_request & request,
    const std::vector<float> & embedding,
    int64_t now,
    std::string & error);
