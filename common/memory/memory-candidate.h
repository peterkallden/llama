// Backend-neutral durable-memory candidate types.
#pragma once

#include "agent/agent-generation.h"
#include "memory/memory-types.h"

#include <optional>
#include <string>
#include <vector>

struct common_memory_candidate {
    common_memory_kind kind = common_memory_kind::procedure;
    std::string content;
    std::string rationale;

    float importance = 0.5f;
    float confidence = 0.5f;
    float expected_reuse = 0.5f;

    // Provenance is evidence, never executable instruction text.
    bool explicit_user_provenance = false;
    std::vector<std::string> evidence_ids;
    std::vector<std::string> source_plan_step_ids;
};

struct common_memory_candidate_result {
    std::optional<common_memory_candidate> candidate;
    std::string reason;
    std::optional<common_agent_generated_text_result> generation;
};
