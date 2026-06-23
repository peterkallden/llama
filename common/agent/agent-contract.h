// Backend-neutral agent feature contract for internal callers.
#pragma once

#include "chat.h"
#include "agent/tool-registry.h"
#include "memory/memory-types.h"
#include "memory/memory-candidate.h"
#include "plan/plan-types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

enum class common_agent_event_type {
    memory_retrieved,
    memory_remembered,
    memory_rejected,
    memory_candidate_extracted,
    memory_candidate_not_stored,
    blueprint_promoted,
    tool_executed,
    tool_rejected,
    plan_created,
    plan_updated,
    reflection_completed,
    response_revised,
};

struct common_agent_event {
    common_agent_event_type type = common_agent_event_type::memory_retrieved;
    std::string detail;
    std::string memory_id;
    std::optional<std::string> plan_id;
};

struct common_agent_request {
    std::vector<common_chat_msg> messages;

    common_memory_scope memory_scope = common_memory_scope::session;
    // Deliberately independent from memory_scope: a global plan never grants
    // global-memory authority.
    common_plan_scope plan_scope = common_plan_scope::turn;

    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;
    // When present, the runtime resumes this exact persisted plan rather than
    // asking the planner to create a new one.
    std::optional<std::string> plan_id;

    bool enable_memory = false;
    bool enable_planning = true;
    bool enable_reflection = true;

    // Runtime inputs are data only; tool execution remains policy-owned by
    // the caller-supplied registry.
    std::string prompt;
    std::vector<common_memory_hit> memories;
    std::optional<common_registered_tool_call> tool_call;
    size_t max_iterations = 2;
    size_t max_reflection_rounds = 1;
    size_t max_tool_batches = 1;
    bool allow_policy_gated_tool_proposals = false;
};

struct common_agent_result {
    std::string response;
    std::string error;

    std::vector<std::string> memory_ids;
    std::optional<common_memory_candidate> learned_memory_candidate;
    std::string memory_learning_summary;
    size_t memory_learning_related_count = 0;

    std::optional<std::string> plan_id;
    uint64_t plan_version = 0;

    bool reflected = false;
    bool revised = false;
    bool limit_reached = false;

    // Events contain structured outcomes only, never raw chain-of-thought.
    std::vector<common_agent_event> events;
};
