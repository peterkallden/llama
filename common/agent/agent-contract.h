// Backend-neutral agent feature contract for internal callers.
#pragma once

#include "chat.h"
#include "memory/memory-types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Planning scope is deliberately independent from memory scope. In particular,
// a global plan does not grant access to global memory without separate policy.
enum class common_plan_scope {
    turn,
    session,
    project,
    global,
};

enum class common_agent_event_type {
    memory_retrieved,
    memory_remembered,
    memory_rejected,
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
    common_plan_scope plan_scope = common_plan_scope::session;

    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;

    bool enable_memory = false;
    bool enable_planning = false;
    bool enable_reflection = false;
};

struct common_agent_result {
    std::string response;

    std::vector<std::string> memory_ids;

    std::optional<std::string> plan_id;
    uint64_t plan_version = 0;

    bool reflected = false;
    bool revised = false;

    // Events contain structured outcomes only, never raw chain-of-thought.
    std::vector<common_agent_event> events;
};
