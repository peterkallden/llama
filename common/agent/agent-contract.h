// Backend-neutral agent feature contract for internal callers.
#pragma once

#include "agent/agent-scope.h"
#include "agent/agent-generation.h"
#include "chat.h"
#include "agent/tool-runtime-contract.h"
#include "memory/memory-types.h"
#include "memory/memory-policy-pack.h"
#include "memory/memory-candidate.h"
#include "plan/plan-types.h"
#include "runtime-trace.h"

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

enum class common_learning_signal_type {
    tool_failure,
    successful_recovery,
    reflection_hint,
    user_correction,
};

// Stable, model-safe failure categories. The runtime keeps raw diagnostics
// outside this contract; repair prompts receive only bounded classifications.
enum class common_agent_failure_class {
    validation,
    policy,
    not_found,
    timeout,
    network,
    execution,
    model_format,
    plan_conflict,
    limit,
};

inline const char * common_agent_failure_class_name(common_agent_failure_class value) {
    switch (value) {
        case common_agent_failure_class::validation:    return "validation";
        case common_agent_failure_class::policy:        return "policy";
        case common_agent_failure_class::not_found:     return "not_found";
        case common_agent_failure_class::timeout:       return "timeout";
        case common_agent_failure_class::network:       return "network";
        case common_agent_failure_class::execution:     return "execution";
        case common_agent_failure_class::model_format:  return "model_format";
        case common_agent_failure_class::plan_conflict: return "plan_conflict";
        case common_agent_failure_class::limit:         return "limit";
    }
    return "execution";
}

struct common_agent_failure {
    std::string code;
    common_agent_failure_class classification = common_agent_failure_class::execution;
    std::string stage;
    std::string tool_name;
    std::string step_id;
    std::string evidence_id;
    bool retryable = false;
    std::string safe_summary;
};

// A caller supplies this only when it has an explicit user correction and the
// turn it corrects. It is evidence for post-turn learning, never a direct
// memory write or a model-controlled provenance claim.
struct common_agent_user_correction {
    std::string source_turn_id;
    std::string statement;
};

inline const char * common_learning_signal_type_name(common_learning_signal_type type) {
    switch (type) {
        case common_learning_signal_type::tool_failure: return "tool_failure";
        case common_learning_signal_type::successful_recovery: return "successful_recovery";
        case common_learning_signal_type::reflection_hint: return "reflection_hint";
        case common_learning_signal_type::user_correction: return "user_correction";
    }
    return "unknown";
}

struct common_agent_event {
    common_agent_event_type type = common_agent_event_type::memory_retrieved;
    std::string detail;
    std::string memory_id;
    std::optional<std::string> plan_id;
};

// Native, evidence-addressable events that may inform post-turn learning.
// They are observations, not model-provided instructions or memory writes.
struct common_learning_signal {
    common_learning_signal_type type = common_learning_signal_type::tool_failure;
    std::string plan_id;
    std::string step_id;
    std::string tool_name;
    std::string evidence_id;
    std::string summary;
};

// Host-owned intent.  It is optional so existing callers retain the prompt as
// their effective purpose, while callers with a UI or API can make the user's
// intended outcome and constraints explicit without asking the model to infer
// authority from free-form context.
struct common_agent_objective {
    std::string purpose;
    std::string desired_outcome;
    std::vector<std::string> success_criteria;
    std::vector<std::string> constraints;
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
    std::optional<common_agent_objective> objective;
    std::optional<common_memory_policy_pack> policy_pack;
    std::vector<common_memory_hit> memories;
    std::optional<common_agent_user_correction> user_correction;
    std::optional<common_agent_tool_call> tool_call;
    size_t max_iterations = 2;
    size_t max_reflection_rounds = 1;
    size_t max_tool_batches = 1;
    bool allow_policy_gated_tool_proposals = false;
};

inline common_agent_scope common_agent_scope_from_request(const common_agent_request & request) {
    common_agent_scope scope;
    scope.memory_scope = request.memory_scope;
    scope.plan_scope = request.plan_scope;
    scope.namespace_id = request.namespace_id;
    scope.session_id = request.session_id;
    scope.project_id = request.project_id;
    scope.turn_id = request.turn_id;
    return scope;
}

inline void common_agent_scope_apply(const common_agent_scope & scope, common_agent_request & request) {
    request.memory_scope = scope.memory_scope;
    request.plan_scope = scope.plan_scope;
    request.namespace_id = scope.namespace_id;
    request.session_id = scope.session_id;
    request.project_id = scope.project_id;
    request.turn_id = scope.turn_id;
}

struct common_agent_result {
    std::string response;
    std::string error;
    int total_decoded_tokens = 0;
    int response_decoded_tokens = 0;
    int reasoning_decoded_tokens = 0;
    common_agent_generation_status response_generation_status = common_agent_generation_status::errored;
    common_agent_generation_stop_reason response_stop_reason = common_agent_generation_stop_reason::error;

    std::vector<std::string> memory_ids;
    std::optional<common_memory_candidate> learned_memory_candidate;
    std::string memory_learning_summary;
    size_t memory_learning_related_count = 0;
    std::vector<common_learning_signal> learning_signals;
    std::vector<common_agent_failure> failures;

    std::optional<std::string> plan_id;
    uint64_t plan_version = 0;

    bool reflected = false;
    bool revised = false;
    bool limit_reached = false;
    std::vector<common_agent_generation_record> generation_records;

    // Events contain structured outcomes only, never raw chain-of-thought.
    std::vector<common_agent_event> events;
    std::vector<common_runtime_trace_entry> trace;
};
