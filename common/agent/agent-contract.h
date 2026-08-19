// Backend-neutral agent feature contract for internal callers.
#pragma once

#include "agent/agent-scope.h"
#include "agent/agent-generation.h"
#include "chat.h"
#include "agent/tooling/contracts/tool-runtime-contract.h"
#include "memory/memory-types.h"
#include "memory/memory-policy-pack.h"
#include "memory/memory-candidate.h"
#include "plan/plan-types.h"
#include "runtime/runtime-trace.h"
#include "agent/thinking/deliberation-policy.h"
#include "agent/thinking/research/research-contract.h"
#include "agent/thinking/research/research-verifier.h"
#include "agent/turn-summary.h"
#include "agent/agent-continuation.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

enum class common_agent_event_type {
    memory_retrieved,
    memory_remembered,
    memory_rejected,
    memory_candidate_extracted,
    memory_candidate_not_stored,
    memory_capture_confirmation_required,
    memory_capture_confirmed,
    blueprint_promoted,
    tool_executed,
    tool_rejected,
    tool_repair_context_created,
    plan_created,
    plan_updated,
    observation_recorded,
    resource_chunk_planned,
    resource_chunk_processed,
    resource_created,
    resource_attached,
    reflection_completed,
    response_revised,
    step_reviewed,
    answer_reviewed,
    plan_revision_requested,
    plan_revision_limit_reached,
    research_started,
    research_reopened,
    thinking_mode_resolved,
    thinking_escalation_allowed,
    thinking_escalation_denied,
    thinking_escalation_requested,
    blueprint_selection_evaluated,
    research_gap_opened,
    research_task_scheduled,
    research_task_started,
    research_task_completed,
    research_task_failed,
    research_iteration_completed,
    research_sources_compared,
    research_source_recorded,
    research_evidence_recorded,
    research_completed,
    research_incomplete,
    resource_processing_started,
    resource_processing_completed,
    resource_processing_failed,
};

inline const char * common_agent_event_type_name(common_agent_event_type type) {
    switch (type) {
        case common_agent_event_type::memory_retrieved: return "memory_retrieved";
        case common_agent_event_type::memory_remembered: return "memory_remembered";
        case common_agent_event_type::memory_rejected: return "memory_rejected";
        case common_agent_event_type::memory_candidate_extracted: return "memory_candidate_extracted";
        case common_agent_event_type::memory_candidate_not_stored: return "memory_candidate_not_stored";
        case common_agent_event_type::memory_capture_confirmation_required: return "memory_capture_confirmation_required";
        case common_agent_event_type::memory_capture_confirmed: return "memory_capture_confirmed";
        case common_agent_event_type::blueprint_promoted: return "blueprint_promoted";
        case common_agent_event_type::tool_executed: return "tool_executed";
        case common_agent_event_type::tool_rejected: return "tool_rejected";
        case common_agent_event_type::tool_repair_context_created: return "tool_repair_context_created";
        case common_agent_event_type::plan_created: return "plan_created";
        case common_agent_event_type::plan_updated: return "plan_updated";
        case common_agent_event_type::observation_recorded: return "observation_recorded";
        case common_agent_event_type::resource_chunk_planned: return "resource_chunk_planned";
        case common_agent_event_type::resource_chunk_processed: return "resource_chunk_processed";
        case common_agent_event_type::resource_created: return "resource_created";
        case common_agent_event_type::resource_attached: return "resource_attached";
        case common_agent_event_type::reflection_completed: return "reflection_completed";
        case common_agent_event_type::response_revised: return "response_revised";
        case common_agent_event_type::step_reviewed: return "step_reviewed";
        case common_agent_event_type::answer_reviewed: return "answer_reviewed";
        case common_agent_event_type::plan_revision_requested: return "plan_revision_requested";
        case common_agent_event_type::plan_revision_limit_reached: return "plan_revision_limit_reached";
        case common_agent_event_type::research_started: return "research_started";
        case common_agent_event_type::research_reopened: return "research_reopened";
        case common_agent_event_type::thinking_mode_resolved: return "thinking_mode_resolved";
        case common_agent_event_type::thinking_escalation_allowed: return "thinking_escalation_allowed";
        case common_agent_event_type::thinking_escalation_denied: return "thinking_escalation_denied";
        case common_agent_event_type::thinking_escalation_requested: return "thinking_escalation_requested";
        case common_agent_event_type::blueprint_selection_evaluated: return "blueprint_selection_evaluated";
        case common_agent_event_type::research_gap_opened: return "research_gap_opened";
        case common_agent_event_type::research_task_scheduled: return "research_task_scheduled";
        case common_agent_event_type::research_task_started: return "research_task_started";
        case common_agent_event_type::research_task_completed: return "research_task_completed";
        case common_agent_event_type::research_task_failed: return "research_task_failed";
        case common_agent_event_type::research_iteration_completed: return "research_iteration_completed";
        case common_agent_event_type::research_sources_compared: return "research_sources_compared";
        case common_agent_event_type::research_source_recorded: return "research_source_recorded";
        case common_agent_event_type::research_evidence_recorded: return "research_evidence_recorded";
        case common_agent_event_type::research_completed: return "research_completed";
        case common_agent_event_type::research_incomplete: return "research_incomplete";
        case common_agent_event_type::resource_processing_started: return "resource_processing_started";
        case common_agent_event_type::resource_processing_completed: return "resource_processing_completed";
        case common_agent_event_type::resource_processing_failed: return "resource_processing_failed";
    }
    return "unknown";
}

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
    // Bounded, host-generated repair input. It contains a schema-derived
    // argument skeleton and the effective tool names, never hidden policy.
    std::string repair_context_json;
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
    std::string step_id;
    std::string observation_id;
    std::string tool_name;
    std::string resource_uri;
};

using common_agent_event_sink = std::function<void(const common_agent_event &)>;

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

struct common_agent_input_resource {
    common_runtime_resource_ref resource;
    std::string role = "reference";
    bool required = false;
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
    common_agent_deliberation_policy deliberation_policy;

    // Runtime inputs are data only; tool execution remains policy-owned by
    // the caller-supplied registry.
    std::string prompt;
    std::vector<common_agent_input_resource> input_resources;
    // Host-owned compact state used only when an internal continuation has
    // deliberately replaced verbose working context with a bounded view.
    std::optional<common_agent_working_state> working_state;
    std::optional<common_agent_objective> objective;
    std::optional<common_memory_policy_pack> policy_pack;
    std::vector<common_memory_hit> memories;
    std::optional<common_agent_user_correction> user_correction;
    // A host may deliberately submit one bounded memory candidate through
    // the same native memory policy used by post-turn learning.
    std::optional<common_memory_candidate> explicit_memory_candidate;
    // Explicit capture is a two-turn-safe workflow: storage requires a
    // host/user confirmation on the request that persists the candidate.
    bool explicit_memory_confirmed = false;
    std::optional<common_agent_tool_call> tool_call;
    // Optional live delivery for host-owned event streams. The runtime still
    // records every event in common_agent_result::events for compatibility.
    common_agent_event_sink event_sink;
    size_t max_iterations = 2;
    size_t max_reflection_rounds = 1;
    size_t max_tool_batches = 1;
    bool allow_policy_gated_tool_proposals = false;

    // Host-owned cancellation hooks used by bounded research acquisition.
    // They remain optional so direct common-runtime callers are source-compatible.
    std::function<bool()> research_should_stop;
    std::function<common_agent_research_stop_reason()> research_stop_reason;
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
    // Present when the controller has produced a bounded continuation state;
    // it is not a durable-memory write and does not make the turn complete.
    std::optional<common_agent_continuation_checkpoint> continuation_checkpoint;
    std::vector<common_agent_generation_record> generation_records;

    std::optional<common_agent_research_result> research_result;
    // Host-owned execution state for resuming an incomplete research phase;
    // this is not durable memory and remains scoped to the active operation.
    std::optional<common_agent_research_workspace_checkpoint> research_workspace_checkpoint;
    std::optional<common_agent_research_verification> research_verification;

    // Events contain structured outcomes only, never raw chain-of-thought.
    std::vector<common_agent_event> events;
    std::vector<common_runtime_trace_entry> trace;
};
