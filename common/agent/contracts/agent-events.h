#pragma once

#include <functional>
#include <optional>
#include <string>

enum class common_agent_event_type {
    model_loading, runtime_ready, runtime_failed,
    memory_retrieved, memory_remembered, memory_rejected,
    memory_candidate_extracted, memory_candidate_not_stored,
    memory_capture_confirmation_required, memory_capture_confirmed,
    blueprint_promoted, tool_executed, tool_rejected,
    tool_repair_context_created, plan_created, plan_updated,
    observation_recorded, resource_chunk_planned, resource_chunk_processed,
    resource_created, resource_attached, reflection_completed, response_revised,
    step_reviewed, answer_reviewed, plan_revision_requested,
    plan_revision_limit_reached, research_started, research_reopened,
    thinking_mode_resolved, thinking_escalation_allowed,
    thinking_escalation_denied, thinking_escalation_requested,
    blueprint_selection_evaluated, research_gap_opened, research_task_scheduled,
    research_task_started, research_task_completed, research_task_failed,
    research_iteration_completed, research_sources_compared,
    research_source_recorded, research_evidence_recorded, research_completed,
    research_incomplete, resource_processing_started,
    resource_processing_completed, resource_processing_failed,
};

inline const char * common_agent_event_type_name(common_agent_event_type type) {
    switch (type) {
        case common_agent_event_type::model_loading: return "model_loading";
        case common_agent_event_type::runtime_ready: return "runtime_ready";
        case common_agent_event_type::runtime_failed: return "runtime_failed";
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
