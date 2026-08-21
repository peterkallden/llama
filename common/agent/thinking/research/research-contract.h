#pragma once

#include "agent/agent-scope.h"
#include "resource/resource-contract.h"
#include "runtime/runtime-state.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

struct common_agent_research_objective {
    std::string objective_id;
    std::string question;
    std::string purpose;
    std::string expected_output;
    std::vector<std::string> success_criteria;
    std::vector<std::string> constraints;
    std::vector<std::string> excluded_topics;
    common_agent_scope scope;
};

enum class common_agent_research_gap_status { open, investigating, sufficiently_answered, contradicted, blocked, abandoned };
struct common_agent_research_gap {
    std::string gap_id, question, reason, completion_criterion;
    int priority = 0;
    common_agent_research_gap_status status = common_agent_research_gap_status::open;
    std::vector<std::string> evidence_ids, child_gap_ids;
};

enum class common_agent_research_task_kind { search, fetch, repository_inspection, resource_read, evidence_extract, compare_sources, verify_claim, summarize_material };
enum class common_agent_research_task_status { pending, active, completed, blocked, failed, cancelled };
struct common_agent_research_task {
    std::string task_id, gap_id, instruction;
    common_agent_research_task_kind kind = common_agent_research_task_kind::search;
    std::vector<std::string> preferred_tools, dependency_ids;
    int priority = 0, attempt = 0, max_attempts = 1;
    common_agent_research_task_status status = common_agent_research_task_status::pending;
};

enum class common_agent_research_source_kind { repository_file, documentation, web_page, api_result, issue, pull_request, memory, user_supplied, remote_agent_result };
struct common_agent_research_source {
    std::string source_id, title, origin, authority, retrieved_at, content_hash;
    common_agent_research_source_kind kind = common_agent_research_source_kind::documentation;
    std::optional<common_runtime_resource_ref> resource_ref;
    double quality_score = 0.0;
    bool primary_source = false;
    std::string memory_id;
    std::string role = "reference";
    bool required = false;
};

enum class common_agent_research_evidence_relation { supports, contradicts, qualifies, contextualizes, inconclusive };
enum class common_agent_research_evidence_origin { direct_source, normalized_tool_result, model_inference };
struct common_agent_research_evidence {
    std::string evidence_id, source_id, claim_id, statement, source_location;
    common_agent_research_evidence_relation relation = common_agent_research_evidence_relation::inconclusive;
    common_agent_research_evidence_origin origin = common_agent_research_evidence_origin::normalized_tool_result;
    double relevance = 0.0, confidence = 0.0;
    bool directly_observed = false, model_inferred = false;
};

enum class common_agent_research_comparison_relation { agree, conflict, qualify };
struct common_agent_research_source_comparison {
    std::string comparison_id;
    std::string gap_id;
    std::vector<std::string> source_ids;
    std::vector<std::string> evidence_ids;
    common_agent_research_comparison_relation relation = common_agent_research_comparison_relation::agree;
    std::string summary;
};

struct common_agent_research_budget {
    int max_iterations = 4, max_tasks = 16, max_tool_calls = 16, max_sources = 16, max_sources_per_gap = 4;
    int minimum_sources = 1;
    int64_t max_tokens = 0, max_elapsed_ms = 0;
    double minimum_coverage = 1.0, minimum_confidence = 0.0;
};
struct common_agent_research_coverage {
    int total_gaps = 0, answered_gaps = 0, blocked_gaps = 0, unresolved_critical_gaps = 0;
    double objective_coverage = 0.0, evidence_quality = 0.0, source_diversity = 0.0;
};

struct common_agent_research_workspace {
    std::string workspace_id, request_id, turn_id, session_id, plan_id, plan_step_id;
    common_agent_scope scope;
    common_agent_research_objective objective;
    common_agent_research_budget budget;
    std::vector<common_agent_research_gap> gaps;
    std::vector<common_agent_research_task> tasks;
    std::vector<common_agent_research_source> sources;
    std::vector<common_agent_research_evidence> evidence;
    std::vector<common_agent_research_source_comparison> comparisons;
    common_agent_research_coverage coverage;
    int iterations_completed = 0;
    int tool_calls = 0;
    int no_progress_iterations = 0;
};

// A bounded snapshot of the active research workspace. This is execution
// state for the current session/turn and is not a second research store or a
// long-term memory record. The workspace's existing budgets remain the source
// of its collection bounds.
struct common_agent_research_workspace_checkpoint {
    std::string checkpoint_id;
    std::string workspace_id;
    std::string request_id;
    std::string turn_id;
    std::string session_id;
    std::string plan_id;
    size_t sequence = 0;
    common_agent_research_workspace workspace;
};

inline bool common_agent_research_workspace_validate(
        const common_agent_research_workspace & workspace,
        std::string & error);

inline common_agent_state_descriptor describe_common_agent_research_workspace(
        const common_agent_research_workspace & workspace) {
    common_agent_state_descriptor descriptor;
    descriptor.state_id = workspace.workspace_id;
    descriptor.state_type = "research_workspace";
    descriptor.state_class = common_agent_state_class::turn_workspace;
    descriptor.lifetime = common_agent_state_lifetime::turn;
    descriptor.persistence = common_agent_state_persistence::none;
    descriptor.identity.namespace_id = workspace.scope.namespace_id;
    descriptor.identity.project_id = workspace.scope.project_id;
    descriptor.identity.session_id = workspace.scope.session_id;
    descriptor.identity.turn_id = workspace.scope.turn_id;
    descriptor.owner = "agent runtime research loop";
    descriptor.source_of_truth = "research workspace";
    return descriptor;
}

inline bool common_agent_research_workspace_checkpoint_valid(
        const common_agent_research_workspace_checkpoint & checkpoint,
        std::string & error) {
    if (checkpoint.checkpoint_id.empty() || checkpoint.workspace_id.empty() ||
            checkpoint.request_id.empty() || checkpoint.turn_id.empty() ||
            checkpoint.session_id.empty() || checkpoint.sequence == 0) {
        error = "research workspace checkpoint requires bounded identity and sequence";
        return false;
    }
    if (checkpoint.workspace.workspace_id != checkpoint.workspace_id ||
            checkpoint.workspace.request_id != checkpoint.request_id ||
            checkpoint.workspace.turn_id != checkpoint.turn_id ||
            checkpoint.workspace.session_id != checkpoint.session_id ||
            checkpoint.workspace.plan_id != checkpoint.plan_id) {
        error = "research workspace checkpoint identity does not match its workspace";
        return false;
    }
    if (!common_agent_research_workspace_validate(checkpoint.workspace, error)) return false;
    error.clear();
    return true;
}

inline common_agent_research_workspace_checkpoint make_common_agent_research_workspace_checkpoint(
        const common_agent_research_workspace & workspace,
        size_t sequence,
        std::string & error) {
    common_agent_research_workspace_checkpoint checkpoint;
    checkpoint.checkpoint_id = "research-checkpoint:" + workspace.workspace_id + ":" +
        std::to_string(sequence);
    checkpoint.workspace_id = workspace.workspace_id;
    checkpoint.request_id = workspace.request_id;
    checkpoint.turn_id = workspace.turn_id;
    checkpoint.session_id = workspace.session_id;
    checkpoint.plan_id = workspace.plan_id;
    checkpoint.sequence = sequence;
    checkpoint.workspace = workspace;
    if (!common_agent_research_workspace_checkpoint_valid(checkpoint, error)) return {};
    error.clear();
    return checkpoint;
}

enum class common_agent_research_stop_reason { success_criteria_met, sufficient_coverage, budget_exhausted, deadline_exceeded, cancelled, no_progress, source_access_blocked, policy_blocked };
struct common_agent_research_result {
    std::string workspace_id;
    common_agent_research_stop_reason stop_reason = common_agent_research_stop_reason::budget_exhausted;
    std::vector<std::string> established_claim_ids, unresolved_claim_ids, critical_evidence_ids;
    std::vector<common_agent_research_source> sources;
    std::vector<common_agent_research_evidence> evidence;
    std::vector<common_agent_research_source_comparison> comparisons;
    std::string synthesis_context;
    common_agent_research_coverage coverage;
    bool complete = false;
};

enum class common_agent_research_lifecycle_event_type {
    gap_opened,
    task_scheduled,
    task_started,
    task_completed,
    task_failed,
    iteration_completed,
    sources_compared,
};

struct common_agent_research_lifecycle_event {
    common_agent_research_lifecycle_event_type type =
        common_agent_research_lifecycle_event_type::gap_opened;
    std::string gap_id;
    std::string task_id;
    int iteration = 0;
    bool retry = false;
};

using common_agent_research_lifecycle_sink =
    std::function<void(const common_agent_research_lifecycle_event &)>;

inline bool common_agent_research_workspace_validate(const common_agent_research_workspace & workspace, std::string & error) {
    if (workspace.workspace_id.empty() || workspace.objective.objective_id.empty() || workspace.objective.question.empty()) {
        error = "research workspace requires workspace/objective/question identities";
        return false;
    }
    if (workspace.scope.namespace_id.empty() || workspace.scope.session_id.empty()) {
        error = "research workspace requires namespace and session scope";
        return false;
    }
    const auto & budget = workspace.budget;
    if (budget.max_iterations < 0 || budget.max_tasks < 0 || budget.max_tool_calls < 0 || budget.max_sources < 0 ||
            budget.max_sources_per_gap < 0 || budget.max_tokens < 0 || budget.max_elapsed_ms < 0 ||
            budget.minimum_sources < 1 ||
            budget.minimum_coverage < 0.0 || budget.minimum_coverage > 1.0 || budget.minimum_confidence < 0.0 ||
            budget.minimum_confidence > 1.0) {
        error = "research workspace budget is invalid";
        return false;
    }
    if (workspace.iterations_completed < 0 || workspace.tool_calls < 0 || workspace.no_progress_iterations < 0 ||
            workspace.iterations_completed > budget.max_iterations || workspace.tool_calls > budget.max_tool_calls) {
        error = "research workspace counters exceed their budget";
        return false;
    }
    std::unordered_set<std::string> ids;
    for (const auto & gap : workspace.gaps) if (gap.gap_id.empty() || gap.question.empty() || gap.completion_criterion.empty() || !ids.insert(gap.gap_id).second) {
        error = "research workspace contains an invalid or duplicate gap"; return false;
    }
    ids.clear();
    for (const auto & task : workspace.tasks) if (task.task_id.empty() || task.gap_id.empty() || task.attempt < 0 || task.max_attempts < 1 || task.attempt > task.max_attempts || !ids.insert(task.task_id).second) {
        error = "research workspace contains an invalid or duplicate task"; return false;
    }
    ids.clear();
    for (const auto & source : workspace.sources) if (source.source_id.empty() || source.quality_score < 0.0 || source.quality_score > 1.0 ||
            (source.kind == common_agent_research_source_kind::memory && source.memory_id.empty()) ||
            !ids.insert(source.source_id).second) {
        error = "research workspace contains an invalid or duplicate source"; return false;
    }
    ids.clear();
    for (const auto & item : workspace.evidence) if (item.evidence_id.empty() || item.source_id.empty() || item.claim_id.empty() || item.statement.empty() || item.relevance < 0.0 || item.relevance > 1.0 || item.confidence < 0.0 || item.confidence > 1.0 || !ids.insert(item.evidence_id).second) {
        error = "research workspace contains invalid or duplicate evidence"; return false;
    }
    ids.clear();
    for (const auto & comparison : workspace.comparisons) if (comparison.comparison_id.empty() ||
            comparison.gap_id.empty() || comparison.source_ids.size() < 2 ||
            comparison.evidence_ids.empty() || comparison.summary.empty() ||
            !ids.insert(comparison.comparison_id).second) {
        error = "research workspace contains invalid or duplicate source comparison"; return false;
    }
    error.clear();
    return true;
}
