#include "agent/thinking/research/research-controller.h"

#include <algorithm>

namespace {

common_agent_research_action complete_action(common_agent_research_stop_reason reason) {
    common_agent_research_action action;
    action.kind = common_agent_research_action_kind::complete;
    action.stop_reason = reason;
    return action;
}

void refresh_coverage(common_agent_research_workspace & workspace) {
    auto & coverage = workspace.coverage;
    coverage.total_gaps = static_cast<int>(workspace.gaps.size());
    coverage.answered_gaps = 0;
    coverage.blocked_gaps = 0;
    for (const auto & gap : workspace.gaps) {
        if (gap.status == common_agent_research_gap_status::sufficiently_answered) ++coverage.answered_gaps;
        if (gap.status == common_agent_research_gap_status::blocked || gap.status == common_agent_research_gap_status::abandoned) ++coverage.blocked_gaps;
    }
    coverage.objective_coverage = coverage.total_gaps == 0
        ? 1.0
        : static_cast<double>(coverage.answered_gaps) / coverage.total_gaps;
    if (workspace.evidence.empty()) {
        coverage.evidence_quality = 0.0;
        coverage.source_diversity = 0.0;
        return;
    }
    double confidence_total = 0.0;
    std::vector<std::string> identities;
    for (const auto & evidence : workspace.evidence) {
        confidence_total += evidence.confidence;
        for (const auto & source : workspace.sources) {
            if (source.source_id == evidence.source_id) {
                const auto identity = source.content_hash.empty()
                    ? (source.resource_ref ? source.resource_ref->uri : source.origin + ":" + source.title)
                    : source.content_hash;
                if (std::find(identities.begin(), identities.end(), identity) == identities.end()) {
                    identities.push_back(identity);
                }
                break;
            }
        }
    }
    coverage.evidence_quality = confidence_total / workspace.evidence.size();
    coverage.source_diversity = static_cast<double>(identities.size()) /
        static_cast<double>(std::max<size_t>(1, workspace.sources.size()));
}

common_agent_research_gap * find_gap(common_agent_research_workspace & workspace, const std::string & id) {
    for (auto & gap : workspace.gaps) if (gap.gap_id == id) return &gap;
    return nullptr;
}

common_agent_research_task * find_task(common_agent_research_workspace & workspace, const std::string & id) {
    for (auto & task : workspace.tasks) if (task.task_id == id) return &task;
    return nullptr;
}

bool source_used_for_gap(
        const common_agent_research_workspace & workspace,
        const common_agent_research_gap & gap,
        const std::string & source_id) {
    for (const auto & evidence_id : gap.evidence_ids) {
        for (const auto & evidence : workspace.evidence) {
            if (evidence.evidence_id == evidence_id && evidence.source_id == source_id) return true;
        }
    }
    return false;
}

int gap_source_count(
        const common_agent_research_workspace & workspace,
        const common_agent_research_gap & gap,
        double minimum_confidence) {
    std::vector<std::string> source_ids;
    for (const auto & evidence_id : gap.evidence_ids) {
        for (const auto & evidence : workspace.evidence) {
            if (evidence.evidence_id == evidence_id && evidence.confidence >= minimum_confidence) {
                for (const auto & source : workspace.sources) {
                    if (source.source_id != evidence.source_id) continue;
                    const auto identity = source.content_hash.empty()
                        ? (source.resource_ref ? source.resource_ref->uri : source.origin + ":" + source.title)
                        : source.content_hash;
                    if (std::find(source_ids.begin(), source_ids.end(), identity) == source_ids.end()) {
                        source_ids.push_back(identity);
                    }
                    break;
                }
            }
        }
    }
    return static_cast<int>(source_ids.size());
}

bool has_gap_comparison(
        const common_agent_research_workspace & workspace,
        const std::string & gap_id) {
    return std::find_if(
        workspace.comparisons.begin(),
        workspace.comparisons.end(),
        [&](const auto & comparison) { return comparison.gap_id == gap_id; }) != workspace.comparisons.end();
}

bool compare_gap_sources(
        common_agent_research_workspace & workspace,
        const common_agent_research_gap & gap,
        std::string & error) {
    if (has_gap_comparison(workspace, gap.gap_id)) return true;
    std::vector<std::string> source_ids;
    std::vector<std::string> evidence_ids;
    common_agent_research_comparison_relation relation =
        common_agent_research_comparison_relation::agree;
    for (const auto & evidence_id : gap.evidence_ids) {
        for (const auto & evidence : workspace.evidence) {
            if (evidence.evidence_id != evidence_id) continue;
            evidence_ids.push_back(evidence.evidence_id);
            if (evidence.relation == common_agent_research_evidence_relation::contradicts) {
                relation = common_agent_research_comparison_relation::conflict;
            } else if (evidence.relation == common_agent_research_evidence_relation::qualifies &&
                    relation == common_agent_research_comparison_relation::agree) {
                relation = common_agent_research_comparison_relation::qualify;
            }
            for (const auto & source : workspace.sources) {
                if (source.source_id != evidence.source_id) continue;
                const auto identity = source.content_hash.empty()
                    ? (source.resource_ref ? source.resource_ref->uri : source.origin + ":" + source.title)
                    : source.content_hash;
                if (std::find(source_ids.begin(), source_ids.end(), identity) == source_ids.end()) {
                    source_ids.push_back(identity);
                }
                break;
            }
            break;
        }
    }
    if (source_ids.size() < 2) return true;
    // Store source IDs, not only identities, so the artifact remains traceable.
    std::vector<std::string> traceable_source_ids;
    for (const auto & evidence_id : evidence_ids) {
        for (const auto & evidence : workspace.evidence) {
            if (evidence.evidence_id == evidence_id &&
                    std::find(traceable_source_ids.begin(), traceable_source_ids.end(), evidence.source_id) == traceable_source_ids.end()) {
                traceable_source_ids.push_back(evidence.source_id);
            }
        }
    }
    return common_agent_research_add_comparison(workspace, {
        gap.gap_id + ":comparison:" + std::to_string(workspace.comparisons.size() + 1),
        gap.gap_id,
        std::move(traceable_source_ids),
        std::move(evidence_ids),
        relation,
        relation == common_agent_research_comparison_relation::conflict
            ? "sources contain conflicting evidence"
            : (relation == common_agent_research_comparison_relation::qualify
                ? "sources agree with qualifying evidence"
                : "sources provide agreeing evidence")}, error);
}

bool has_evidence(const common_agent_research_workspace & workspace, const std::string & id) {
    for (const auto & evidence : workspace.evidence) if (evidence.evidence_id == id) return true;
    return false;
}

std::string query_for_attempt(
        const common_agent_research_gap & gap,
        int attempt) {
    if (attempt <= 0) return gap.question;
    if (attempt == 1) return "Find an authoritative source for: " + gap.question;
    return "Verify or contradict this claim using an independent source: " + gap.question;
}

common_agent_research_action schedule_next_gap(
        common_agent_research_workspace & workspace,
        std::string & error) {
    common_agent_research_gap * selected = nullptr;
    for (auto & gap : workspace.gaps) {
        if (gap.status != common_agent_research_gap_status::open) continue;
        if (selected == nullptr || gap.priority > selected->priority) selected = &gap;
    }
    if (selected == nullptr) {
        refresh_coverage(workspace);
        return complete_action(
            workspace.coverage.objective_coverage >= workspace.budget.minimum_coverage
                ? common_agent_research_stop_reason::sufficient_coverage
                : common_agent_research_stop_reason::no_progress);
    }
    if (workspace.budget.max_sources_per_gap > 0 &&
            gap_source_count(
                workspace,
                *selected,
                workspace.budget.minimum_confidence) >= workspace.budget.max_sources_per_gap) {
        if (!common_agent_research_transition_gap(
                workspace, selected->gap_id,
                common_agent_research_gap_status::blocked, error)) {
            return complete_action(common_agent_research_stop_reason::policy_blocked);
        }
        return schedule_next_gap(workspace, error);
    }
    int next_attempt = 1;
    for (const auto & task : workspace.tasks) {
        if (task.gap_id == selected->gap_id) next_attempt = std::max(next_attempt, task.attempt + 2);
    }
    const std::string task_id = selected->gap_id + ":task:" + std::to_string(next_attempt);
    common_agent_research_task task;
    task.task_id = task_id;
    task.gap_id = selected->gap_id;
    task.kind = common_agent_research_task_kind::search;
    task.attempt = next_attempt - 1;
    task.instruction = query_for_attempt(*selected, task.attempt);
    task.preferred_tools = task.attempt <= 0
        ? std::vector<std::string>{"resource_read", "repository.search", "web_search"}
        : (task.attempt == 1
            ? std::vector<std::string>{"repository.search", "resource_read", "web_search"}
            : std::vector<std::string>{"web_search", "web_fetch", "repository.search"});
    task.priority = selected->priority;
    task.max_attempts = std::max(1, workspace.budget.max_iterations);
    for (const auto & source : workspace.sources) {
        if (source.kind == common_agent_research_source_kind::memory && !source.memory_id.empty() &&
                !source_used_for_gap(workspace, *selected, source.source_id)) {
            task.instruction = source.memory_id;
            task.preferred_tools = {"memory_get"};
            break;
        }
        if (source.kind == common_agent_research_source_kind::user_supplied && source.resource_ref &&
                !source_used_for_gap(workspace, *selected, source.source_id)) {
            task.instruction = source.resource_ref->uri;
            task.preferred_tools = {"resource_read"};
            break;
        }
    }
    if (!common_agent_research_add_task(workspace, task, error)) return complete_action(common_agent_research_stop_reason::budget_exhausted);
    if (!common_agent_research_transition_task(
            workspace, task.task_id, common_agent_research_task_status::active, error) ||
            !common_agent_research_transition_gap(
                workspace, selected->gap_id,
                common_agent_research_gap_status::investigating, error)) {
        return complete_action(common_agent_research_stop_reason::policy_blocked);
    }
    common_agent_research_action action;
    action.kind = common_agent_research_action_kind::schedule_task;
    action.task_id = task.task_id;
    action.gap_id = task.gap_id;
    action.instruction = task.instruction;
    action.preferred_tools = task.preferred_tools;
    action.dependency_ids = task.dependency_ids;
    error.clear();
    return action;
}

} // namespace

common_agent_research_action common_agent_research_controller::begin(
        common_agent_research_workspace & workspace,
        std::string & error) const {
    if (!common_agent_research_workspace_validate(workspace, error)) return complete_action(common_agent_research_stop_reason::policy_blocked);
    if (workspace.budget.max_iterations == 0 || workspace.budget.max_tasks == 0) return complete_action(common_agent_research_stop_reason::budget_exhausted);
    return schedule_next_gap(workspace, error);
}

common_agent_research_action common_agent_research_controller::advance(
        common_agent_research_workspace & workspace,
        const common_agent_research_event & event,
        std::string & error) const {
    if (event.type == common_agent_research_event_type::cancelled) return complete_action(common_agent_research_stop_reason::cancelled);
    auto * task = find_task(workspace, event.task_id);
    auto * gap = find_gap(workspace, event.gap_id);
    if (task == nullptr || gap == nullptr || task->gap_id != gap->gap_id) {
        error = "research event references an unknown task or gap";
        return complete_action(common_agent_research_stop_reason::policy_blocked);
    }
    if (task->status != common_agent_research_task_status::active) {
        error = "research event targets a task that is not active";
        return complete_action(common_agent_research_stop_reason::policy_blocked);
    }
    if (event.type == common_agent_research_event_type::task_completed && event.evidence_count > 0) {
        if (event.gap_confidence < 0.0 || event.gap_confidence > 1.0) {
            error = "research event gap confidence is invalid";
            return complete_action(common_agent_research_stop_reason::policy_blocked);
        }
        for (const auto & evidence_id : event.evidence_ids) {
            if (!has_evidence(workspace, evidence_id)) {
                error = "research event references unknown evidence";
                return complete_action(common_agent_research_stop_reason::policy_blocked);
            }
        }
        if (!common_agent_research_transition_task(
                workspace, task->task_id,
                common_agent_research_task_status::completed, error)) {
            return complete_action(common_agent_research_stop_reason::policy_blocked);
        }
        for (const auto & evidence_id : event.evidence_ids) {
            if (std::find(gap->evidence_ids.begin(), gap->evidence_ids.end(), evidence_id) == gap->evidence_ids.end()) {
                gap->evidence_ids.push_back(evidence_id);
            }
        }
        const bool assessed_sufficiently =
            event.assessment.status == common_agent_research_assessment_status::sufficient ||
            (event.assessment.status == common_agent_research_assessment_status::inconclusive &&
                event.gap_sufficiently_answered);
        if (assessed_sufficiently && gap_source_count(
                workspace, *gap, workspace.budget.minimum_confidence) >= 2) {
            if (!compare_gap_sources(workspace, *gap, error)) return complete_action(common_agent_research_stop_reason::policy_blocked);
        }
        const bool sufficient_evidence = assessed_sufficiently &&
            std::max(event.gap_confidence, event.assessment.confidence) >= workspace.budget.minimum_confidence &&
            gap_source_count(workspace, *gap, workspace.budget.minimum_confidence) >= workspace.budget.minimum_sources &&
            (workspace.budget.minimum_sources < 2 || has_gap_comparison(workspace, gap->gap_id));
        if (!common_agent_research_transition_gap(
                workspace, gap->gap_id,
                sufficient_evidence
                    ? common_agent_research_gap_status::sufficiently_answered
                    : common_agent_research_gap_status::open,
                error)) {
            return complete_action(common_agent_research_stop_reason::policy_blocked);
        }
        workspace.no_progress_iterations = gap->status == common_agent_research_gap_status::open
            ? workspace.no_progress_iterations + 1
            : 0;
    } else {
        const auto task_target = event.type == common_agent_research_event_type::task_failed
            ? common_agent_research_task_status::failed
            : common_agent_research_task_status::completed;
        if (!common_agent_research_transition_task(
                workspace, task->task_id, task_target, error)) {
            return complete_action(common_agent_research_stop_reason::policy_blocked);
        }
        if (event.type == common_agent_research_event_type::task_failed &&
                event.retryable && task->attempt + 1 < task->max_attempts) {
            if (!common_agent_research_transition_gap(
                    workspace, gap->gap_id,
                    common_agent_research_gap_status::open, error)) {
                return complete_action(common_agent_research_stop_reason::policy_blocked);
            }
            workspace.no_progress_iterations += 1;
        } else {
            if (!common_agent_research_transition_gap(
                    workspace, gap->gap_id,
                    common_agent_research_gap_status::blocked, error)) {
                return complete_action(common_agent_research_stop_reason::policy_blocked);
            }
        }
    }
    refresh_coverage(workspace);
    if (workspace.no_progress_iterations >= 2) {
        return complete_action(common_agent_research_stop_reason::no_progress);
    }
    if (workspace.coverage.objective_coverage >= workspace.budget.minimum_coverage) return complete_action(common_agent_research_stop_reason::sufficient_coverage);
    return schedule_next_gap(workspace, error);
}

common_agent_research_result common_agent_research_controller::finalize(
        const common_agent_research_workspace & workspace) const {
    common_agent_research_result result;
    result.workspace_id = workspace.workspace_id;
    result.coverage = workspace.coverage;
    result.sources = workspace.sources;
    result.evidence = workspace.evidence;
    result.comparisons = workspace.comparisons;
    result.complete = result.coverage.objective_coverage >= workspace.budget.minimum_coverage;
    result.stop_reason = result.complete
        ? common_agent_research_stop_reason::sufficient_coverage
        : common_agent_research_stop_reason::no_progress;
    for (const auto & gap : workspace.gaps) {
        if (gap.status == common_agent_research_gap_status::sufficiently_answered) {
            result.established_claim_ids.push_back(gap.gap_id);
        }
        else result.unresolved_claim_ids.push_back(gap.gap_id);
    }
    result.synthesis_context = "Research coverage: " +
        std::to_string(result.coverage.answered_gaps) + "/" +
        std::to_string(result.coverage.total_gaps) + " gaps answered.\n";
    if (!result.unresolved_claim_ids.empty()) {
        result.synthesis_context += "Unresolved gaps: ";
        for (const auto & gap_id : result.unresolved_claim_ids) result.synthesis_context += gap_id + " ";
        result.synthesis_context += "\n";
    }
    result.synthesis_context += "Evidence with provenance:\n";
    for (const auto & evidence : workspace.evidence) {
        result.critical_evidence_ids.push_back(evidence.evidence_id);
        result.synthesis_context += "- [" + evidence.evidence_id + "] source=" +
            evidence.source_id + " claim=" + evidence.claim_id + ": " + evidence.statement + "\n";
        if (result.synthesis_context.size() >= 8192) break;
    }
    if (result.synthesis_context.size() < 8192 && !workspace.comparisons.empty()) {
        result.synthesis_context += "Source comparisons:\n";
        for (const auto & comparison : workspace.comparisons) {
            result.synthesis_context += "- [" + comparison.comparison_id + "] " + comparison.summary + "\n";
            if (result.synthesis_context.size() >= 8192) break;
        }
    }
    if (result.synthesis_context.size() > 8192) result.synthesis_context.resize(8192);
    return result;
}
