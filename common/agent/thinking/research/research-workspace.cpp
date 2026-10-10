#include "agent/thinking/research/research-workspace.h"

#include <algorithm>
#include <sstream>

namespace {

template <typename T, typename Predicate>
bool has_item(const std::vector<T> & items, Predicate predicate) {
    return std::find_if(items.begin(), items.end(), predicate) != items.end();
}

bool valid_unit_value(double value) {
    return value >= 0.0 && value <= 1.0;
}

} // namespace

bool common_agent_research_add_gap(
        common_agent_research_workspace & workspace,
        common_agent_research_gap gap,
        std::string & error) {
    if (gap.gap_id.empty() || gap.question.empty()) {
        error = "research gap requires id and question";
        return false;
    }
    if (has_item(workspace.gaps, [&](const auto & item) { return item.gap_id == gap.gap_id; })) {
        error = "research gap id already exists";
        return false;
    }
    workspace.gaps.push_back(std::move(gap));
    error.clear();
    return true;
}

bool common_agent_research_add_task(
        common_agent_research_workspace & workspace,
        common_agent_research_task task,
        std::string & error) {
    if (task.task_id.empty() || task.gap_id.empty() || task.instruction.empty()) {
        error = "research task requires id, gap and instruction";
        return false;
    }
    if (!has_item(workspace.gaps, [&](const auto & item) { return item.gap_id == task.gap_id; })) {
        error = "research task references an unknown gap";
        return false;
    }
    if (workspace.budget.max_tasks >= 0 &&
            workspace.tasks.size() >= static_cast<size_t>(workspace.budget.max_tasks)) {
        error = "research task budget exhausted";
        return false;
    }
    if (task.attempt < 0 || task.max_attempts < 1 || task.attempt > task.max_attempts) {
        error = "research task attempt budget is invalid";
        return false;
    }
    if (has_item(workspace.tasks, [&](const auto & item) { return item.task_id == task.task_id; })) {
        error = "research task id already exists";
        return false;
    }
    workspace.tasks.push_back(std::move(task));
    error.clear();
    return true;
}

bool common_agent_research_add_source(
        common_agent_research_workspace & workspace,
        common_agent_research_source source,
        std::string & error) {
    if (source.source_id.empty() || source.title.empty() || !valid_unit_value(source.quality_score)) {
        error = "research source requires id, title and bounded quality";
        return false;
    }
    if (workspace.budget.max_sources >= 0 &&
            workspace.sources.size() >= static_cast<size_t>(workspace.budget.max_sources)) {
        error = "research source budget exhausted";
        return false;
    }
    if (has_item(workspace.sources, [&](const auto & item) { return item.source_id == source.source_id; })) {
        error = "research source id already exists";
        return false;
    }
    workspace.sources.push_back(std::move(source));
    error.clear();
    return true;
}

bool common_agent_research_record_evidence(
        common_agent_research_workspace & workspace,
        common_agent_research_evidence evidence,
        std::string & error) {
    if (evidence.evidence_id.empty() || evidence.source_id.empty() || evidence.claim_id.empty() || evidence.statement.empty() ||
            !valid_unit_value(evidence.relevance) || !valid_unit_value(evidence.confidence)) {
        error = "research evidence requires id, source, statement and bounded scores";
        return false;
    }
    if (!has_item(workspace.sources, [&](const auto & item) { return item.source_id == evidence.source_id; })) {
        error = "research evidence references an unknown source";
        return false;
    }
    if (has_item(workspace.evidence, [&](const auto & item) { return item.evidence_id == evidence.evidence_id; })) {
        error = "research evidence id already exists";
        return false;
    }
    workspace.evidence.push_back(std::move(evidence));
    error.clear();
    return true;
}

bool common_agent_research_seed_chunk_observations(
        common_agent_research_workspace & workspace,
        const std::vector<common_plan_observation> & observations,
        std::string & error) {
    auto gap = workspace.gaps.end();
    for (auto it = workspace.gaps.begin(); it != workspace.gaps.end(); ++it) {
        if (it->status != common_agent_research_gap_status::open) continue;
        if (gap != workspace.gaps.end()) {
            error.clear();
            return true; // Multiple candidate gaps require an explicit assessment.
        }
        gap = it;
    }
    if (gap == workspace.gaps.end()) {
        error.clear();
        return true;
    }

    for (const auto & observation : observations) {
        if (observation.source != "resource_chunk" || observation.resource_refs.size() != 1) continue;
        const auto & resource = observation.resource_refs.front();
        if (resource.uri.empty() || !common_runtime_resource_lineage_is_chunk(resource.lineage)) continue;

        std::string pending_finding;
        size_t finding_index = 0;
        std::istringstream lines(observation.summary);
        for (std::string line; std::getline(lines, line);) {
            constexpr const char * finding_prefix = "finding: ";
            constexpr const char * evidence_prefix = "evidence: ";
            if (line.rfind(finding_prefix, 0) == 0) {
                pending_finding = line.substr(std::char_traits<char>::length(finding_prefix));
                continue;
            }
            if (line.rfind(evidence_prefix, 0) != 0 || pending_finding.empty()) continue;
            const std::string quote = line.substr(std::char_traits<char>::length(evidence_prefix));
            if (quote.empty()) {
                pending_finding.clear();
                continue;
            }

            auto source = std::find_if(workspace.sources.begin(), workspace.sources.end(),
                [&](const auto & candidate) {
                    return candidate.resource_ref && candidate.resource_ref->uri == resource.uri;
                });
            if (source == workspace.sources.end()) {
                common_agent_research_source chunk_source;
                chunk_source.source_id = workspace.workspace_id + ":workflow:chunks:" +
                    std::to_string(resource.lineage.chunk_index);
                chunk_source.title = resource.name.empty() ? "Validated resource text chunk" : resource.name;
                chunk_source.origin = "host.chunk.text.v1";
                chunk_source.authority = "host-resource-runtime";
                chunk_source.kind = common_agent_research_source_kind::documentation;
                chunk_source.availability = common_agent_research_source_availability::materialized_context;
                chunk_source.resource_ref = resource;
                chunk_source.quality_score = 0.85;
                chunk_source.primary_source = true;
                chunk_source.role = "workflow_evidence";
                if (!common_agent_research_add_source(workspace, std::move(chunk_source), error)) return false;
                source = std::prev(workspace.sources.end());
            }

            const std::string evidence_id = workspace.workspace_id + ":chunk:" +
                std::to_string(resource.lineage.chunk_index) + ":finding:" +
                std::to_string(++finding_index);
            if (std::find_if(workspace.evidence.begin(), workspace.evidence.end(),
                    [&](const auto & candidate) { return candidate.evidence_id == evidence_id; }) ==
                    workspace.evidence.end()) {
                common_agent_research_evidence evidence;
                evidence.evidence_id = evidence_id;
                evidence.source_id = source->source_id;
                evidence.claim_id = gap->gap_id;
                evidence.statement = pending_finding + " (quoted source evidence: " + quote + ")";
                evidence.source_location = resource.uri + "#bytes=" +
                    std::to_string(resource.lineage.byte_offset) + "+" +
                    std::to_string(resource.lineage.byte_length);
                evidence.relation = common_agent_research_evidence_relation::supports;
                evidence.origin = common_agent_research_evidence_origin::model_inference;
                evidence.relevance = 0.85;
                evidence.confidence = 0.75;
                evidence.model_inferred = true;
                if (!common_agent_research_record_evidence(workspace, evidence, error)) return false;
                gap->evidence_ids.push_back(evidence_id);
            }
            pending_finding.clear();
        }
    }
    error.clear();
    return true;
}

bool common_agent_research_add_comparison(
        common_agent_research_workspace & workspace,
        common_agent_research_source_comparison comparison,
        std::string & error) {
    if (comparison.comparison_id.empty() || comparison.gap_id.empty() ||
            comparison.source_ids.size() < 2 || comparison.evidence_ids.empty() ||
            comparison.summary.empty()) {
        error = "research source comparison requires gap, sources, evidence and summary";
        return false;
    }
    if (!has_item(workspace.gaps, [&](const auto & gap) { return gap.gap_id == comparison.gap_id; })) {
        error = "research source comparison references an unknown gap";
        return false;
    }
    for (const auto & source_id : comparison.source_ids) {
        if (!has_item(workspace.sources, [&](const auto & source) { return source.source_id == source_id; })) {
            error = "research source comparison references an unknown source";
            return false;
        }
    }
    for (const auto & evidence_id : comparison.evidence_ids) {
        if (!has_item(workspace.evidence, [&](const auto & evidence) { return evidence.evidence_id == evidence_id; })) {
            error = "research source comparison references unknown evidence";
            return false;
        }
    }
    if (has_item(workspace.comparisons, [&](const auto & item) { return item.comparison_id == comparison.comparison_id; })) {
        error = "research source comparison id already exists";
        return false;
    }
    workspace.comparisons.push_back(std::move(comparison));
    error.clear();
    return true;
}

bool common_agent_research_update_coverage(
        common_agent_research_workspace & workspace,
        common_agent_research_coverage coverage,
        std::string & error) {
    if (coverage.total_gaps < 0 || coverage.answered_gaps < 0 || coverage.blocked_gaps < 0 ||
            coverage.unresolved_critical_gaps < 0 || coverage.answered_gaps > coverage.total_gaps ||
            coverage.blocked_gaps > coverage.total_gaps || !valid_unit_value(coverage.objective_coverage) ||
            !valid_unit_value(coverage.evidence_quality) || !valid_unit_value(coverage.source_diversity)) {
        error = "research coverage is invalid";
        return false;
    }
    workspace.coverage = coverage;
    error.clear();
    return true;
}

namespace {

bool valid_gap_transition(
        common_agent_research_gap_status from,
        common_agent_research_gap_status to) {
    if (from == to) return true;
    switch (from) {
        case common_agent_research_gap_status::open:
            return to == common_agent_research_gap_status::investigating ||
                to == common_agent_research_gap_status::abandoned;
        case common_agent_research_gap_status::investigating:
            return to == common_agent_research_gap_status::sufficiently_answered ||
                to == common_agent_research_gap_status::contradicted ||
                to == common_agent_research_gap_status::blocked ||
                to == common_agent_research_gap_status::abandoned ||
                to == common_agent_research_gap_status::open;
        case common_agent_research_gap_status::sufficiently_answered:
            return to == common_agent_research_gap_status::open;
        case common_agent_research_gap_status::contradicted:
            return to == common_agent_research_gap_status::investigating ||
                to == common_agent_research_gap_status::open;
        case common_agent_research_gap_status::blocked:
            return to == common_agent_research_gap_status::investigating ||
                to == common_agent_research_gap_status::open;
        case common_agent_research_gap_status::abandoned:
            return false;
    }
    return false;
}

bool valid_task_transition(
        common_agent_research_task_status from,
        common_agent_research_task_status to) {
    if (from == to) return true;
    switch (from) {
        case common_agent_research_task_status::pending:
            return to == common_agent_research_task_status::active ||
                to == common_agent_research_task_status::cancelled;
        case common_agent_research_task_status::active:
            return to == common_agent_research_task_status::completed ||
                to == common_agent_research_task_status::blocked ||
                to == common_agent_research_task_status::failed ||
                to == common_agent_research_task_status::cancelled;
        case common_agent_research_task_status::failed:
        case common_agent_research_task_status::blocked:
            return to == common_agent_research_task_status::pending;
        case common_agent_research_task_status::completed:
        case common_agent_research_task_status::cancelled:
            return false;
    }
    return false;
}

} // namespace

bool common_agent_research_transition_gap(
        common_agent_research_workspace & workspace,
        const std::string & gap_id,
        common_agent_research_gap_status target,
        std::string & error) {
    for (auto & gap : workspace.gaps) {
        if (gap.gap_id != gap_id) continue;
        if (!valid_gap_transition(gap.status, target)) {
            error = "invalid research gap transition";
            return false;
        }
        gap.status = target;
        error.clear();
        return true;
    }
    error = "research gap not found";
    return false;
}

bool common_agent_research_transition_task(
        common_agent_research_workspace & workspace,
        const std::string & task_id,
        common_agent_research_task_status target,
        std::string & error) {
    for (auto & task : workspace.tasks) {
        if (task.task_id != task_id) continue;
        if (!valid_task_transition(task.status, target)) {
            error = "invalid research task transition";
            return false;
        }
        task.status = target;
        error.clear();
        return true;
    }
    error = "research task not found";
    return false;
}
