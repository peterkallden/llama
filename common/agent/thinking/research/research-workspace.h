#pragma once

#include "agent/thinking/research/research-contract.h"
#include <algorithm>

inline bool common_agent_research_is_candidate(const common_agent_research_source & source) {
    return source.role == "discovery_candidate" || source.role == "api_discovery_candidate";
}
inline size_t common_agent_research_candidate_count(const common_agent_research_workspace & workspace) {
    return std::count_if(workspace.sources.begin(), workspace.sources.end(), common_agent_research_is_candidate);
}
inline size_t common_agent_research_source_count(const common_agent_research_workspace & workspace) {
    return workspace.sources.size() - common_agent_research_candidate_count(workspace);
}

bool common_agent_research_add_gap(
        common_agent_research_workspace & workspace,
        common_agent_research_gap gap,
        std::string & error);

bool common_agent_research_add_task(
        common_agent_research_workspace & workspace,
        common_agent_research_task task,
        std::string & error);

bool common_agent_research_add_source(
        common_agent_research_workspace & workspace,
        common_agent_research_source source,
        std::string & error);

bool common_agent_research_materialize_candidate(common_agent_research_workspace & workspace,
        common_agent_research_source & candidate, const common_runtime_resource_ref & resource, std::string & error);

// Update a retained source and its source-local evidence locations together.
inline void common_agent_research_rebind_resource(common_agent_research_workspace & workspace,
        common_agent_research_source & source, const common_runtime_resource_ref & retained) {
    if (!source.resource_ref) return;
    const auto old_uri = source.resource_ref->uri;
    if (!old_uri.empty()) {
        for (auto & progress : workspace.resource_progress) {
            if (progress.source_id == source.source_id && progress.position.resource_uri == old_uri)
                progress.position.resource_uri = retained.uri;
        }
        for (auto & evidence : workspace.evidence) {
            if (evidence.source_id == source.source_id &&
                    evidence.source_location.compare(0, old_uri.size(), old_uri) == 0 &&
                    (evidence.source_location.size() == old_uri.size() || evidence.source_location[old_uri.size()] == '#'))
                evidence.source_location.replace(0, old_uri.size(), retained.uri);
        }
    }
    *source.resource_ref = retained;
}

bool common_agent_research_record_evidence(
        common_agent_research_workspace & workspace,
        common_agent_research_evidence evidence,
        std::string & error);

// Convert host-validated chunk observations into the existing Research
// evidence/source contracts. This is a transient projection; observations
// remain the plan's progress and resume record.
bool common_agent_research_chunk_evidence_from_observations(
        const std::vector<common_plan_observation> & observations,
        const std::string & claim_id,
        std::vector<common_agent_research_source> & sources,
        std::vector<common_agent_research_evidence> & evidence,
        std::string & error);

// Carry host-validated resource-chunk findings into research as preliminary
// evidence. This does not mark any research gap as answered.
bool common_agent_research_seed_chunk_observations(
        common_agent_research_workspace & workspace,
        const std::vector<common_plan_observation> & observations,
        std::string & error);

bool common_agent_research_add_comparison(
        common_agent_research_workspace & workspace,
        common_agent_research_source_comparison comparison,
        std::string & error);

bool common_agent_research_update_coverage(
        common_agent_research_workspace & workspace,
        common_agent_research_coverage coverage,
        std::string & error);

bool common_agent_research_transition_gap(
        common_agent_research_workspace & workspace,
        const std::string & gap_id,
        common_agent_research_gap_status target,
        std::string & error);

bool common_agent_research_transition_task(
        common_agent_research_workspace & workspace,
        const std::string & task_id,
        common_agent_research_task_status target,
        std::string & error);
