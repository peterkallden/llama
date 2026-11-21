#pragma once

#include "agent-resource-chunker.h"
#include "agent/agent-inference.h"
#include "common/resource/resource-text-view.h"

// Administrative state for one host-owned evidence lane. Semantic evidence
// selection is model work; this type only owns coverage and provenance.
struct agent_resource_evidence_session {
    agent_resource_chunk_plan plan;
    size_t active_index = 0;
    size_t unresolved_count = 0;
};

enum class agent_resource_evidence_transition { next, complete, unresolved, invalid };

bool make_agent_resource_evidence_session(
        const agent_resource_store & store,
        const std::string & uri,
        const agent_resource_read_authority & authority,
        const common_runtime_resource_chunk_policy & policy,
        agent_resource_evidence_session & out,
        std::string & error);

bool read_agent_resource_evidence_slice(
        const agent_resource_store & store,
        const agent_resource_read_authority & authority,
        const agent_resource_evidence_session & session,
        common_agent_generation_resource_chunk & slice,
        std::string & error);

agent_resource_evidence_transition advance_agent_resource_evidence_session(
        agent_resource_evidence_session & session,
        bool resolved);
