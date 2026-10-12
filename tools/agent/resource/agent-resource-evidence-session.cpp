#include "agent-resource-evidence-session.h"

bool make_agent_resource_evidence_session(
        const agent_resource_store & store,
        const std::string & uri,
        const agent_resource_read_authority & authority,
        const common_runtime_resource_chunk_policy & policy,
        agent_resource_evidence_session & out,
        std::string & error) {
    out = {};
    if (!plan_agent_resource_text_chunks(store, uri, authority, policy, out.plan, error)) return false;
    error.clear();
    return true;
}

bool read_agent_resource_evidence_slice(
        const agent_resource_store & store,
        const agent_resource_read_authority & authority,
        const agent_resource_evidence_session & session,
        common_agent_generation_resource_chunk & slice,
        std::string & error) {
    if (session.active_index >= session.plan.ranges.size()) {
        error = "resource evidence session cursor is exhausted";
        return false;
    }
    const auto & range = session.plan.ranges[session.active_index];
    slice = {};
    slice.source = make_agent_resource_chunk_ref(session.plan, range);
    slice.chunk_index = range.chunk_index;
    slice.chunk_count = range.chunk_count;
    slice.byte_offset = range.byte_offset;
    slice.byte_length = range.byte_length;
    std::string raw_text;
    if (!read_agent_resource_chunk(store, authority, session.plan, session.active_index, raw_text, error)) return false;
    common_runtime_resource_text_view text_view;
    if (!common_runtime_resource_make_text_view(raw_text, text_view, error)) return false;
    slice.text = common_runtime_resource_render_text_segments(
        text_view, 0, text_view.source_offsets.back(), slice.segments);
    if (slice.segments.empty()) {
        error = "resource evidence slice contains no text segments";
        return false;
    }
    error.clear();
    return true;
}

agent_resource_evidence_transition advance_agent_resource_evidence_session(
        agent_resource_evidence_session & session,
        bool resolved) {
    if (session.active_index >= session.plan.ranges.size()) return agent_resource_evidence_transition::invalid;
    if (!resolved) ++session.unresolved_count;
    ++session.active_index;
    return session.active_index >= session.plan.ranges.size()
        ? agent_resource_evidence_transition::complete
        : resolved ? agent_resource_evidence_transition::next
                   : agent_resource_evidence_transition::unresolved;
}
