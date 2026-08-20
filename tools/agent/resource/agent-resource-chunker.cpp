#include "agent-resource-chunker.h"

#include <algorithm>

namespace {

bool find_range(
        const agent_resource_chunk_plan & plan,
        size_t chunk_index,
        agent_resource_chunk_range & out) {
    if (chunk_index >= plan.ranges.size()) {
        return false;
    }
    out = plan.ranges[chunk_index];
    return true;
}

} // namespace

bool plan_agent_resource_text_chunks(
        const agent_resource_store & store,
        const std::string & uri,
        const agent_resource_read_authority & authority,
        const common_runtime_resource_chunk_policy & policy,
        agent_resource_chunk_plan & out,
        std::string & error) {
    out = {};
    out.policy = policy;
    if (policy.max_bytes == 0) {
        error = "resource chunk max_bytes must be non-zero";
        return false;
    }
    if (policy.overlap_bytes >= policy.max_bytes) {
        error = "resource chunk overlap_bytes must be smaller than max_bytes";
        return false;
    }
    if (!store.stat(uri, authority, out.parent, error)) {
        return false;
    }
    if (!common_resource_media_type_is_text_like(out.parent.mime_type)) {
        error = "resource chunking requires a text-oriented resource";
        return false;
    }
    size_t offset = 0;
    while (offset < out.parent.size_bytes) {
        const size_t overlap = out.ranges.empty()
            ? 0 : policy.overlap_bytes;
        const size_t read_limit = policy.max_bytes - overlap;
        std::string window;
        if (!store.read_text_range(uri, authority, offset, read_limit, window, error)) {
            out = {};
            return false;
        }
        if (window.empty()) {
            error = "resource chunk range read returned no progress";
            out = {};
            return false;
        }

        std::vector<common_runtime_resource_chunk> candidates;
        if (!common_runtime_resource_chunk_text(
                    window,
                    {read_limit, 0},
                    candidates,
                    error) || candidates.empty()) {
            out = {};
            return false;
        }
        const auto & candidate = candidates.front();
        if (candidate.byte_length == 0) {
            error = "resource chunk boundary returned no progress";
            out = {};
            return false;
        }
        out.ranges.push_back({
            out.ranges.size(),
            0,
            offset - overlap,
            candidate.byte_length + overlap,
            overlap,
            candidate.boundary,
        });
        offset += candidate.byte_length;
    }

    for (auto & range : out.ranges) {
        range.chunk_count = out.ranges.size();
    }
    error.clear();
    return true;
}

bool read_agent_resource_chunk(
        const agent_resource_store & store,
        const agent_resource_read_authority & authority,
        const agent_resource_chunk_plan & plan,
        size_t chunk_index,
        std::string & out,
        std::string & error) {
    agent_resource_chunk_range range;
    if (!find_range(plan, chunk_index, range)) {
        error = "resource chunk index is out of range";
        return false;
    }
    return store.read_text_range(
        plan.parent.uri,
        authority,
        range.byte_offset,
        range.byte_length,
        out,
        error);
}

common_runtime_resource_ref make_agent_resource_chunk_ref(
        const agent_resource_chunk_plan & plan,
        const agent_resource_chunk_range & range) {
    common_runtime_resource_ref resource = plan.parent;
    resource.size_bytes = range.byte_length;
    resource.lineage = {
        plan.parent.uri,
        range.chunk_index,
        range.chunk_count,
        range.byte_offset,
        range.byte_length,
        range.overlap_bytes,
        "host.chunk.text.v1",
    };
    return resource;
}
