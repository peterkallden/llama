#pragma once

#include "agent-resource-store.h"
#include "common/resource/resource-chunker.h"

struct agent_resource_chunk_range {
    size_t chunk_index = 0;
    size_t chunk_count = 0;
    size_t byte_offset = 0;
    size_t byte_length = 0;
    size_t overlap_bytes = 0;
    std::string boundary;
};

struct agent_resource_chunk_plan {
    agent_resource_descriptor parent;
    common_runtime_resource_chunk_policy policy;
    std::vector<agent_resource_chunk_range> ranges;
};

bool plan_agent_resource_text_chunks(
        const agent_resource_store & store,
        const std::string & uri,
        const agent_resource_read_authority & authority,
        const common_runtime_resource_chunk_policy & policy,
        agent_resource_chunk_plan & out,
        std::string & error);

bool read_agent_resource_chunk(
        const agent_resource_store & store,
        const agent_resource_read_authority & authority,
        const agent_resource_chunk_plan & plan,
        size_t chunk_index,
        std::string & out,
        std::string & error);

common_runtime_resource_ref make_agent_resource_chunk_ref(
        const agent_resource_chunk_plan & plan,
        const agent_resource_chunk_range & range);
