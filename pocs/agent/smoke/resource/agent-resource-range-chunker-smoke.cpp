#include "tools/agent/resource/agent-resource-chunker.h"

#include <cstdio>
#include <memory>

int main() {
    std::string error;
    auto blob_store = std::make_shared<agent_in_memory_blob_store>();
    agent_catalogued_resource_store store(
        blob_store,
        std::make_unique<agent_in_memory_resource_catalog>());

    agent_resource_descriptor descriptor;
    if (!store.put_text({
            "chunk-input.md", "Chunk input", "text/markdown",
            "Heading\n\nFirst paragraph with a bounded decision.\n\nSecond paragraph with verification.",
            common_runtime_resource_scope::turn,
            "local", "session-1", "project-1", "turn-1", "tool-1",
            "native", "test", 0, 0, {}, {},
        }, descriptor, error)) {
        std::fprintf(stderr, "put_text failed: %s\n", error.c_str());
        return 1;
    }

    agent_resource_read_authority authority;
    authority.session_id = "session-1";
    authority.project_id = "project-1";
    authority.turn_id = "turn-1";
    agent_resource_chunk_plan plan;
    if (!plan_agent_resource_text_chunks(
                store, descriptor.uri, authority, {32, 4}, plan, error)) {
        std::fprintf(stderr, "chunk planning failed: %s\n", error.c_str());
        return 1;
    }
    if (plan.ranges.size() < 2 || plan.ranges.front().chunk_count != plan.ranges.size()) {
        std::fprintf(stderr, "chunk plan metadata is incomplete\n");
        return 1;
    }
    std::string chunk_text;
    if (!read_agent_resource_chunk(store, authority, plan, 1, chunk_text, error) ||
            chunk_text.empty()) {
        std::fprintf(stderr, "bounded chunk read failed: %s\n", error.c_str());
        return 1;
    }
    const auto chunk_ref = make_agent_resource_chunk_ref(plan, plan.ranges[1]);
    if (chunk_ref.lineage.parent_uri != descriptor.uri ||
            chunk_ref.lineage.chunk_count != plan.ranges.size() ||
            chunk_ref.lineage.byte_length != plan.ranges[1].byte_length) {
        std::fprintf(stderr, "chunk reference lineage is incomplete\n");
        return 1;
    }
    if (read_agent_resource_chunk(store, authority, plan, plan.ranges.size(), chunk_text, error) ||
            error != "resource chunk index is out of range") {
        std::fprintf(stderr, "invalid chunk index was not rejected\n");
        return 1;
    }
    std::printf("planned_chunks=%zu\n", plan.ranges.size());
    return 0;
}
