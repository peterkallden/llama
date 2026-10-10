#include "tools/agent/resource/agent-resource-chunker.h"
#include "tools/agent/resource/agent-resource-store.h"

#include <cassert>
#include <memory>

int main() {
    const std::string text =
        "Radiation flux increased during the campaign.\n"
        "The camera remained operational in the plasma experiment.\n"
        "The shield used borated polyethylene.";
    auto blobs = std::make_shared<agent_in_memory_blob_store>();
    auto catalog = std::make_unique<agent_in_memory_resource_catalog>();
    agent_catalogued_resource_store store(blobs, std::move(catalog));

    agent_resource_put_request put;
    put.name = "paper-ocr.txt";
    put.mime_type = "text/plain";
    put.text = text;
    put.turn_id = "chunk-test";
    agent_resource_descriptor descriptor;
    std::string error;
    assert(store.put_text(put, descriptor, error));

    agent_resource_runtime runtime;
    runtime.store = &store;
    runtime.turn_id = "chunk-test";
    const auto authority = make_agent_resource_read_authority(runtime);
    const common_runtime_resource_chunk_policy policy{48, 8};
    agent_resource_chunk_plan plan;
    assert(plan_agent_resource_text_chunks(store, descriptor.uri, authority, policy, plan, error));
    assert(plan.ranges.size() > 1);

    for (size_t i = 0; i < plan.ranges.size(); ++i) {
        std::string actual;
        assert(read_agent_resource_chunk(store, authority, plan, i, actual, error));
        const auto & range = plan.ranges[i];
        assert(actual == text.substr(range.byte_offset, range.byte_length));
        assert(actual.find_first_not_of(" \n\r\t") != std::string::npos);
    }
    const auto ref = make_agent_resource_chunk_ref(plan, plan.ranges[1]);
    assert(common_runtime_resource_lineage_is_chunk(ref.lineage));
    assert(ref.lineage.parent_uri == descriptor.uri);
    assert(ref.lineage.chunk_index == 1);
    assert(ref.lineage.chunk_count == plan.ranges.size());
}
