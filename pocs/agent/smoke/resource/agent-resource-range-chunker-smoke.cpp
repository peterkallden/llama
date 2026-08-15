#include "tools/agent/resource/agent-resource-chunker.h"
#include "agent/input-resources.h"

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
    const std::string rendered = common_agent_render_input_resource_context(
        {{chunk_ref, "resource_chunk", true}}, 2048);
    if (rendered.find("chunk_index=1") == std::string::npos ||
            rendered.find("parent_uri=" + descriptor.uri) == std::string::npos ||
            rendered.find("resource_read") == std::string::npos) {
        std::fprintf(stderr, "chunk input view did not expose bounded read metadata\n");
        return 1;
    }
    common_runtime_resource_ref csv_resource;
    csv_resource.uri = "agent-resource://resource/csv";
    csv_resource.name = "cities.csv";
    csv_resource.mime_type = "text/csv";
    const auto csv_view = common_agent_render_input_resource_context(
        {{csv_resource, "reference", true}}, 2048);
    if (csv_view.find("id=r1") == std::string::npos ||
            csv_view.find("inspection=resource_inspect,dataset.schema,dataset.sample") == std::string::npos ||
            csv_view.find("statistics.describe") == std::string::npos) {
        std::fprintf(stderr, "CSV input view did not expose bounded inspection seams\n");
        return 1;
    }
    common_runtime_resource_ref document_resource = csv_resource;
    document_resource.name = "report.docx";
    document_resource.mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
    const auto document_view = common_agent_render_input_resource_context(
        {{document_resource, "reference", true}}, 2048);
    if (document_view.find("inspection=resource_inspect,document.tables,resource_read") == std::string::npos) {
        std::fprintf(stderr, "document input view did not expose document inspection seams\n");
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
