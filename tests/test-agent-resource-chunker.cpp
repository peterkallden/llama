#include "tools/agent/resource/agent-resource-chunker.h"
#include "tools/agent/resource/agent-resource-evidence-session.h"
#include "tools/agent/resource/agent-resource-store.h"
#include "common/resource/resource-text-view.h"

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

    common_runtime_resource_text_view view;
    assert(common_runtime_resource_make_text_view(text, view, error));
    assert(!view.segments.empty());
    std::vector<common_runtime_resource_text_segment> selected;
    const auto rendered = common_runtime_resource_render_text_segments(
        view, 0, text.size(), selected);
    assert(rendered.find("[P1]") == std::string::npos);
    assert(rendered.find("Radiation flux increased") != std::string::npos);
    assert(selected.front().text.find("Radiation flux") != std::string::npos);

    agent_resource_evidence_session session;
    assert(make_agent_resource_evidence_session(
        store, descriptor.uri, authority, policy, session, error));
    common_agent_generation_resource_chunk evidence_slice;
    assert(read_agent_resource_evidence_slice(
        store, authority, session, evidence_slice, error));
    assert(!evidence_slice.segments.empty());
    assert(evidence_slice.text.find("[P") == std::string::npos);
    assert(evidence_slice.text.find("Radiation flux") != std::string::npos);
    assert(evidence_slice.source.lineage.byte_offset == session.plan.ranges[0].byte_offset);
    assert(advance_agent_resource_evidence_session(session, false) ==
        agent_resource_evidence_transition::unresolved);
    assert(session.unresolved_count == 1);

    const std::string crlf_text = "First line\r\ncontinues.\r\n\r\nSecond paragraph.";
    common_runtime_resource_text_view normalized_view;
    assert(common_runtime_resource_make_text_view(crlf_text, normalized_view, error));
    assert(normalized_view.segments.size() == 2);
    const auto & first_segment = normalized_view.segments.front();
    assert(first_segment.text == "First line\ncontinues.");
    assert(crlf_text.substr(first_segment.byte_offset, first_segment.byte_length) ==
        "First line\r\ncontinues.");
    std::vector<common_runtime_resource_text_segment> all_crlf_segments;
    const auto rendered_crlf = common_runtime_resource_render_text_segments(
        normalized_view, 0, normalized_view.source_offsets.back(), all_crlf_segments);
    assert(all_crlf_segments.size() == 2);
    assert(rendered_crlf.find("Second paragraph.") != std::string::npos);
}
