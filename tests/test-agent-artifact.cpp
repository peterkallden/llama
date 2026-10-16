#include "agent/artifact/artifact-contract.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    common_agent_artifact_profile profile;
    profile.id = "research";
    profile.revision = "v1";
    profile.title = "Research report";
    profile.purpose = "Grounded findings and explicit gaps.";
    profile.sections = {{"findings", "Findings", "Summarize supported findings.", {"web.search"}},
        {"limitations", "Limitations", "Record unresolved coverage.", {}}};
    std::string error;
    assert(common_agent_artifact_profile_valid(profile, error));

    common_plan_state plan;
    plan.id = "plan-1";
    plan.version = 7;
    common_runtime_resource_ref source;
    source.uri = "resource://session/article";
    source.name = "article.txt";
    source.mime_type = "text/plain";
    plan.observations.push_back({"tool:search:1", "web.search", R"({"claim":"A finding"})", 1.0f,
        {"evidence-1"}, {source}, 1, {}});

    common_agent_artifact_snapshot snapshot;
    assert(common_agent_artifact_snapshot_from_plan(plan, profile, snapshot, error));
    assert(snapshot.plan_version == 7 && snapshot.observations.size() == 1);
    std::string serialized;
    assert(common_agent_artifact_snapshot_to_json(snapshot, serialized, error));
    common_agent_artifact_snapshot restored;
    assert(common_agent_artifact_snapshot_from_json(serialized, restored, error));
    assert(restored.fingerprint == snapshot.fingerprint);
    assert(restored.observations.front().resource_refs.front().uri == source.uri);

    common_agent_artifact_section section;
    assert(common_agent_artifact_parse_section_dsl(
        "block: observation\ntext: A finding from the source.\nevidence: evidence-1\nsource: tool:search:1",
        profile.sections.front(), snapshot, section, error));
    assert(section.blocks.size() == 1);
    assert(section.blocks.front().validation == common_agent_artifact_validation::references_resolve);
    common_agent_artifact_section gap_section;
    assert(common_agent_artifact_parse_section_dsl(
        "block: conclusion\ntext: Unsupported conclusion.\nevidence: invented-id",
        profile.sections.front(), snapshot, gap_section, error));
    assert(gap_section.blocks.front().validation == common_agent_artifact_validation::gap);

    // Once captured, the snapshot is immutable across later report sections
    // and unrelated observations appended to the active plan.
    plan.observations.push_back({"artifact-snapshot:plan-1", "artifact_snapshot", serialized, 1.0f, {}, {}, 2, {}});
    plan.observations.push_back({"late-observation", "web.fetch", "late data", 1.0f, {"evidence-2"}, {}, 3, {}});
    common_agent_artifact_snapshot frozen;
    assert(common_agent_artifact_snapshot_from_plan(plan, profile, frozen, error));
    assert(frozen.fingerprint == snapshot.fingerprint);
    assert(frozen.plan_version == snapshot.plan_version);
    assert(frozen.observations.size() == 1);

    common_agent_artifact_document document;
    assert(common_agent_artifact_document_from_plan(plan, profile, "resource://session/snapshot", document, error));
    assert(document.sections.size() == 2);
    assert(document.sections[0].blocks.size() == 1);
    assert(document.sections[1].blocks.front().validation == common_agent_artifact_validation::gap);
    std::string markdown;
    assert(common_agent_artifact_render_markdown(document, markdown, error));
    assert(markdown.find("## Findings") != std::string::npos);
    assert(markdown.find("No validated content was produced") != std::string::npos);
    assert(markdown.find("Evidence status: gap") != std::string::npos);

    common_plan_state empty_plan;
    empty_plan.id = "plan-empty";
    empty_plan.version = 1;
    common_agent_artifact_document empty_document;
    assert(common_agent_artifact_document_from_plan(empty_plan, profile, "resource://session/empty-snapshot", empty_document, error));
    assert(!empty_document.coverage_complete);
    assert(empty_document.unresolved_gaps.size() == 1);
    assert(common_agent_artifact_render_markdown(empty_document, markdown, error));
    assert(markdown.find("Coverage incomplete") != std::string::npos);

    common_plan_state exported_plan;
    common_plan_step publish;
    publish.id = "publish";
    publish.status = common_plan_step_status::completed;
    publish.tool_call = common_plan_tool_call{"artifact.export", R"({"format":"markdown"})"};
    exported_plan.steps.push_back(publish);
    exported_plan.observations.push_back({"tool:publish:artifact.export", "artifact.export",
        R"({"resource":"agent-resource://resource/md","report_resource":"agent-resource://resource/report","snapshot_resource":"agent-resource://resource/snapshot","markdown_resource":"agent-resource://resource/md","profile_id":"research","coverage_complete":false})",
        1.0f, {}, {}, 0, {}});
    std::string completion;
    assert(common_agent_artifact_completion_summary(exported_plan, completion));
    assert(completion.find("coverage is incomplete") != std::string::npos);
    assert(completion.find("agent-resource://resource/report") != std::string::npos);
    assert(completion.find("agent-resource://resource/md") != std::string::npos);
    exported_plan.observations.back().summary =
        R"({"resource":"agent-resource://resource/md","report_resource":"agent-resource://resource/report","snapshot_resource":"agent-resource://resource/snapshot","markdown_resource":"agent-resource://resource/md","profile_id":"research","coverage_complete":true})";
    assert(common_agent_artifact_completion_summary(exported_plan, completion));
    assert(completion.find("from the frozen source snapshot") != std::string::npos);
    exported_plan.observations.back().summary = R"({"structuredContent":{"report_resource":"agent-resource://resource/report","snapshot_resource":"agent-resource://resource/snapshot","markdown_resource":"agent-resource://resource/md","profile_id":"research","coverage_complete":false}})";
    assert(common_agent_artifact_completion_summary(exported_plan, completion));
    assert(completion.find("coverage is incomplete") != std::string::npos);
    exported_plan.observations.back().summary = R"({"report_resource":"","markdown_resource":"","coverage_complete":false})";
    assert(!common_agent_artifact_completion_summary(exported_plan, completion));
    assert(completion.empty());
    return 0;
}
