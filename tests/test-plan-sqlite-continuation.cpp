#include "plan/sqlite/plan-sqlite.h"

#include <cassert>
#include <filesystem>
#include <cstdlib>
#include <iostream>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "llama-agent-plan-continuation.sqlite";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::string error;
    common_plan_sqlite_store store;
    assert(store.open(path.string(), error));

    common_plan_state plan;
    plan.id = "continuation-roundtrip";
    plan.status = common_plan_status::active;
    common_plan_step source;
    source.id = "source-tool";
    source.status = common_plan_step_status::completed;
    source.tool_call = common_plan_tool_call{"dataset.inspect", "{}"};
    source.artifact_binding = common_plan_artifact_binding{
        "compose-findings", "research-report", "v2", "findings", "snapshot-7"};
    plan.steps.push_back(source);
    common_plan_step legacy_artifact_step;
    legacy_artifact_step.id = "legacy-artifact-step";
    legacy_artifact_step.mode = common_plan_step_mode::artifact_compose;
    legacy_artifact_step.semantic_alias =
        "workflow-transition:compose-summary,artifact-section:research-report@v1:summary";
    plan.steps.push_back(legacy_artifact_step);
    common_plan_observation observation;
    observation.id = "tool:source-tool:dataset.inspect";
    observation.source = "dataset.inspect";
    observation.summary = R"({"dataset":{"uri":"dataset://import/one"}})";
    observation.dataset_refs.push_back({"dataset://import/one", "catalogue", 250, 8,
        "resource://eurostat/toc", "text/tab-separated-values", "", "", "", 0, "hash-1"});
    observation.resource_refs.push_back({"resource://eurostat/toc", "toc", "", "text/plain", 4096});
    observation.resource_refs.back().metadata.source_provider="native";
    observation.resource_refs.back().metadata.source_operation="web.fetch";
    observation.resource_refs.back().metadata.source_request_json=R"({"url":"https://example.org/catalogue"})";
    observation.resource_refs.back().metadata.retrieved_at=1234;
    observation.resource_refs.back().metadata.content_hash="sha256:catalogue";
    observation.resource_read_provenance=common_plan_resource_read_provenance{
        "task-fingerprint", "question", "resource://eurostat/toc", "sha256:catalogue",
        "text/plain", "resource://eurostat/original"};
    plan.observations.push_back(observation);
    plan.workflow_definition = common_plan_workflow_definition{};
    plan.workflow_definition->workflow_ref = "workflow://eurostat/catalog";
    plan.workflow_definition->workflow_revision = "v3";
    plan.workflow_definition->family = "dataset";
    plan.workflow_definition->graph_revision = "catalog-graph-v3";
    plan.workflow_definition->required_capabilities = {"dataset.inspect"};
    common_plan_workflow_continuation_link link;
    link.source_transition_id = "materialize";
    link.source_terminal_state = "complete";
    link.target_workflow_ref = "workflow://eurostat/metadata";
    link.target_workflow_revision = "v2";
    link.argument_bindings.push_back({"inspect", "dataset.inspect", "/resource", "/dataset/uri", true, "choose-code"});
    plan.workflow_definition->continuation_links.push_back(link);
    assert(store.create(plan, error));

    common_plan_route_binding route{"route-1", {}, {}, "workflow://eurostat/metadata", "v2",
        "metadata-graph-v2", "envelope-1", "policy-1"};
    route.continuation_link_id = "materialize@complete->workflow://eurostat/metadata@v2";
    route.continuation_source_fingerprint = "fp-1";
    common_plan_workflow_definition target;
    target.workflow_ref = "workflow://eurostat/metadata";
    target.workflow_revision = "v2";
    target.preferred_thinking_mode = "deliberate";
    target.require_declared_claims = true;
    common_plan_verification_check evidence_check;
    evidence_check.id = "evidence-present";
    evidence_check.phase = common_plan_verification_phase::before_answer;
    evidence_check.left.literal_json = "1";
    target.verification_checks = {evidence_check};
    target.family = "openapi";
    target.graph_revision = "metadata-graph-v2";
    target.required_capabilities = {"dataset.inspect"};
    target.continuation_links = {link};
    common_plan_workflow_transition choice;
    choice.id = "choose-code";
    choice.from_state = "start";
    choice.to_state = "selected";
    choice.kind = common_plan_workflow_transition_kind::model_choice;
    choice.candidate_json_pointer = "/result/rows";
    choice.candidate_id_json_pointer = "/code";
    common_plan_workflow_transition search;
    search.id = "search-catalogue";
    search.from_state = "selected";
    search.to_state = "searched";
    search.tool_name = "data.query";
    search.model_instruction = "Use the verified dataset code.";
    target.transitions.push_back(choice);
    target.transitions.push_back(search);
    common_plan_operation transition;
    transition.kind = common_plan_operation_kind::request_route_transition;
    transition.plan_id = plan.id;
    transition.expected_version = plan.version;
    transition.route_binding = route;
    transition.workflow_definition = target;
    transition.route_required_capabilities = target.required_capabilities;
    transition.host_authorized = true;
    transition.workflow_continuation_record = common_plan_workflow_continuation_record{
        "materialize@complete->workflow://eurostat/metadata@v2", "fp-1",
        "workflow://eurostat/catalog", "v3", "complete", "workflow://eurostat/metadata", "v2"};
    common_plan_state updated;
    assert(store.apply(transition, updated, error));
    store.close();

    assert(store.open(path.string(), error));
    const auto restored = store.get(plan.id, error);
    assert(restored && restored->workflow_continuation_history.size() == 1);
    assert(restored->workflow_continuation_history.front().source_fingerprint == "fp-1");
    assert(restored->observations.size() == 1 && restored->observations.front().dataset_refs.size() == 1);
    assert(restored->observations.front().dataset_refs.front().content_hash == "hash-1");
    assert(restored->observations.front().resource_refs.size() == 1 &&
        restored->observations.front().resource_refs.front().uri == "resource://eurostat/toc");
    const auto & metadata=restored->observations.front().resource_refs.front().metadata;
    assert(metadata.source_provider=="native"&&metadata.source_operation=="web.fetch");
    assert(metadata.source_request_json==R"({"url":"https://example.org/catalogue"})");
    assert(metadata.retrieved_at==1234&&metadata.content_hash=="sha256:catalogue");
    const auto & read=restored->observations.front().resource_read_provenance;
    assert(read&&read->task_fingerprint=="task-fingerprint"&&read->intent=="question");
    assert(read->source_uri=="resource://eurostat/toc"&&read->source_fingerprint=="sha256:catalogue");
    assert(read->representation=="text/plain"&&read->source_parent_uri=="resource://eurostat/original");
    assert(restored->workflow_definition && restored->workflow_definition->workflow_ref == "workflow://eurostat/metadata");
    assert(restored->workflow_definition->preferred_thinking_mode == "deliberate");
    assert(restored->workflow_definition->require_declared_claims);
    assert(restored->workflow_definition->verification_checks.front().left.literal_json == "1");
    assert(restored->route_binding && restored->route_binding->workflow_revision == "v2");
    assert(restored->route_binding->continuation_link_id == route.continuation_link_id &&
        restored->route_binding->continuation_source_fingerprint == "fp-1");
    assert(restored->route_binding->workflow_step_start_index == 2);
    assert(restored->workflow_definition->continuation_links.front().argument_bindings.front().source_transition_id == "choose-code");
    assert(restored->workflow_definition->transitions.front().candidate_id_json_pointer == "/code");
    assert(restored->workflow_definition->transitions.back().model_instruction ==
        "Use the verified dataset code.");
    assert(restored->steps.size() == 2 && restored->steps.front().artifact_binding &&
        restored->steps.front().artifact_binding->transition_id == "compose-findings" &&
        restored->steps.front().artifact_binding->profile_ref == "research-report" &&
        restored->steps.front().artifact_binding->profile_revision == "v2" &&
        restored->steps.front().artifact_binding->section_id == "findings" &&
        restored->steps.front().artifact_binding->snapshot_id == "snapshot-7");
    assert(restored->steps.back().artifact_binding &&
        restored->steps.back().artifact_binding->transition_id == "compose-summary" &&
        restored->steps.back().artifact_binding->profile_ref == "research-report" &&
        restored->steps.back().artifact_binding->profile_revision == "v1" &&
        restored->steps.back().artifact_binding->section_id == "summary");
    store.close();

    // A real close/reopen preserves the task contract independently of the
    // original continuation fixture. These checks remain active with NDEBUG.
    const std::filesystem::path analysis_path = path.string() + ".analysis";
    common_plan_state analysis;
    analysis.id = "analysis-result-roundtrip";
    analysis.workflow_definition = common_plan_workflow_definition{};
    analysis.workflow_definition->workflow_ref = "workflow://dataset/analysis";
    analysis.workflow_definition->workflow_revision = "v4";
    analysis.workflow_definition->result_contracts = {{"statistics.describe","statistics.describe.v1"}};
    common_plan_step measure;
    measure.id = "measure"; measure.title = "Describe amount";
    measure.tool_call = common_plan_tool_call{"statistics.describe", R"({"dataset":"dataset://orders","columns":["amount"]})"};
    measure.result_summary = "Pending analysis";
    measure.tool_call->result_requirement_json = R"({"schema":"analysis-result.v1","profile":"statistics.describe.v1","value":{"dataset":"dataset://orders","columns":["amount"],"measures":["count","mean"],"group_by":[],"coverage":"complete"}})";
    analysis.steps = {measure};
    {
        common_plan_sqlite_store writer;
        if (!writer.open(analysis_path.string(), error) || !writer.create(analysis, error)) {
            std::cerr << "analysis persistence failed: " << error << '\n'; std::exit(1);
        }
    }
    {
        common_plan_sqlite_store reader;
        if (!reader.open(analysis_path.string(), error)) { std::cerr << error << '\n'; std::exit(1); }
        const auto saved = reader.get(analysis.id, error);
        if (!saved || saved->steps.size() != 1 || !saved->steps[0].tool_call ||
                saved->steps[0].tool_call->result_requirement_json != measure.tool_call->result_requirement_json ||
                !saved->workflow_definition || saved->workflow_definition->result_contracts != analysis.workflow_definition->result_contracts) {
            std::cerr << "analysis result contract lost on restart\n"; std::exit(1);
        }
    }
    std::filesystem::remove(analysis_path);
    std::filesystem::remove(path, ignored);
    return 0;
}
