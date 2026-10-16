#include "plan/cozo/plan-cozo.h"

#include <cassert>
#include <filesystem>

int main() {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / "llama-plan-cozo-test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    std::string error;

    {
        common_plan_cozo_store store;
        assert(store.open((dir / "plan.db").string(), error));
        common_plan_state plan;
        plan.id = "p";
        plan.scope = common_plan_scope::global;
        plan.purpose = "persist and resume";
        plan.goal = "persist";
        plan.workflow_policy = common_plan_workflow_policy::required;
        plan.procedure_refs = {"data-understanding-v1"};
        plan.workflow_bindings = {{"workflow://dataset/analysis", "v1"}};
        plan.selected_workflow = common_plan_workflow_binding{"workflow://dataset/analysis", "v1"};
        plan.workflow_definition = common_plan_workflow_definition{
            "workflow://dataset/analysis", "v1", "dataset", "graph-v1", {"dataset.select"},
            {"dataset.resolve", "dataset.inspect"}, {}, {"context.dataset.available"}};
        plan.workflow_definition->required_capabilities = {"dataset.resolve", "dataset.inspect"};
        plan.workflow_definition->optional_capabilities = {"data.aggregate"};
        plan.workflow_definition->required_context = {"context.dataset.available"};
        plan.workflow_definition->continuation_workflow_targets = {"workflow://dataset/quality"};
        plan.workflow_definition->resource_read_policy =
            common_plan_workflow_resource_read_policy::chunked;
        common_plan_workflow_transition choose_depth{
            "choose-depth", "record", "full", common_plan_workflow_transition_kind::model_choice,
            {}, "{}", {}, 1.0f};
        choose_depth.candidate_id_json_pointer = "/code";
        choose_depth.choices = {
            {"abstract", "Use the abstract", "complete", {}},
            {"full-text", "Read full text", "full", {"/result/best_oa_location/pdf_url"}},
        };
        common_plan_workflow_transition fetch_full_text{
            "fetch-full-text", "full", "complete", common_plan_workflow_transition_kind::tool,
            "web.fetch", R"({"max_bytes":8388608})", {}, 1.0f};
        fetch_full_text.argument_bindings = {
            {"/url", "get-work", "/result/best_oa_location/pdf_url", true},
        };
        plan.workflow_definition->start_state = "record";
        plan.workflow_definition->terminal_states = {"complete"};
        plan.workflow_definition->transitions = {choose_depth, fetch_full_text};
        plan.route_binding = common_plan_route_binding{
            "blueprint:dataset-analysis:workflow:workflow://dataset/analysis@v1",
            "dataset-analysis", "bp-v1", "workflow://dataset/analysis", "v1",
            "graph-v1", "env-v1", "route-policy-v1"};
        common_plan_observation observation;observation.id="source";observation.source="web.fetch";
        observation.summary="Host retained source provenance.";observation.confidence=1.0f;
        common_runtime_resource_ref ref;ref.uri="agent-resource://derived";ref.name="paper.txt";
        ref.metadata.source_provider="native";ref.metadata.source_operation="web.fetch";
        ref.metadata.source_request_json=R"({"url":"https://example.org/paper.pdf"})";
        ref.metadata.content_hash="sha256:text";ref.metadata.retrieved_at=1234;
        ref.lineage.parent_uri="agent-resource://original";ref.lineage.chunk_count=1;
        ref.lineage.derivation="resource.process:pdf";
        observation.resource_refs.push_back(ref);
        common_agent_dataset_ref dataset;dataset.uri="dataset://source";dataset.source_provider="public-api";
        dataset.source_operation="list";dataset.source_request_json=R"({"search":"battery recycling"})";
        dataset.content_hash="sha256:data";dataset.retrieved_at=1234;
        observation.dataset_refs.push_back(dataset);plan.observations.push_back(observation);
        assert(store.create(plan, error));

        common_plan_workflow_definition transitioned_workflow;
        transitioned_workflow.workflow_ref = "workflow://dataset/quality";
        transitioned_workflow.workflow_revision = "v2";
        transitioned_workflow.preferred_thinking_mode = "research";
        transitioned_workflow.require_declared_claims = true;
        common_plan_verification_check evidence_check;
        evidence_check.id = "evidence-present";
        evidence_check.phase = common_plan_verification_phase::before_answer;
        evidence_check.left.literal_json = "1";
        transitioned_workflow.verification_checks = {evidence_check};
        transitioned_workflow.family = "dataset";
        transitioned_workflow.graph_revision = "quality-graph-v2";
        transitioned_workflow.allowed_tools = {"dataset.select"};
        transitioned_workflow.required_capabilities = {"dataset.resolve", "dataset.inspect"};
        transitioned_workflow.optional_capabilities = {"data.aggregate"};
        transitioned_workflow.required_context = {"context.dataset.available"};
        transitioned_workflow.continuation_workflow_targets = {"workflow://dataset/analysis"};
        transitioned_workflow.continuation_links = {{
            "fetch-full-text", "complete", "workflow://resource/document-analysis", "v5",
            {{"read-resource", "resource.read", "/uri", "/resources/0/uri", true, "choose-depth"}}}};
        transitioned_workflow.resource_read_policy = common_plan_workflow_resource_read_policy::chunked;
        transitioned_workflow.start_state = "record";
        transitioned_workflow.terminal_states = {"complete"};
        transitioned_workflow.transitions = {choose_depth, fetch_full_text};
        common_plan_route_binding transitioned_binding{
            "blueprint:dataset-quality:workflow:workflow://dataset/quality@v2",
            "dataset-quality", "bp-v2", "workflow://dataset/quality", "v2",
            "quality-graph-v2", "env-v2", "route-policy-v2"};
        common_plan_operation transition;
        transition.kind = common_plan_operation_kind::request_route_transition;
        transition.plan_id = plan.id;
        transition.expected_version = plan.version;
        transition.reason_summary = "host-selected dataset quality continuation";
        transition.route_binding = transitioned_binding;
        transition.workflow_definition = transitioned_workflow;
        transition.route_required_capabilities = {"dataset.resolve", "dataset.inspect"};
        transition.route_procedure_refs = {"data-quality-v1"};
        transition.workflow_continuation_record = common_plan_workflow_continuation_record{
            "fetch-full-text@complete->workflow://resource/document-analysis@v5", "fingerprint-1",
            "workflow://dataset/analysis", "v1", "complete",
            "workflow://dataset/quality", "v2"};
        transition.host_authorized = true;
        assert(store.apply(transition, plan, error));

        auto op = common_plan_operation{};
        op.kind = common_plan_operation_kind::add_step;
        op.plan_id = "p";
        op.expected_version = 1;
        op.reason_summary = "test";
        op.step = common_plan_step{"s", "step", "objective"};
        op.step->intended_contribution = "preserve the persisted evidence";
        op.step->selected_tool = "web.search";
        op.step->tool_call = common_plan_tool_call{"web.search", R"({"query":"cozo"})"};
        op.step->artifact_binding = common_plan_artifact_binding{
            "compose-findings", "research-report", "v2", "findings", "snapshot-7"};
        assert(store.apply(op, plan, error));
        op.expected_version = plan.version;
        op.step = common_plan_step{"legacy-artifact-step", "legacy artifact", "legacy contract"};
        op.step->mode = common_plan_step_mode::artifact_compose;
        op.step->semantic_alias =
            "workflow-transition:compose-summary,artifact-section:research-report@v1:summary";
        assert(store.apply(op, plan, error));
    }

    {
        common_plan_cozo_store store;
        assert(store.open((dir / "plan.db").string(), error));
        auto plan = store.get("p", error);
        assert(plan);
        assert(plan->observations.size()==1);
        const auto & ref=plan->observations.front().resource_refs.front();
        assert(ref.metadata.source_request_json==R"({"url":"https://example.org/paper.pdf"})");
        assert(ref.metadata.content_hash=="sha256:text"&&ref.metadata.retrieved_at==1234);
        assert(ref.lineage.parent_uri=="agent-resource://original"&&ref.lineage.derivation=="resource.process:pdf");
        const auto & dataset=plan->observations.front().dataset_refs.front();
        assert(dataset.source_provider=="public-api"&&dataset.source_operation=="list");
        assert(dataset.source_request_json==R"({"search":"battery recycling"})");
        assert(dataset.content_hash=="sha256:data"&&dataset.retrieved_at==1234);
        assert(plan->scope == common_plan_scope::global && plan->purpose == "persist and resume");
        assert(plan->steps.size() == 2 &&
            plan->steps[0].intended_contribution == "preserve the persisted evidence" &&
            plan->steps[0].tool_call && plan->steps[0].tool_call->name == "web.search");
        assert(plan->steps[0].artifact_binding &&
            plan->steps[0].artifact_binding->transition_id == "compose-findings" &&
            plan->steps[0].artifact_binding->profile_ref == "research-report" &&
            plan->steps[0].artifact_binding->profile_revision == "v2" &&
            plan->steps[0].artifact_binding->section_id == "findings" &&
            plan->steps[0].artifact_binding->snapshot_id == "snapshot-7");
        assert(plan->steps[1].artifact_binding &&
            plan->steps[1].artifact_binding->transition_id == "compose-summary" &&
            plan->steps[1].artifact_binding->profile_ref == "research-report" &&
            plan->steps[1].artifact_binding->profile_revision == "v1" &&
            plan->steps[1].artifact_binding->section_id == "summary");
        assert(plan->version == 3 && plan->workflow_policy == common_plan_workflow_policy::required &&
            plan->procedure_refs == std::vector<std::string>{"data-quality-v1"});
        assert(plan->workflow_bindings.size() == 2 && plan->selected_workflow &&
            plan->selected_workflow->workflow_ref == "workflow://dataset/quality");
        assert(plan->workflow_definition && plan->workflow_definition->family == "dataset" &&
            plan->workflow_definition->workflow_ref == "workflow://dataset/quality" &&
            plan->workflow_definition->resource_read_policy ==
                common_plan_workflow_resource_read_policy::chunked);
        assert(plan->workflow_definition->preferred_thinking_mode == "research");
        assert(plan->workflow_definition->require_declared_claims);
        assert(plan->workflow_definition->verification_checks.size() == 1);
        assert(plan->workflow_definition->verification_checks.front().left.literal_json == "1");
        assert(plan->workflow_definition->required_capabilities ==
            std::vector<std::string>({"dataset.resolve", "dataset.inspect"}) &&
            plan->workflow_definition->optional_capabilities ==
                std::vector<std::string>({"data.aggregate"}) &&
            plan->workflow_definition->required_context ==
                std::vector<std::string>({"context.dataset.available"}) &&
            plan->workflow_definition->continuation_workflow_targets ==
                std::vector<std::string>({"workflow://dataset/analysis"}));
        assert(plan->workflow_definition->continuation_links.size() == 1 &&
            plan->workflow_definition->continuation_links.front().source_transition_id == "fetch-full-text" &&
            plan->workflow_definition->continuation_links.front().source_terminal_state == "complete" &&
            plan->workflow_definition->continuation_links.front().argument_bindings.front().target_transition_id == "read-resource" &&
            plan->workflow_definition->continuation_links.front().argument_bindings.front().source_transition_id == "choose-depth");
        assert(plan->workflow_continuation_history.size() == 1 &&
            plan->workflow_continuation_history.front().source_fingerprint == "fingerprint-1");
        assert(plan->workflow_definition->transitions.size() == 2 &&
            plan->workflow_definition->transitions[0].choices.size() == 2 &&
            plan->workflow_definition->transitions[0].candidate_id_json_pointer == "/code" &&
            plan->workflow_definition->transitions[0].choices[1].id == "full-text" &&
            plan->workflow_definition->transitions[1].argument_bindings.size() == 1 &&
            plan->workflow_definition->transitions[1].argument_bindings[0].source_transition_id == "get-work");
        assert(plan->route_binding &&
            plan->route_binding->route_id.find("blueprint:dataset-quality") == 0 &&
            plan->route_binding->execution_envelope_fingerprint == "env-v2");
        auto events = store.history("p", error);
        assert(events.size() == 3);
        assert(store.erase("p", error));
    }

    fs::remove_all(dir, ec);
    return 0;
}
