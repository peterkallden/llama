#include "tools/agent/runtime/agent-route-compiler.h"
#include "tools/agent/runtime/agent-dataset-workflow-adapter.h"
#include "plan/plan-in-memory.h"

#include <algorithm>
#include <cassert>
#include <nlohmann/json.hpp>

namespace {

common_plan_state make_workflow() {
    common_plan_state workflow;
    workflow.id = "bootstrap:local:session:s:workflow:dataset";
    workflow.namespace_id = "local";
    workflow.session_id = "s";
    workflow.kind = common_plan_kind::workflow;
    workflow.scope = common_plan_scope::session;
    workflow.workflow_definition = common_plan_workflow_definition{
        "workflow://dataset/analysis", "v1", "dataset", "graph-v1", {},
        {"dataset.resolve", "dataset.inspect", "data.query"},
        {"data.filter", "data.aggregate", "data.join", "data.transform",
            "statistics.describe", "artifact.export"},
        {"context.dataset.available"}};
    return workflow;
}

common_plan_state make_blueprint(bool required) {
    common_plan_state blueprint;
    blueprint.id = "bootstrap:local:session:s:blueprint:dataset-analysis";
    blueprint.namespace_id = "local";
    blueprint.session_id = "s";
    blueprint.kind = common_plan_kind::blueprint;
    blueprint.scope = common_plan_scope::session;
    blueprint.source_revision = "starter@v1";
    blueprint.goal = "Analyse a dataset";
    blueprint.success_criteria = "Verified result";
    blueprint.workflow_policy = required
        ? common_plan_workflow_policy::required
        : common_plan_workflow_policy::preferred;
    blueprint.workflow_bindings = {{"workflow://dataset/analysis", "v1"}};
    blueprint.steps = {{"orient", "Orient", "Resolve dataset"}};
    return blueprint;
}

common_blueprint_candidate make_candidate() {
    common_blueprint_candidate candidate;
    candidate.logical_id = "dataset-analysis";
    candidate.persisted_id = "bootstrap:local:session:s:blueprint:dataset-analysis";
    candidate.description = "Analyse a dataset";
    candidate.source_revision = "starter@v1";
    return candidate;
}

} // namespace

int main() {
    common_plan_in_memory_store store;
    std::string error;
    assert(store.open("", error));
    auto workflow = make_workflow();
    auto blueprint = make_blueprint(false);
    assert(store.create(workflow, error));
    assert(store.create(blueprint, error));

    common_agent_scope scope;
    scope.namespace_id = "local";
    scope.session_id = "s";
    scope.plan_scope = common_plan_scope::session;
    common_agent_request request;
    request.prompt = "Describe the sales dataset";
    request.namespace_id = "local";
    request.session_id = "s";
    request.plan_scope = common_plan_scope::session;

    common_agent_runtime_tooling tooling;
    tooling.profile_tools_active = true;
    // The CLI's built-in profile exposes concrete tools but does not publish
    // configured semantic capability ids.  Route eligibility must derive the
    // stable built-in dataset capability from that host-owned tool view.
    tooling.capabilities = {"dataset.resolve", "dataset.inspect", "statistics.describe",
        "data.query", "data.filter", "data.aggregate", "data.join", "data.transform",
        "artifact.export"};
    tooling.available_context = {"context.dataset.available"};
    tooling.capability_tools = {
        {"dataset.resolve", {"dataset.select"}},
        {"dataset.inspect", {"dataset.inspect"}},
        {"statistics.describe", {"statistics.describe"}},
        {"data.query", {"data.query"}},
        {"data.filter", {"data.filter"}},
        {"data.aggregate", {"data.aggregate"}},
        {"data.join", {"data.join"}},
        {"data.transform", {"data.transform"}},
        {"artifact.export", {"artifact.export"}},
    };
    tooling.tools = {
        {"dataset.select", "select", R"({"type":"object"})"},
        {"dataset.inspect", "inspect", R"({"type":"object"})"},
        {"statistics.describe", "describe", R"({"type":"object"})"},
        {"data.query", "query", R"({"type":"object"})"},
        {"data.filter", "filter", R"({"type":"object"})"},
        {"data.aggregate", "aggregate", R"({"type":"object"})"},
        {"data.join", "join", R"({"type":"object"})"},
        {"data.transform", "transform", R"({"type":"object"})"},
        {"artifact.export", "export", R"({"type":"object"})"},
        {"web.fetch", "fetch", R"({"type":"object"})"},
    };
    common_agent_dataset_descriptor dataset;
    dataset.ref.uri = "dataset://local/sales";
    dataset.columns = {{"amount", common_agent_dataset_column_type::decimal, false}};
    tooling.available_datasets = {dataset};

    common_agent_route_catalog catalog;
    assert(compile_agent_route_catalog(request, store, scope, {make_candidate()}, tooling, "", catalog, error));
    assert(catalog.candidates.size() == 2);
    assert(catalog.candidates[0].id == "normal-plan");
    assert(catalog.candidates[1].kind == common_agent_route_kind::blueprint_workflow);
    assert(catalog.candidates[1].resolved_optional_capabilities.size() == 6);
    common_agent_execution_envelope envelope;
    assert(build_agent_execution_envelope(catalog.candidates[1], tooling, envelope, error));
    assert(envelope.workflow_ref == "workflow://dataset/analysis");
    assert(common_agent_execution_envelope_allows(
        envelope, common_agent_execution_phase::normal, "dataset.inspect"));
    for (const auto * operation : {"data.query", "data.filter", "data.aggregate",
            "data.join", "data.transform"}) {
        assert(common_agent_execution_envelope_allows(
            envelope, common_agent_execution_phase::normal, operation));
    }
    assert(!common_agent_execution_envelope_allows(
        envelope, common_agent_execution_phase::normal, "web.fetch"));
    assert(common_agent_execution_envelope_allows(
        envelope, common_agent_execution_phase::normal, "artifact.export"));
    assert(common_agent_execution_envelope_allows(
        envelope, common_agent_execution_phase::research, "web.fetch"));
    assert(!envelope.fingerprint.empty());

    auto reduced_tooling = tooling;
    reduced_tooling.tools.erase(std::remove_if(reduced_tooling.tools.begin(), reduced_tooling.tools.end(),
        [](const auto & tool) { return tool.name == "data.aggregate"; }), reduced_tooling.tools.end());
    reduced_tooling.capability_tools.erase("data.aggregate");
    reduced_tooling.tools.erase(std::remove_if(reduced_tooling.tools.begin(), reduced_tooling.tools.end(),
        [](const auto & tool) { return tool.name == "artifact.export"; }), reduced_tooling.tools.end());
    reduced_tooling.capability_tools.erase("artifact.export");
    assert(compile_agent_route_catalog(request, store, scope, {make_candidate()},
        reduced_tooling, "", catalog, error));
    assert(catalog.candidates.size() == 2);
    assert(std::find(catalog.candidates[1].resolved_optional_capabilities.begin(),
        catalog.candidates[1].resolved_optional_capabilities.end(), "data.aggregate") ==
        catalog.candidates[1].resolved_optional_capabilities.end());
    common_agent_execution_envelope reduced_envelope;
    assert(build_agent_execution_envelope(catalog.candidates[1], reduced_tooling,
        reduced_envelope, error));
    assert(!common_agent_execution_envelope_allows(
        reduced_envelope, common_agent_execution_phase::normal, "data.aggregate"));
    assert(!common_agent_execution_envelope_allows(
        reduced_envelope, common_agent_execution_phase::normal, "artifact.export"));

    tooling.available_datasets.clear();
    tooling.available_context.clear();
    assert(store.erase(blueprint.id, error));
    assert(store.create(blueprint, error));
    assert(compile_agent_route_catalog(request, store, scope, {make_candidate()}, tooling, "", catalog, error));
    assert(catalog.candidates.size() == 2 &&
        catalog.candidates[1].kind == common_agent_route_kind::blueprint);

    blueprint.workflow_policy = common_plan_workflow_policy::required;
    assert(store.erase(blueprint.id, error));
    assert(store.create(blueprint, error));
    assert(compile_agent_route_catalog(request, store, scope, {make_candidate()}, tooling, "", catalog, error));
    assert(catalog.candidates.size() == 1 && catalog.candidates.front().id == "normal-plan");
    assert(!catalog.rejections.empty());

    // A workflow family unrelated to datasets is eligible from its semantic
    // capability contract alone; the route compiler does not require dataset
    // inventory or a dataset-specific materializer to recognize it.
    common_plan_state repository_workflow;
    repository_workflow.id = "bootstrap:local:session:s:workflow:repository-change";
    repository_workflow.namespace_id = "local";
    repository_workflow.session_id = "s";
    repository_workflow.kind = common_plan_kind::workflow;
    repository_workflow.scope = common_plan_scope::session;
    repository_workflow.workflow_definition = common_plan_workflow_definition{
        "workflow://repository/inspection", "v1", "repository", "graph-repository-v1", {},
        {"repository.read"}, {}, {"context.repository.available"}};
    assert(store.create(repository_workflow, error));

    common_plan_state repository_blueprint = make_blueprint(true);
    repository_blueprint.id = "bootstrap:local:session:s:blueprint:repository-change";
    repository_blueprint.goal = "Inspect and change a repository";
    repository_blueprint.required_capabilities = {"repository.read"};
    repository_blueprint.workflow_bindings = {{"workflow://repository/inspection", "v1"}};
    assert(store.create(repository_blueprint, error));

    common_blueprint_candidate repository_candidate = make_candidate();
    repository_candidate.logical_id = "repository-change";
    repository_candidate.persisted_id = repository_blueprint.id;
    repository_candidate.description = repository_blueprint.goal;
    tooling.capabilities = {"repository.read"};
    tooling.capability_tools = {{"repository.read", {"repository.read"}}};
    tooling.tools = {{"repository.read", "Read repository file", R"({"type":"object"})"}};
    tooling.available_context = {"context.repository.available"};
    tooling.available_datasets.clear();
    assert(compile_agent_route_catalog(request, store, scope, {repository_candidate},
        tooling, "", catalog, error));
    assert(catalog.candidates.size() == 2);
    assert(catalog.candidates[1].kind == common_agent_route_kind::blueprint_workflow);
    assert(catalog.candidates[1].workflow &&
        catalog.candidates[1].workflow->workflow_ref == "workflow://repository/inspection");
    tooling.available_context.clear();
    assert(compile_agent_route_catalog(request, store, scope, {repository_candidate},
        tooling, "", catalog, error));
    assert(catalog.candidates.size() == 1 && catalog.candidates.front().id == "normal-plan");

    common_agent_route_candidate document_route;
    document_route.id = "route:document-analysis";
    document_route.kind = common_agent_route_kind::blueprint_workflow;
    document_route.blueprint_logical_id = "resource-document-analysis";
    document_route.blueprint_revision = "bp-v1";
    document_route.workflow = common_plan_workflow_binding{
        "workflow://resource/document-analysis", "v1"};
    document_route.workflow_revision = "v1";
    document_route.graph_revision = "resource-document-analysis-graph@v1";
    document_route.resolved_tools = {"data.aggregate"};
    auto continuation = make_agent_resource_document_workflow_continuation(
        document_route, {{"data.aggregate",
            R"({"dataset":"host-dataset","measures":[{"function":"sum","column":"amount"}]})",
            "host-analysis-request", false}});
    common_plan_state document_plan;
    document_plan.id = "document-plan";
    document_plan.kind = common_plan_kind::task;
    document_plan.route_binding = common_plan_route_binding{
        document_route.id, document_route.blueprint_logical_id, "bp-v1",
        "workflow://resource/document-analysis", "v1", document_route.graph_revision,
        "env-v1", "policy-v1"};
    document_plan.workflow_definition = common_plan_workflow_definition{
        "workflow://resource/document-analysis", "v1", "resource",
        document_route.graph_revision, {}, {"document.inspect"}, {},
        {"context.resource.available"}};
    common_plan_step table_step{"table", "Select table", "Materialize the selected table"};
    table_step.status = common_plan_step_status::completed;
    table_step.tool_call = common_plan_tool_call{"document.table", R"({"resource":"r1","table":"Budget"})"};
    common_plan_step query_step{"query", "Query table", "Read the requested rows"};
    query_step.status = common_plan_step_status::completed;
    query_step.tool_call = common_plan_tool_call{"data.query", R"({"dataset":"$from_step"})"};
    common_plan_step answer_step{"answer", "Answer", "Report verified result"};
    answer_step.mode = common_plan_step_mode::final_response;
    answer_step.depends_on = {"query"};
    document_plan.steps = {table_step, query_step, answer_step};
    common_plan_observation query_observation;
    query_observation.id = "query-observation";
    query_observation.source = "data.query";
    std::vector<common_plan_step> continuation_steps;
    assert(continuation(document_plan, "query", query_observation,
        continuation_steps, error));
    assert(error.empty() && continuation_steps.size() == 1);
    assert(continuation_steps.front().tool_call &&
        continuation_steps.front().tool_call->name == "data.aggregate" &&
        continuation_steps.front().depends_on == std::vector<std::string>{"query"});
    const auto aggregate_arguments = nlohmann::json::parse(
        continuation_steps.front().tool_call->arguments_json);
    assert(aggregate_arguments["dataset"].value("$from_step", "") == "table" &&
        aggregate_arguments["dataset"].value("$json_pointer", "") == "/dataset");
    document_route.resolved_tools.clear();
    auto denied_continuation = make_agent_resource_document_workflow_continuation(
        document_route, {{"data.aggregate", R"({"measures":[]})", "host-request", false}});
    continuation_steps.clear();
    assert(denied_continuation(document_plan, "query", query_observation,
        continuation_steps, error));
    assert(error.empty() && continuation_steps.empty());
    return 0;
}
