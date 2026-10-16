#include "tools/agent/runtime/agent-route-compiler.h"
#include "tools/agent/runtime/agent-dataset-workflow-adapter.h"
#include "tools/agent/runtime/agent-workflow-transition.h"
#include "plan/plan-in-memory.h"
#include "agent/agent-package-json.h"
#include "memory/memory-in-memory.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
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
    common_plan_workflow_transition static_fetch{
        "fetch-toc", "catalog_pending", "catalog_fetched",
        common_plan_workflow_transition_kind::tool, "web.fetch",
        R"({"url":"https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt?lang=en","accept":"text/plain","max_bytes":4194304})"};
    common_chat_tool web_fetch_tool{
        "web.fetch", "Fetch a public resource",
        R"({"type":"object","required":["url"],"properties":{"url":{"type":"string"},"accept":{"type":"string"},"max_bytes":{"type":"integer"}}})"};
    std::string static_arguments, static_error;
    assert(common_agent_workflow_static_tool_arguments(
        static_fetch, web_fetch_tool, static_arguments, static_error));
    const auto normalized_fetch = nlohmann::json::parse(static_arguments);
    assert(normalized_fetch["accept"] == "text/plain");
    assert(normalized_fetch["url"].get<std::string>().find("/catalogue/toc/txt?lang=en") != std::string::npos);
    auto incomplete_fetch = static_fetch;
    incomplete_fetch.arguments_template_json = R"({"accept":"text/plain"})";
    assert(!common_agent_workflow_static_tool_arguments(
        incomplete_fetch, web_fetch_tool, static_arguments, static_error));
    assert(static_error.empty()); // Model may fill a missing required URL.
    incomplete_fetch.arguments_template_json = R"({"accept":5})";
    assert(!common_agent_workflow_static_tool_arguments(
        incomplete_fetch, web_fetch_tool, static_arguments, static_error));
    assert(static_error.find("static arguments violate") != std::string::npos);
    auto invalid_fetch = static_fetch;
    invalid_fetch.arguments_template_json = R"({"url":"https://example.org","accept":5})";
    assert(!common_agent_workflow_static_tool_arguments(
        invalid_fetch, web_fetch_tool, static_arguments, static_error));
    assert(static_error.find("static arguments violate") != std::string::npos);

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
        {"memory.search", "Search scoped memory", R"({"type":"object"})"},
        {"memory.get", "Read scoped memory", R"({"type":"object"})"},
        {"memory.remember", "Propose memory", R"({"type":"object"})"},
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
        envelope, common_agent_execution_phase::normal, "memory.search"));
    assert(common_agent_execution_envelope_allows(
        envelope, common_agent_execution_phase::reflection, "memory.get"));
    assert(common_agent_execution_envelope_allows(
        envelope, common_agent_execution_phase::research, "memory.search"));
    assert(!common_agent_execution_envelope_allows(
        envelope, common_agent_execution_phase::normal, "memory.remember"));
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

    // OpenAPI onboarding first fetches and admits a specification. Read
    // operations are optional because the dynamic provider is resolved on
    // the following turn, after openapi.connect has admitted it.
    common_plan_state openapi_workflow;
    openapi_workflow.id = "bootstrap:local:session:s:workflow:openapi-onboarding";
    openapi_workflow.namespace_id = "local";
    openapi_workflow.session_id = "s";
    openapi_workflow.kind = common_plan_kind::workflow;
    openapi_workflow.scope = common_plan_scope::session;
    openapi_workflow.workflow_definition = common_plan_workflow_definition{
        "workflow://openapi/resource-onboarding", "v1", "openapi",
        "graph-openapi-onboarding-v1", {},
        {"web.fetch", "openapi.connect"},
        {"web.search", "openapi.read", "artifact.export"},
        {"context.network.allowed"}};
    assert(store.create(openapi_workflow, error));

    common_plan_state openapi_blueprint = make_blueprint(true);
    openapi_blueprint.id = "bootstrap:local:session:s:blueprint:openapi-onboarding";
    openapi_blueprint.goal = "Fetch an OpenAPI specification and use its read-only API";
    openapi_blueprint.required_capabilities = {"workflow.openapi"};
    openapi_blueprint.workflow_bindings = {
        {"workflow://openapi/resource-onboarding", "v1"}};
    assert(store.create(openapi_blueprint, error));

    common_blueprint_candidate openapi_candidate = make_candidate();
    openapi_candidate.logical_id = "openapi-onboarding";
    openapi_candidate.persisted_id = openapi_blueprint.id;
    openapi_candidate.description = openapi_blueprint.goal;
    tooling.capabilities = {"workflow.openapi", "web.fetch", "openapi.connect"};
    tooling.capability_tools = {
        {"web.fetch", {"web.fetch"}},
        {"openapi.connect", {"openapi.connect"}}};
    tooling.tools = {
        {"web.fetch", "Fetch a public resource", R"({"type":"object"})"},
        {"openapi.connect", "Admit a fetched OpenAPI specification", R"({"type":"object"})"}};
    tooling.available_context = {"context.network.allowed"};
    assert(compile_agent_route_catalog(request, store, scope, {openapi_candidate},
        tooling, "", catalog, error));
    assert(catalog.candidates.size() == 2);
    const auto & openapi_route = catalog.candidates[1];
    assert(openapi_route.kind == common_agent_route_kind::blueprint_workflow);
    assert(openapi_route.resolved_tools.size() == 2);
    assert(std::find(openapi_route.resolved_optional_capabilities.begin(),
        openapi_route.resolved_optional_capabilities.end(), "openapi.read") ==
        openapi_route.resolved_optional_capabilities.end());

    tooling.capabilities.push_back("openapi.read");
    tooling.capability_tools["openapi.read"] = {"openalex.listWorks"};
    tooling.tools.push_back({"openalex.listWorks", "List works", R"({"type":"object"})"});
    assert(compile_agent_route_catalog(request, store, scope, {openapi_candidate},
        tooling, "", catalog, error));
    assert(catalog.candidates.size() == 2);
    const auto & admitted_openapi_route = catalog.candidates[1];
    assert(std::find(admitted_openapi_route.resolved_optional_capabilities.begin(),
        admitted_openapi_route.resolved_optional_capabilities.end(), "openapi.read") !=
        admitted_openapi_route.resolved_optional_capabilities.end());
    assert(std::find(admitted_openapi_route.resolved_tools.begin(),
        admitted_openapi_route.resolved_tools.end(), "openalex.listWorks") !=
        admitted_openapi_route.resolved_tools.end());

    // A host-configured provider is available immediately; it does not need
    // the resource-onboarding workflow used for an unknown API. OpenAlex
    // research routes are therefore eligible only after the host has exposed
    // its read-only operations and declared the corresponding context fact.
    common_plan_state openalex_workflow;
    openalex_workflow.id = "bootstrap:local:session:s:workflow:openalex-work-search";
    openalex_workflow.namespace_id = "local";
    openalex_workflow.session_id = "s";
    openalex_workflow.kind = common_plan_kind::workflow;
    openalex_workflow.scope = common_plan_scope::session;
    openalex_workflow.workflow_definition = common_plan_workflow_definition{
        "workflow://openalex/work-search", "v1", "openapi", "openalex-work-search-graph@v1", {},
        {"openapi.read", "resource.read"}, {"artifact.export"}, {"context.openapi.available"}};
    assert(store.create(openalex_workflow, error));
    common_plan_state openalex_blueprint = make_blueprint(true);
    openalex_blueprint.id = "bootstrap:local:session:s:blueprint:openalex-work-search";
    openalex_blueprint.goal = "Find scholarly work candidates through configured OpenAlex";
    openalex_blueprint.required_capabilities = {"tool.openapi", "workflow.openapi"};
    openalex_blueprint.workflow_bindings = {{"workflow://openalex/work-search", "v1"}};
    assert(store.create(openalex_blueprint, error));
    common_blueprint_candidate openalex_candidate = make_candidate();
    openalex_candidate.logical_id = "openalex-work-search";
    openalex_candidate.persisted_id = openalex_blueprint.id;
    openalex_candidate.description = openalex_blueprint.goal;
    tooling.capabilities = {"openapi.read", "resource.read"};
    tooling.capability_tools = {
        {"openapi.read", {"openalex.listWorks", "openalex.getWork",
            "openalex.listAuthors", "openalex.getAuthor",
            "openalex.listInstitutions", "openalex.getInstitution",
            "openalex.listTopics", "openalex.getTopic"}},
        {"resource.read", {"resource.read"}},
    };
    tooling.tools = {
        {"openalex.listWorks", "List OpenAlex works", R"({"type":"object"})"},
        {"openalex.getWork", "Get OpenAlex work", R"({"type":"object"})"},
        {"openalex.listAuthors", "List OpenAlex authors", R"({"type":"object"})"},
        {"openalex.getAuthor", "Get OpenAlex author", R"({"type":"object"})"},
        {"openalex.listInstitutions", "List OpenAlex institutions", R"({"type":"object"})"},
        {"openalex.getInstitution", "Get OpenAlex institution", R"({"type":"object"})"},
        {"openalex.listTopics", "List OpenAlex topics", R"({"type":"object"})"},
        {"openalex.getTopic", "Get OpenAlex topic", R"({"type":"object"})"},
        {"resource.read", "Read host resource", R"({"type":"object"})"},
    };
    tooling.available_context = {"context.openapi.available"};
    assert(compile_agent_route_catalog(request, store, scope, {openalex_candidate},
        tooling, "", catalog, error));
    assert(catalog.candidates.size() == 2);
    assert(catalog.candidates[1].kind == common_agent_route_kind::blueprint_workflow);
    assert(std::find(catalog.candidates[1].resolved_tools.begin(),
        catalog.candidates[1].resolved_tools.end(), "openalex.listWorks") !=
        catalog.candidates[1].resolved_tools.end());
    assert(std::find(catalog.candidates[1].resolved_tools.begin(),
        catalog.candidates[1].resolved_tools.end(), "openalex.listTopics") !=
        catalog.candidates[1].resolved_tools.end());

    common_plan_state openalex_entity_workflow;
    openalex_entity_workflow.id = "bootstrap:local:session:s:workflow:openalex-entity-filtered-work-search";
    openalex_entity_workflow.namespace_id = "local";
    openalex_entity_workflow.session_id = "s";
    openalex_entity_workflow.kind = common_plan_kind::workflow;
    openalex_entity_workflow.scope = common_plan_scope::session;
    openalex_entity_workflow.workflow_definition = common_plan_workflow_definition{
        "workflow://openalex/entity-filtered-work-search", "v1", "openapi",
        "openalex-entity-filtered-work-search-graph@v1", {},
        {"openapi.read", "resource.read"}, {}, {"context.openapi.available"}};
    assert(store.create(openalex_entity_workflow, error));
    common_plan_state openalex_entity_blueprint = make_blueprint(true);
    openalex_entity_blueprint.id =
        "bootstrap:local:session:s:blueprint:openalex-entity-filtered-work-search";
    openalex_entity_blueprint.goal =
        "Find works by title, author, institution, topic or publication year";
    openalex_entity_blueprint.required_capabilities = {"tool.openapi", "workflow.openapi"};
    openalex_entity_blueprint.workflow_bindings = {
        {"workflow://openalex/entity-filtered-work-search", "v1"}};
    assert(store.create(openalex_entity_blueprint, error));
    common_blueprint_candidate openalex_entity_candidate = make_candidate();
    openalex_entity_candidate.logical_id = "openalex-entity-filtered-work-search";
    openalex_entity_candidate.persisted_id = openalex_entity_blueprint.id;
    openalex_entity_candidate.description = openalex_entity_blueprint.goal;
    assert(compile_agent_route_catalog(request, store, scope,
        {openalex_candidate, openalex_entity_candidate}, tooling, "", catalog, error));
    const auto entity_route = std::find_if(catalog.candidates.begin(), catalog.candidates.end(),
        [](const common_agent_route_candidate & candidate) {
            return candidate.blueprint_logical_id == "openalex-entity-filtered-work-search";
        });
    assert(entity_route != catalog.candidates.end() &&
        entity_route->kind == common_agent_route_kind::blueprint_workflow &&
        std::find(entity_route->resolved_tools.begin(), entity_route->resolved_tools.end(),
            "openalex.listAuthors") != entity_route->resolved_tools.end() &&
        std::find(entity_route->resolved_tools.begin(), entity_route->resolved_tools.end(),
            "openalex.listInstitutions") != entity_route->resolved_tools.end() &&
        std::find(entity_route->resolved_tools.begin(), entity_route->resolved_tools.end(),
            "openalex.listTopics") != entity_route->resolved_tools.end());
    tooling.available_context.clear();
    assert(compile_agent_route_catalog(request, store, scope,
        {openalex_candidate, openalex_entity_candidate},
        tooling, "", catalog, error));
    assert(catalog.candidates.size() == 1);

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

    // The generic graph runner is not OpenAPI-specific: it takes the next
    // host-declared path, lets the model choose only among observed IDs, and
    // binds that ID into the following host-approved tool call.
    common_plan_workflow_definition graph_definition;
    graph_definition.workflow_ref = "workflow://research/entity-lookup";
    graph_definition.workflow_revision = "v1";
    graph_definition.family = "research";
    graph_definition.graph_revision = "entity-lookup-graph@v1";
    graph_definition.required_capabilities = {"catalog.read"};
    graph_definition.start_state = "resolve";
    graph_definition.terminal_states = {"complete"};
    graph_definition.transitions = {
        {"resolve-author", "resolve", "candidates", common_plan_workflow_transition_kind::tool,
            "catalog.listAuthors", "{}", {}, 1.0f},
        {"choose-author", "candidates", "selected", common_plan_workflow_transition_kind::model_choice,
            {}, "{}", "/results", 1.0f},
        {"retrieve-works", "selected", "complete", common_plan_workflow_transition_kind::tool,
            "catalog.listWorks", R"({"filter":"author.id:$choice","source":{"$from_step":"$previous"}})", {}, 1.0f},
    };
    common_agent_route_candidate graph_route;
    graph_route.id = "route:entity-lookup";
    graph_route.workflow_definition = graph_definition;
    graph_route.resolved_tools = {"catalog.listAuthors", "catalog.listWorks"};
    common_plan_state graph_plan;
    graph_plan.id = "entity-lookup-plan";
    graph_plan.route_binding = common_plan_route_binding{graph_route.id, "entity-lookup", "bp-v1",
        graph_definition.workflow_ref, graph_definition.workflow_revision, graph_definition.graph_revision,
        "env-v1", "policy-v1"};
    graph_plan.workflow_definition = graph_definition;
    common_plan_step resolved_step{"resolve", "Resolve author", "Find candidate authors"};
    resolved_step.status = common_plan_step_status::completed;
    resolved_step.semantic_alias = "workflow-transition:resolve-author";
    graph_plan.steps = {resolved_step};
    common_plan_observation candidates;
    candidates.id = "author-candidates";
    candidates.source = "catalog.listAuthors";
    candidates.summary = R"({"results":[{"id":"A1","display_name":"Ada Lovelace"}]})";
    auto graph_continuation = make_agent_workflow_graph_continuation(graph_route,
        [](const common_plan_observation &, const common_plan_workflow_transition &, std::string & id, std::string & selection_error) {
            id = "A1"; selection_error.clear(); return true;
        });
    continuation_steps.clear();
    assert(graph_continuation(graph_plan, "resolve", candidates, continuation_steps, error));
    assert(error.empty() && continuation_steps.size() == 1);
    assert(continuation_steps.front().tool_call &&
        continuation_steps.front().tool_call->name == "catalog.listWorks" &&
        continuation_steps.front().semantic_alias &&
        *continuation_steps.front().semantic_alias == "workflow-transition:choose-author=A1,retrieve-works");
    const auto graph_arguments = nlohmann::json::parse(continuation_steps.front().tool_call->arguments_json);
    assert(graph_arguments["filter"] == "author.id:A1");
    assert(graph_arguments["source"].value("$from_step", "") == "resolve");
    auto rejected_graph_continuation = make_agent_workflow_graph_continuation(graph_route,
        [](const common_plan_observation &, const common_plan_workflow_transition &, std::string & id, std::string & selection_error) {
            id = "invented-author"; selection_error.clear(); return true;
        });
    continuation_steps.clear(); error.clear();
    assert(!rejected_graph_continuation(graph_plan, "resolve", candidates, continuation_steps, error));
    assert(error.find("outside the host observation") != std::string::npos);

    common_plan_workflow_definition table_choice_definition;
    table_choice_definition.workflow_ref = "workflow://eurostat/catalogue-choice";
    table_choice_definition.workflow_revision = "v1";
    table_choice_definition.family = "dataset";
    table_choice_definition.graph_revision = "catalogue-choice@v1";
    table_choice_definition.start_state = "query_pending";
    table_choice_definition.terminal_states = {"complete"};
    table_choice_definition.transitions = {
        {"search", "query_pending", "candidates", common_plan_workflow_transition_kind::tool,
            "data.query", "{}", {}, 1.0f},
        {"choose-dataflow", "candidates", "selected", common_plan_workflow_transition_kind::model_choice,
            {}, "{}", "/result/rows", 1.0f},
        {"get-structure", "selected", "complete", common_plan_workflow_transition_kind::tool,
            "eurostat_sdmx.getStructure", "{}", {}, 1.0f},
    };
    table_choice_definition.transitions[1].candidate_id_json_pointer = "/code";
    common_agent_route_candidate table_choice_route;
    table_choice_route.id = "route:eurostat-table-choice";
    table_choice_route.workflow_definition = table_choice_definition;
    table_choice_route.resolved_tools = {"data.query", "eurostat_sdmx.getStructure"};
    common_plan_state table_choice_plan;
    table_choice_plan.id = "eurostat-table-choice-plan";
    table_choice_plan.route_binding = common_plan_route_binding{table_choice_route.id,
        "eurostat-table-choice", "bp-v1", table_choice_definition.workflow_ref,
        table_choice_definition.workflow_revision, table_choice_definition.graph_revision,
        "env-v1", "policy-v1"};
    table_choice_plan.workflow_definition = table_choice_definition;
    common_plan_step searched_step{"search-step", "Search", "Find candidates"};
    searched_step.status = common_plan_step_status::completed;
    searched_step.semantic_alias = "workflow-transition:search";
    table_choice_plan.steps = {searched_step};
    common_plan_observation table_candidates;
    table_candidates.id = "eurostat-query-results";
    table_candidates.source = "data.query";
    table_candidates.summary = R"({"result":{"columns":["code","title"],"rows":[["TPS00003","Population density"],["demo","Demo"]]}})";
    auto table_choice_continuation = make_agent_workflow_graph_continuation(table_choice_route,
        [](const common_plan_observation &, const common_plan_workflow_transition &,
                std::string & id, std::string & selection_error) {
            id = "TPS00003"; selection_error.clear(); return true;
        });
    continuation_steps.clear(); error.clear();
    assert(table_choice_continuation(table_choice_plan, "search-step", table_candidates,
        continuation_steps, error));
    assert(error.empty() && continuation_steps.size() == 1 && continuation_steps.front().tool_call &&
        continuation_steps.front().tool_call->name == "eurostat_sdmx.getStructure");
    assert(continuation_steps.front().semantic_alias &&
        *continuation_steps.front().semantic_alias == "workflow-transition:choose-dataflow=TPS00003,get-structure");

    // Candidate-backed model choices must survive state replay. Once the
    // selected tool is complete, replay reaches the terminal state instead
    // of asking for the candidate list from the tool's unrelated result.
    auto selected_structure = continuation_steps.front();
    selected_structure.status = common_plan_step_status::completed;
    table_choice_plan.steps.push_back(selected_structure);
    continuation_steps.clear(); error.clear();
    assert(table_choice_continuation(table_choice_plan, selected_structure.id,
        table_candidates, continuation_steps, error));
    assert(error.empty() && continuation_steps.empty());

    // A verified model choice remains bindable after an intervening tool
    // transition; the choice must come from its host-owned observation, not
    // from a transient `$choice` substitution.
    common_plan_workflow_definition delayed_choice_definition;
    delayed_choice_definition.workflow_ref = "workflow://eurostat/metadata-resolution";
    delayed_choice_definition.workflow_revision = "v1";
    delayed_choice_definition.family = "openapi";
    delayed_choice_definition.graph_revision = "eurostat-metadata@v1";
    delayed_choice_definition.start_state = "catalogue_searched";
    delayed_choice_definition.terminal_states = {"constraints_resolved"};
    delayed_choice_definition.transitions = {
        {"choose-dataflow", "catalogue_searched", "dataflow_code_selected",
            common_plan_workflow_transition_kind::model_choice, {}, "{}", {}, 1.0f},
        {"get-dataflow-structure", "dataflow_code_selected", "dataflow_structure_resolved",
            common_plan_workflow_transition_kind::tool, "eurostat_sdmx.getStructure",
            R"({"artifactType":"dataflow","version":"1.0"})", {}, 1.0f},
        {"get-content-constraint", "dataflow_structure_resolved", "constraints_resolved",
            common_plan_workflow_transition_kind::tool, "eurostat_sdmx.getStructure",
            R"({"artifactType":"contentconstraint","version":"1.0"})", {}, 1.0f},
    };
    delayed_choice_definition.transitions[2].argument_bindings = {
        {"/resourceID", "choose-dataflow", "/choice", true},
    };
    common_agent_route_candidate delayed_choice_route;
    delayed_choice_route.id = "route:eurostat-delayed-choice";
    delayed_choice_route.workflow_definition = delayed_choice_definition;
    delayed_choice_route.resolved_tools = {"eurostat_sdmx.getStructure"};
    common_plan_state delayed_choice_plan;
    delayed_choice_plan.id = "eurostat-delayed-choice-plan";
    delayed_choice_plan.route_binding = common_plan_route_binding{delayed_choice_route.id,
        "eurostat-metadata", "bp-v1", delayed_choice_definition.workflow_ref,
        delayed_choice_definition.workflow_revision, delayed_choice_definition.graph_revision,
        "env-v1", "policy-v1"};
    delayed_choice_plan.workflow_definition = delayed_choice_definition;
    common_plan_step structure_step{"get-structure", "Get dataflow", "Resolve DSD"};
    structure_step.status = common_plan_step_status::completed;
    structure_step.semantic_alias = "workflow-transition:choose-dataflow=TPS00003,get-dataflow-structure";
    delayed_choice_plan.steps.push_back(structure_step);
    common_plan_observation selected_dataflow;
    selected_dataflow.id = "workflow-choice:choose-dataflow";
    selected_dataflow.source = "workflow_choice";
    selected_dataflow.confidence = 1.0f;
    selected_dataflow.summary = R"({"transition_id":"choose-dataflow","choice":"TPS00003","verified_against_host_candidates":true})";
    delayed_choice_plan.observations.push_back(selected_dataflow);
    common_plan_observation structure_result;
    structure_result.id = "tool:get-structure:1";
    structure_result.source = "eurostat_sdmx.getStructure";
    structure_result.summary = R"({"ok":true,"result":{"id":"TPS00003"}})";
    auto delayed_choice_continuation = make_agent_workflow_graph_continuation(delayed_choice_route, {});
    continuation_steps.clear(); error.clear();
    assert(delayed_choice_continuation(delayed_choice_plan, "get-structure", structure_result,
        continuation_steps, error));
    assert(error.empty() && continuation_steps.size() == 1 && continuation_steps.front().tool_call);
    const auto constraint_call = nlohmann::json::parse(continuation_steps.front().tool_call->arguments_json);
    assert(constraint_call["artifactType"] == "contentconstraint");
    assert(constraint_call["resourceID"] == "TPS00003");

    // Fixed host choices can depend on a structured observation, and a
    // selected tool transition binds only the declared scalar result field.
    common_plan_workflow_definition evidence_definition;
    evidence_definition.workflow_ref = "workflow://openalex/full-text-evidence";
    evidence_definition.workflow_revision = "v1";
    evidence_definition.family = "openapi";
    evidence_definition.graph_revision = "openalex-full-text@v1";
    evidence_definition.start_state = "identifier_pending";
    evidence_definition.terminal_states = {"abstract_complete", "full_text_fetched"};
    evidence_definition.transitions = {
        {"get-work", "identifier_pending", "work_retrieved", common_plan_workflow_transition_kind::tool,
            "openalex.getWork", "{}", {}, 1.0f},
        {"choose-depth", "work_retrieved", "full_text_pending", common_plan_workflow_transition_kind::model_choice,
            {}, "{}", {}, 1.0f},
        {"fetch-full-text", "full_text_pending", "full_text_fetched", common_plan_workflow_transition_kind::tool,
            "web.fetch", "{}", {}, 1.0f},
    };
    evidence_definition.transitions[1].choices = {
        {"answer_from_abstract", "Abstract is sufficient", "abstract_complete", {}},
        {"fetch_full_text", "Need full text", "full_text_pending", {"/result/best_oa_location/pdf_url"}},
    };
    evidence_definition.transitions[2].argument_bindings = {
        {"/url", "get-work", "/result/best_oa_location/pdf_url", true},
    };
    common_agent_route_candidate evidence_route;
    evidence_route.id = "route:openalex-full-text";
    evidence_route.workflow_definition = evidence_definition;
    evidence_route.resolved_tools = {"openalex.getWork", "web.fetch"};
    common_plan_state evidence_plan;
    evidence_plan.id = "openalex-evidence-plan";
    evidence_plan.route_binding = common_plan_route_binding{evidence_route.id, "openalex-full-text", "bp-v1",
        evidence_definition.workflow_ref, evidence_definition.workflow_revision,
        evidence_definition.graph_revision, "env-v1", "policy-v1"};
    evidence_plan.workflow_definition = evidence_definition;
    common_plan_step get_work_step{"get-work-step", "Get work", "Retrieve OpenAlex work"};
    get_work_step.status = common_plan_step_status::completed;
    get_work_step.semantic_alias = "workflow-transition:get-work";
    evidence_plan.steps.push_back(get_work_step);
    common_plan_observation openalex_observation;
    openalex_observation.id = "tool:get-work-step:1";
    openalex_observation.source = "openalex.getWork";
    openalex_observation.summary = R"({"ok":true,"result":{"best_oa_location":{"pdf_url":"https://example.org/paper.pdf"},"abstract_text":"Short abstract"}})";
    evidence_plan.observations.push_back(openalex_observation);
    auto evidence_continuation = make_agent_workflow_graph_continuation(evidence_route,
        [](const common_plan_observation &, const common_plan_workflow_transition &, std::string & id, std::string & selection_error) {
            id = "fetch_full_text"; selection_error.clear(); return true;
        });
    continuation_steps.clear(); error.clear();
    assert(evidence_continuation(evidence_plan, "get-work-step", openalex_observation, continuation_steps, error));
    assert(error.empty() && continuation_steps.size() == 1);
    assert(continuation_steps.front().tool_call && continuation_steps.front().tool_call->name == "web.fetch");
    const auto fetch_arguments = nlohmann::json::parse(continuation_steps.front().tool_call->arguments_json);
    assert(fetch_arguments["url"] == "https://example.org/paper.pdf");
    assert(continuation_steps.front().semantic_alias &&
        *continuation_steps.front().semantic_alias == "workflow-transition:choose-depth=fetch_full_text,fetch-full-text");

    auto missing_url_observation = openalex_observation;
    missing_url_observation.summary = R"({"ok":true,"result":{"abstract_text":"Short abstract"}})";
    continuation_steps.clear(); error.clear();
    assert(!evidence_continuation(evidence_plan, "get-work-step", missing_url_observation, continuation_steps, error));
    assert(error.find("outside the host observation") != std::string::npos);
    // Import the shipped package, then exercise real route applicability.
    // Parsing a workflow alone cannot detect mismatched workflow families.
    common_agent_bootstrap_package package;
    std::ifstream package_file(std::string(LLAMA_AGENT_SOURCE_DIR) +
        "/docs/examples/agent-bootstrap-workflows-v1.json");
    assert(package_file);
    std::stringstream package_text;
    package_text << package_file.rdbuf();
    assert(common_agent_package_parse_json(package_text.str(), package, error));
    common_memory_in_memory_store package_memory;
    common_plan_in_memory_store package_plans;
    assert(package_memory.open("", error) && package_plans.open("", error));
    common_agent_bootstrap_config bootstrap;
    bootstrap.namespace_id = "local";
    bootstrap.session_id = "s";
    common_agent_bootstrap_result installed;
    assert(common_agent_install_bootstrap_package(package_memory, package_plans, bootstrap, package,
        [](const std::string &, std::vector<float> & embedding, std::string & e) {
            embedding = {1.0f, 0.0f}; e.clear(); return true;
        }, installed, error));
    std::vector<common_blueprint_candidate> eurostat_candidates;
    for (const auto & id : installed.installed_blueprint_ids) {
        if (id.find("eurostat-") == std::string::npos) continue;
        common_blueprint_candidate candidate;
        candidate.logical_id = id;
        candidate.persisted_id = id;
        eurostat_candidates.push_back(std::move(candidate));
    }
    assert(eurostat_candidates.size() == 3);
    common_agent_runtime_tooling eurostat_tools;
    eurostat_tools.profile_tools_active = true;
    // A configured profile may expose concrete tools without repeating their
    // stable semantic capability.  The route compiler should derive tool.web
    // from web.fetch/web.search just as it derives tool.dataset above.
    eurostat_tools.capabilities = {"web.fetch", "dataset.inspect", "dataset.schema", "dataset.sample", "data.query", "openapi.read"};
    eurostat_tools.available_context = {"context.network.allowed", "context.openapi.available"};
    eurostat_tools.capability_tools = {
        {"web.fetch", {"web.fetch"}}, {"dataset.inspect", {"dataset.inspect"}},
        {"dataset.schema", {"dataset.schema"}},
        {"dataset.sample", {"dataset.sample"}}, {"data.query", {"data.query"}},
        {"openapi.read", {"eurostat.getData", "eurostat_sdmx.getStructure", "openalex.listWorks"}}};
    for (const auto & name : {"web.fetch", "dataset.inspect", "dataset.schema", "dataset.sample", "data.query",
            "eurostat.getData", "eurostat_sdmx.getStructure", "openalex.listWorks"}) {
        eurostat_tools.tools.push_back({name, name, R"({"type":"object"})"});
    }
    assert(compile_agent_route_catalog(request, package_plans, scope, eurostat_candidates,
        eurostat_tools, "", catalog, error));
    assert(catalog.candidates.size() == 4); // Normal plus all three Eurostat routes.
    for (const auto & candidate : catalog.candidates) {
        if (!candidate.workflow_definition) continue;
        assert(candidate.kind == common_agent_route_kind::blueprint_workflow);
        const auto & workflow_ref = candidate.workflow->workflow_ref;
        const auto expected_revision = workflow_ref == "workflow://eurostat/catalog-discovery" ? "v5" :
            workflow_ref == "workflow://eurostat/metadata-resolution" ? "v7" :
            workflow_ref == "workflow://eurostat/data-retrieval" ? "v7" : "v4";
        assert(candidate.workflow_revision == expected_revision);
        assert(std::find(candidate.resolved_tools.begin(), candidate.resolved_tools.end(),
            "openalex.listWorks") == candidate.resolved_tools.end());
    }
    eurostat_tools.tools.erase(std::remove_if(eurostat_tools.tools.begin(), eurostat_tools.tools.end(),
        [](const auto & tool) { return tool.name == "eurostat.getData" || tool.name == "eurostat_sdmx.getStructure"; }),
        eurostat_tools.tools.end());
    assert(compile_agent_route_catalog(request, package_plans, scope, eurostat_candidates,
        eurostat_tools, "", catalog, error));
    assert(catalog.candidates.size() == 2); // OpenAlex cannot substitute for Eurostat.
    return 0;
}
