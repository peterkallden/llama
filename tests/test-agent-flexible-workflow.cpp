#include "tools/agent/cli/agent-cli-selection.h"
#include "tools/agent/runtime/agent-runtime-assembly.h"
#include "tools/agent/runtime/agent-workflow-transition.h"
#include "tools/agent/tooling/agent-selection-contracts.h"
#include "agent/agent-package-json.h"
#include "agent/input-resources.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/catalog/model-projection.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "plan/plan-blueprint.h"
#include "plan/plan-workflow-binding.h"
#include "plan/plan-workflow-state.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using json = nlohmann::ordered_json;
namespace {
void require(bool value, const std::string & message) {
    if (!value) { std::cerr << message << '\n'; std::exit(1); }
}
struct scripted_inference final : common_agent_inference {
    std::vector<std::string> replies;
    std::vector<common_agent_generation_request> requests;
    bool generate(const common_agent_generation_request & request, common_agent_generation_result & result) override {
        require(requests.size() < replies.size(), "unexpected inference call");
        result.content = replies[requests.size()]; requests.push_back(request);
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::eos;
        return true;
    }
};

void selection_formats() {
    common_agent_request request;
    require(request.tool_output_format == common_agent_tool_output_format::compact_dsl, "DSL default changed");
    common_agent_generation_config config;
    config.n_predict = 128;
    common_agent_route_catalog catalog;
    common_agent_route_candidate normal; normal.id = "normal-plan";
    auto workflow = normal; workflow.id = "blueprint:investigation:workflow:workflow://investigation/identity-analysis@v5";
    catalog.candidates = {normal, workflow};
    scripted_inference inference;
    inference.replies = {"open! selection.route\nroute_id: normal-plan\nconfidence: 0.9"};
    std::string error;
    const auto result = select_llama_cli_route(inference, config, request, catalog, error);
    require(error.empty() && result.route_id == normal.id, "DSL route selection failed: " + error);
    require(inference.requests.front().json_schema.empty() && !inference.requests.front().grammar.empty(),
        "DSL selector exposed native JSON instead of grammar");
    request.tool_output_format = common_agent_tool_output_format::native;
    scripted_inference native;
    native.replies = {R"({"route_id":"normal-plan","confidence":0.9})"};
    require(select_llama_cli_route(native, config, request, catalog, error).route_id == normal.id && error.empty(),
        "explicit JSON selection was lost");
    require(!native.requests.front().json_schema.empty() && native.requests.front().grammar.empty(),
        "JSON selector received DSL grammar");

    request.tool_output_format = common_agent_tool_output_format::compact_dsl;
    scripted_inference invalid;
    invalid.replies = {R"({"route_id":"normal-plan","confidence":0.9})"};
    require(!select_llama_cli_route(invalid, config, request, catalog, error).route_id && !error.empty(),
        "DSL silently accepted JSON fallback");

    const auto schema = make_agent_plan_selection_schema_json({"work-plan"});
    agent_selection_presentation presentation;
    require(make_agent_selection_presentation(request.tool_output_format, "selection.plan", schema, presentation, error), error);
    json value;
    require(parse_agent_selection_response(request.tool_output_format, "selection.plan", schema,
        "open! selection.plan\ndecision: resume\nplan_id: work-plan\nconfidence: 0.9", value, error), error);
    require(value["plan_id"] == "work-plan", "plan selection lost its identity");
    require(!parse_agent_selection_response(request.tool_output_format, "selection.plan", schema,
        "open! selection.plan\ndecision: resume\nplan_id: invented\nconfidence: 0.9", value, error),
        "selection accepted an invented plan");

    common_plan_workflow_transition transition;
    transition.choices.push_back({"inspect", "Inspect empty or nonempty observed records", "inspect", {"/count", "/items", "/complete"}});
    common_plan_observation observation{"choice-source", "investigation.events", R"({"count":0,"items":[],"complete":false})", 1.0f};
    require(common_plan_workflow_choice_available(observation, transition.choices.front()),
        "zero/false/empty observations were confused with missing fields");
    scripted_inference choice;
    choice.replies = {"open! selection.candidate\nid: inspect"};
    std::string selected;
    require(select_llama_cli_workflow_candidate(choice, config, request, observation, transition, selected, error), error);
    require(selected == "inspect" && choice.requests.front().json_schema.empty(), "workflow choice escaped DSL");
    observation.summary = R"({"ok":false,"count":0,"items":[],"complete":false})";
    require(!common_plan_workflow_choice_available(observation, transition.choices.front()),
        "failed observations enabled a choice");
}

void workflow_action_dataset_handles() {
    common_agent_request request;
    request.prompt = "Describe only the amount column.";
    common_agent_dataset_descriptor customers;
    customers.ref.uri = "dataset://seed/customers";
    customers.ref.name = "customers";
    customers.columns = {{"customer_id"}, {"segment"}};
    customers.ref.column_count = customers.columns.size();
    common_agent_dataset_descriptor orders;
    orders.ref.uri = "dataset://seed/orders";
    orders.ref.name = "orders";
    orders.columns = {{"order_id"}, {"customer_id"}, {"amount"}};
    orders.ref.column_count = orders.columns.size();
    request.available_datasets = {customers, orders};

    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    std::string error;
    require(catalog.bootstrap("analysis", bootstrap, error), error);
    const auto * definition = catalog.find_definition("statistics.describe");
    require(definition != nullptr, "registered statistics.describe contract is missing");
    const common_chat_tool tool{definition->name, definition->description, common_tool_model_input_schema(*definition)};
    common_agent_generation_config config;
    config.n_predict = 128;
    scripted_inference inference;
    inference.replies = {"open! statistics.describe\ndataset: d2\ncolumns: amount"};
    const auto selected = select_llama_cli_workflow_action(inference, config, request,
        {tool}, {}, {}, common_agent_tool_output_format::compact_dsl, error);
    require(error.empty() && selected.action.has_value(), "dataset handle action selection failed: " + error);
    const auto arguments = json::parse(selected.action->arguments_json, nullptr, false);
    require(arguments.is_object() && arguments.value("dataset", std::string()) == orders.ref.uri &&
        arguments.value("columns", json::array()) == json::array({"amount"}),
        "scoped dataset handle did not resolve to the selected dataset");
    require(inference.requests.size() == 1 && inference.requests.front().json_schema.empty() &&
        inference.requests.front().grammar.find("d1") != std::string::npos &&
        inference.requests.front().grammar.find("d2") != std::string::npos &&
        inference.requests.front().grammar.find("null_count") == std::string::npos,
        "DSL action grammar did not constrain dataset selection to current host handles");

    scripted_inference native;
    native.replies = {R"({"tool":"statistics.describe","args":{"dataset":"d2","columns":["amount"]}})"};
    const auto native_action = select_llama_cli_workflow_action(native, config, request,
        {tool}, {}, {}, common_agent_tool_output_format::native, error);
    require(error.empty() && native_action.action && !native.requests.front().json_schema.empty() &&
        native.requests.front().grammar.empty() && json::parse(native_action.action->arguments_json)["dataset"] == orders.ref.uri,
        "explicit JSON mode did not preserve the same host dataset identity");

    scripted_inference repair;
    repair.replies = {
        "open! statistics.describe\ndataset: d2\ncolumns: amount\ngroup_by: null_count",
        "open! statistics.describe\ndataset: d2\ncolumns: amount",
    };
    const auto repaired = select_llama_cli_workflow_action(repair, config, request,
        {tool}, {}, {}, common_agent_tool_output_format::compact_dsl, error);
    require(error.empty() && repaired.action.has_value() && repair.requests.size() == 2,
        "workflow action did not recover after a rejected optional field: " + error);
    require(repair.requests[1].json_schema.empty() &&
        repair.requests[1].messages.back().content.find("dataset: d2") != std::string::npos &&
        repair.requests[1].messages.back().content.find("not partially approved") != std::string::npos &&
        repair.requests[0].messages[0].content.find("$orders.dataset") == std::string::npos &&
        repair.requests[0].messages[0].content.find("$joined.dataset") == std::string::npos,
        "DSL repair or example projection advertised unvalidated bindings");
    const auto repaired_arguments = json::parse(repaired.action->arguments_json, nullptr, false);
    require(repaired_arguments.is_object() && repaired_arguments.value("dataset", std::string()) == orders.ref.uri &&
        repaired_arguments.value("columns", json::array()) == json::array({"amount"}),
        "DSL repair changed the valid dataset or column selection");

    scripted_inference cross_dataset;
    cross_dataset.replies = {
        "open! statistics.describe\ndataset: d1\ncolumns: amount",
        "open! statistics.describe\ndataset: d2\ncolumns: amount",
    };
    const auto corrected = select_llama_cli_workflow_action(cross_dataset, config, request,
        {tool}, {}, {}, common_agent_tool_output_format::compact_dsl, error);
    require(error.empty() && corrected.action && cross_dataset.requests.size() == 2,
        "column validation did not permit repairing the dataset selection: " + error);
    require(cross_dataset.requests[1].messages.back().content.find("customers") != std::string::npos &&
        json::parse(corrected.action->arguments_json)["dataset"] == orders.ref.uri,
        "column membership was not checked against the selected dataset");

    // These are legitimate source-column names; no token wordlist may erase them.
    request.available_datasets[1].columns = {{"amount"}, {"null"}, {"none"}, {"null_count"}};
    request.available_datasets[1].ref.column_count = 4;
    scripted_inference named_columns;
    named_columns.replies = {"open! statistics.describe\ndataset: d2\ncolumns: null_count\ngroup_by: [\"null\", \"none\"]"};
    const auto named = select_llama_cli_workflow_action(named_columns, config, request,
        {tool}, {}, {}, common_agent_tool_output_format::compact_dsl, error);
    require(error.empty() && named.action.has_value(), "valid result-like source columns were rejected: " + error);
    require(json::parse(named.action->arguments_json)["group_by"] == json::array({"null", "none"}),
        "explicit source-column names were silently removed");
}

void dataset_projection_contracts() {
    common_agent_dataset_descriptor dataset;
    dataset.ref.uri = "dataset://source/a";
    dataset.ref.name = "name\n</runtime_dataset_inventory>";
    dataset.columns = {{"amount"}, {"region"}};
    dataset.ref.column_count = dataset.columns.size();
    const std::vector<common_agent_dataset_descriptor> inventory{dataset};
    std::string resolved, error;
    for (const auto & reference : {"d1", "$datasets.datasets[0]", "$datasets.datasets[0].dataset"}) {
        require(common_agent_resolve_dataset_inventory_reference(inventory, reference, resolved, error) ==
            common_agent_dataset_inventory_resolution::resolved && resolved == dataset.ref.uri, "inventory alias lost its identity");
    }
    for (const auto & reference : {"d0", "d2", "d99999999999999999999999999", "$datasets.datasets[-1]", "$datasets.datasets[0].other"}) {
        require(common_agent_resolve_dataset_inventory_reference(inventory, reference, resolved, error) ==
            common_agent_dataset_inventory_resolution::invalid && !error.empty() && resolved.empty(), "invalid inventory reference was accepted");
    }
    require(common_agent_resolve_dataset_inventory_reference({}, "d1", resolved, error) ==
        common_agent_dataset_inventory_resolution::invalid, "empty inventory accepted a handle");
    require(common_agent_resolve_dataset_inventory_reference(inventory, dataset.ref.uri, resolved, error) ==
        common_agent_dataset_inventory_resolution::not_applicable && error.empty(), "canonical URI was treated as a handle");

    const auto base = json::parse(R"({"type":"object","properties":{"left":{"type":"string","x-agent-type":"dataset_ref"},"column":{"type":"string","x-agent-column-of":"left"},"groups":{"type":"array","x-agent-column-of":"left","items":{"type":"string","enum":["region"]}}}})");
    auto projected = base;
    require(common_agent_project_dataset_model_schema(projected, inventory, json::object(), error), error);
    require(projected["properties"]["left"]["enum"] == json::array({"d1"}) &&
        projected["properties"]["column"]["enum"] == json::array({"amount", "region"}) &&
        projected["properties"]["groups"]["items"]["enum"] == json::array({"region"}),
        "dataset projection broadened a registered constraint or missed a generic reference field");
    auto unknown = dataset;
    unknown.ref.uri = "dataset://source/b";
    unknown.columns.clear();
    projected = base;
    require(common_agent_project_dataset_model_schema(projected, {dataset, unknown}, json::object(), error) &&
        !projected["properties"]["column"].contains("enum"), "unknown schema was treated as complete");
    projected = base;
    require(common_agent_project_dataset_model_schema(projected, {dataset, unknown}, json{{"left", dataset.ref.uri}}, error) &&
        projected["properties"]["column"]["enum"] == json::array({"amount", "region"}),
        "explicit host binding did not narrow the known source schema");
    auto rendered = common_agent_render_dataset_inventory(inventory, 65536, true);
    require(rendered.find("columns=[\"amount\", \"region\"]") != std::string::npos &&
        rendered.find("name=\"name\\n") != std::string::npos &&
        rendered.find("\n</runtime_dataset_inventory>") == rendered.rfind("\n</runtime_dataset_inventory>"),
        "dataset metadata escaped its quoted boundary or omitted schema");
    auto example_schema = base;
    example_schema["x-agent-examples"] = json::array();
    const auto description = common_render_compact_tool_description("statistics.describe", "Describe", example_schema.dump(), "{}", error);
    require(error.empty() && description.find("$joined.dataset") == std::string::npos,
        "explicitly removed examples were replaced with stale hardcoded examples");
}

void transitions_and_resume() {
    common_agent_route_candidate route;
    route.id = "bound-route"; route.resolved_tools = {"inspect", "query"};
    common_plan_workflow_definition definition;
    definition.start_state = "start"; definition.terminal_states = {"done"};
    definition.transitions = {{"inspect", "start", "inspected", common_plan_workflow_transition_kind::tool, "inspect", "{}"},
        {"followup", "inspected", "done", common_plan_workflow_transition_kind::tool, "query", R"({"scope":"fixed"})"}};
    definition.transitions.back().model_supplies_arguments = true;
    route.workflow_definition = definition;
    common_plan_state plan;
    plan.id = "p"; plan.route_binding = common_plan_route_binding{}; plan.route_binding->route_id = route.id;
    plan.workflow_definition = definition;
    common_plan_step inspect{"inspect-step", "Inspect", "Inspect"};
    inspect.status = common_plan_step_status::completed; inspect.tool_call = common_plan_tool_call{"inspect", "{}"};
    inspect.semantic_alias = "workflow-transition:inspect"; plan.steps.push_back(inspect);
    common_plan_observation observation{"tool:inspect-step:inspect", "inspect", R"({"ok":true,"result":{"id":"known"}})", 1.0f};
    plan.observations.push_back(observation);
    size_t calls = 0;
    auto continuation = make_agent_workflow_graph_continuation(route, {},
        [&](const common_plan_state &, const common_plan_observation &, const common_plan_workflow_transition & transition,
                const std::string & seed, std::string & arguments, std::string & error) {
            require(transition.id == "followup" && json::parse(seed)["scope"] == "fixed", "later transition lost its host seed");
            ++calls; arguments = R"({"scope":"fixed","query":"a model follow-up"})"; error.clear(); return true;
        });
    std::vector<common_plan_step> steps;
    std::string error;
    require(continuation(plan, inspect.id, observation, steps, error) && steps.size() == 1 && calls == 1, error);
    require(json::parse(steps[0].tool_call->arguments_json)["query"] == "a model follow-up", "later model arguments were lost");
    steps[0].status = common_plan_step_status::completed; plan.steps.push_back(steps[0]);
    require(common_plan_workflow_is_terminal(plan, definition), "completed transition was not replayed");
    require(continuation(plan, steps[0].id, observation, steps, error) && steps.empty() && calls == 1,
        "resume regenerated completed work");
    plan.steps.pop_back();
    auto missing = make_agent_workflow_graph_continuation(route);
    require(!missing(plan, inspect.id, observation, steps, error), "missing later argument selector was silently ignored");
    auto tampered = make_agent_workflow_graph_continuation(route, {},
        [](const common_plan_state &, const common_plan_observation &, const common_plan_workflow_transition &,
                const std::string &, std::string & arguments, std::string &) {
            arguments = R"({"scope":"other"})"; return true;
        });
    require(!tampered(plan, inspect.id, observation, steps, error), "model changed the host seed");
}

void blueprint_and_publication_context() {
    common_plan_state blueprint;
    blueprint.id = "template"; blueprint.kind = common_plan_kind::blueprint;
    blueprint.goal = "Understand the evidence"; blueprint.success_criteria = "Supported result";
    blueprint.steps = {{"inspect", "Inspect", "Inspect sources"}, {"assess", "Assess", "Assess evidence"}};
    blueprint.steps.back().depends_on = {"inspect"}; blueprint.next_action = "inspect";
    common_plan_state instance; std::string error;
    require(common_plan_instantiate_blueprint(blueprint, "task", "session", instance, error), error);
    require(instance.workflow_bindings.empty() && instance.next_action == "task:inspect" &&
        instance.steps.back().depends_on == std::vector<std::string>{"task:inspect"}, "unbound blueprint lost its plan structure");
    common_plan_workflow_binding binding{"workflow://method", "v5", {{"inspect", {"open"}}, {"assess", {"query"}}}};
    require(common_plan_workflow_binding_from_json(common_plan_workflow_binding_to_json(binding)).phase_bindings == binding.phase_bindings,
        "phase binding codec lost metadata");
    require(!common_plan_workflow_binding_to_json({"workflow://legacy", "v4"}).contains("phase_bindings"),
        "legacy fingerprints changed for an empty extension");

    common_agent_request request; request.investigation_context = common_investigation_context_binding{"case", 2};
    common_plan_state plan; plan.id = "case-plan";
    common_plan_step read{"read", "Open", "Open case"}; read.status = common_plan_step_status::completed;
    read.tool_call = common_plan_tool_call{"investigation.case.open", R"({"case_id":"case"})"}; plan.steps.push_back(read);
    plan.observations.push_back({"tool:read:open", "investigation.case.open", R"({"ok":true,"result":{"case":{"case_id":"case","revision":2}}})", 1.0f});
    auto publish = read; publish.id = "publish"; publish.tool_call = common_plan_tool_call{"investigation.collect", R"({"case_id":"case","expected_revision":2})"};
    plan.steps.push_back(publish);
    plan.observations.push_back({"tool:publish:collect", "investigation.collect", R"({"ok":true,"result":{"case_id":"case","revision":3}})", 1.0f});
    common_chat_tool events{"investigation.events", "Read events", R"({"type":"object","required":["case_id","case_revision"],"properties":{"case_id":{"type":"string"},"case_revision":{"type":"integer"}},"additionalProperties":false})"};
    common_plan_workflow_transition next; next.tool_name = events.name;
    scripted_inference inference; common_agent_generation_config config; std::string args;
    require(select_llama_cli_workflow_transition_arguments(inference, config, request, {events}, plan,
        plan.observations.back(), next, "{}", args, error), error);
    require(json::parse(args)["case_revision"] == 3 && inference.requests.empty(),
        "deterministic continuation failed to use its successful publication baseline");
    request.investigation_context->revision = 3;
    require(select_llama_cli_workflow_transition_arguments(inference, config, request, {events}, plan,
        plan.observations.back(), next, "{}", args, error), "resume failed to replay its own revision chain: " + error);
    auto concurrent = read; concurrent.id = "concurrent-read"; plan.steps.push_back(concurrent);
    plan.observations.push_back({"tool:concurrent-read:open", "investigation.case.open", R"({"ok":true,"result":{"case":{"case_id":"case","revision":4}}})", 1.0f});
    require(!select_llama_cli_workflow_transition_arguments(inference, config, request, {events}, plan,
        plan.observations.back(), next, "{}", args, error), "a read silently rebased the workflow onto an external revision");
}

void package_contracts() {
    std::ifstream input(std::string(LLAMA_AGENT_SOURCE_DIR) + "/docs/examples/agent-investigation-workflows-v1.json");
    std::stringstream text; text << input.rdbuf(); common_agent_bootstrap_package package; std::string error;
    require(common_agent_package_parse_json(text.str(), package, error), error);
    require(package.version == "v5" && package.workflows.size() == 9, "new investigation methods missing");
    for (const auto & workflow : package.workflows) require(!workflow.definition.transitions.empty(), "method still has no executable graph");
    for (const auto & blueprint : package.blueprints) require(!blueprint.workflow_bindings.front().phase_bindings.empty(), "blueprint phases are unbound");
    std::string exported;
    require(common_agent_package_to_json(package, exported, error), error);
    common_agent_bootstrap_package restored;
    require(common_agent_package_parse_json(exported, restored, error), error);
    require(restored.blueprints.front().workflow_bindings.front().phase_bindings == package.blueprints.front().workflow_bindings.front().phase_bindings,
        "package roundtrip dropped phase references");
    auto invalid = json::parse(exported);
    invalid["blueprints"][0]["workflow_bindings"][0]["phase_bindings"]["invented"] = {"open-case"};
    require(!common_agent_package_parse_json(invalid.dump(), restored, error), "invalid phase mapping was imported");
}
}
int main(int argc, char ** argv) {
    require(argc == 2, "choose formats, dataset-handles, continuation, blueprint or package");
    const std::string group = argv[1];
    if (group == "formats") selection_formats();
    else if (group == "dataset-handles") { workflow_action_dataset_handles(); dataset_projection_contracts(); }
    else if (group == "continuation") transitions_and_resume();
    else if (group == "blueprint") blueprint_and_publication_context();
    else if (group == "package") package_contracts();
    else require(false, "unknown contract group");
}
