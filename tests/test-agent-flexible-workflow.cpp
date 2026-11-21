#include "tools/agent/runtime/agent-plan-orchestration.h"
#include "memory/memory-in-memory.h"
#include "tools/agent/runtime/agent-planning-selection.h"
#include "tools/agent/runtime/agent-runtime-assembly.h"
#include "tools/agent/runtime/agent-workflow-transition.h"
#include "tools/agent/tooling/agent-selection-contracts.h"
#include "tools/agent/tooling/agent-tool-provider.h"
#include "agent/agent-package-json.h"
#include "agent/input-resources.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/catalog/model-projection.h"
#include "agent/tooling/schema/tool-schema-compact.h"
#include "agent/runtime-json-contracts.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "plan/plan-blueprint.h"
#include "plan/plan-in-memory.h"
#include "plan/plan-workflow-binding.h"
#include "plan/plan-workflow-state.h"
#ifdef LLAMA_PLAN_USE_SQLITE
#include "plan/sqlite/plan-sqlite.h"
#include <chrono>
#include <filesystem>
#endif

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
    std::vector<common_agent_generation_stop_reason> stops;
    size_t capacity = 0;
    size_t context_capacity_tokens() const override { return capacity; }
    common_agent_generation_status status = common_agent_generation_status::completed;
    common_agent_generation_stop_reason stop = common_agent_generation_stop_reason::eos;
    bool generate(const common_agent_generation_request & request, common_agent_generation_result & result) override {
        require(requests.size() < replies.size(), "unexpected inference call");
        result.content = replies[requests.size()]; requests.push_back(request);
        result.status = status;
        result.stop_reason = stops.empty() ? stop : stops.at(requests.size() - 1);
        return true;
    }
};

struct canonical_resource_view final : agent_tool_view {
    std::vector<common_chat_tool> tools;
    std::string execution_schema;
    std::string denied_uri;
    const std::vector<common_chat_tool> & chat_tools() const override { return tools; }
    bool exposes_tool(const std::string & name) const override { return name == "resource.read" || name == "resource.inspect"; }
    bool is_read_only(const std::string & name) const override { return name != "workspace.update"; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool validate(const agent_tool_call & call, std::string & error) const override {
        const auto args = json::parse(call.arguments_json, nullptr, false);
        if (!exposes_tool(call.name) || !args.is_object() || !args.contains("uri") ||
                !args["uri"].is_string() || args.contains("id")) {
            error = "host resource.read requires a canonical uri";
            return false;
        }
        if (!denied_uri.empty() && args["uri"] == denied_uri) { error = "resource is outside the authorized view"; return false; }
        std::string normalized;
        if (!execution_schema.empty() && !common_schema_normalize_and_validate_object(
                call.arguments_json, execution_schema, normalized, error)) return false;
        error.clear(); return true;
    }
    agent_tool_result call(const agent_tool_call &, std::string &) override { return {}; }
    bool supports_async_call(const std::string &) const override { return false; }
    bool begin_call_async(const agent_tool_call &, agent_tool_pending_call &, std::string &) override { return false; }
    bool poll_call_async(const agent_tool_pending_call &, bool &, agent_tool_result &, std::string &) override { return false; }
    bool cancel_call_async(const agent_tool_pending_call &, std::string &) override { return false; }
};

void resource_argument_contracts() {
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    std::string error;
    require(catalog.bootstrap("minimal", bootstrap, error), error);
    common_agent_generation_config config; config.n_predict = 1024;
    common_agent_request request;
    common_agent_input_resource source;
    source.resource.uri = "agent-resource://report/retained";
    request.input_resources = {source};
    for (const auto * name : {"resource.read", "resource.inspect"}) {
        const auto * definition = catalog.find_definition(name);
        require(definition != nullptr, "resource contract is absent");
        common_chat_tool tool{name, definition->description, common_tool_model_input_schema(*definition)};
        canonical_resource_view view;
        view.tools = {tool}; view.execution_schema = definition->input_schema_json;
        common_plan_workflow_transition transition;
        transition.id = "read-evidence"; transition.tool_name = name; transition.model_supplies_arguments = true;
        scripted_inference missing_view;
        request.tool_output_format = common_agent_tool_output_format::native;
        require(!select_agent_planning_workflow_action(missing_view, config, request, {tool},
            {{name,json{{"uri",source.resource.uri}}.dump(),"host",false}}, {},
            request.tool_output_format, error).action && missing_view.requests.empty(),
            "missing execution view asked the model to repair a host URI");
        for (const auto format : {common_agent_tool_output_format::native, common_agent_tool_output_format::compact_dsl}) {
            request.tool_output_format = format;
            const auto reply = [&](const json & args) {
                if (format == common_agent_tool_output_format::native)
                    return json{{"tool", name}, {"args", args}}.dump();
                std::string text = std::string("open! ") + name;
                for (const auto & field : args.items()) text += "\n" + field.key() + ": " + field.value().dump();
                return text;
            };
            scripted_inference bound;
            bound.replies = {reply(json::object())};
            std::string arguments;
            require(select_agent_planning_workflow_transition_arguments(bound, config, request, {tool}, {}, {},
                transition, json{{"uri", source.resource.uri}}.dump(), arguments, error, &view) &&
                json::parse(arguments)["uri"] == source.resource.uri && !json::parse(arguments).contains("id"),
                "host-bound resource was compared with the model schema: " + error);
            require(bound.requests.size() == 1 && bound.requests.front().messages.front().content.find("id (required") == std::string::npos,
                "host-bound URI still required a model identity");
            scripted_inference unbound;
            unbound.replies = {reply({{"id", "r1"}})};
            require(select_agent_planning_workflow_transition_arguments(unbound, config, request, {tool}, {}, {},
                transition, "{}", arguments, error, &view) && json::parse(arguments)["uri"] == source.resource.uri,
                "model resource handle did not reach the execution contract: " + error);
            // Non-strict JSON providers may echo a hidden field. The DSL
            // parser rejects undeclared fields before normalization instead.
            if (format == common_agent_tool_output_format::native) {
                scripted_inference echo;
                echo.replies = {reply({{"id", "r1"}})};
                require(select_agent_planning_workflow_transition_arguments(echo, config, request, {tool}, {}, {},
                    transition, json{{"uri", source.resource.uri}}.dump(), arguments, error, &view),
                    "equivalent handle echo conflicted with its canonical host binding: " + error);
                const auto schema = json::parse(bound.requests.front().json_schema);
                require(!schema["oneOf"][0]["properties"]["args"]["properties"].contains("id"),
                    "bound resource handle leaked into the model schema");
            }
            scripted_inference conflict;
            conflict.replies = {reply({{"id", "agent-resource://different"}}), reply({{"id", "agent-resource://different"}})};
            require(!select_agent_planning_workflow_transition_arguments(conflict, config, request, {tool}, {}, {},
                transition, json{{"uri", source.resource.uri}}.dump(), arguments, error, &view) && arguments.empty(),
                "model resource alias replaced the host-bound URI");
        }
        request.tool_output_format = common_agent_tool_output_format::native;
        scripted_inference repaired;
        repaired.replies = {json{{"tool",name},{"args",{{"id","r99"}}}}.dump(),
            json{{"tool",name},{"args",{{"id","r1"}}}}.dump()};
        const auto repaired_action = select_agent_planning_workflow_action(repaired, config, request, {tool}, {}, {},
            request.tool_output_format, error, false, &view);
        require(repaired_action.action && repaired.requests.size() == 2 &&
            json::parse(repaired_action.action->arguments_json)["uri"] == source.resource.uri,
            "invalid model resource handle did not receive bounded repair: " + error);
        scripted_inference malformed;
        malformed.replies = {json{{"tool",name},{"args",{{"id","r1"},{"invented",true}}}}.dump(),
            json{{"tool",name},{"args",{{"id","r1"},{"invented",true}}}}.dump()};
        require(!select_agent_planning_workflow_action(malformed, config, request, {tool}, {}, {},
            request.tool_output_format, error, false, &view).action && malformed.requests.size() == 2,
            "execution view bypassed the model schema for unknown fields");
        scripted_inference denied;
        denied.replies = {json{{"tool",name},{"args",{{"id","r1"}}}}.dump(),
            json{{"tool",name},{"args",{{"id","r1"}}}}.dump()};
        view.denied_uri = source.resource.uri;
        require(!select_agent_planning_workflow_action(denied, config, request, {tool}, {}, {},
            request.tool_output_format, error, false, &view).action,
            "canonical resource normalization bypassed execution-view rejection");
    }
    // The ordinary runtime and workflow share conflict checks before defaults.
    request.tool_argument_bindings = {{"resource.read", json{{"uri",source.resource.uri}}.dump(), "host", false}};
    json normalized;
    bool changed = false;
    require(common_agent_runtime_apply_safe_tool_defaults_to_json(request, "resource.read", {{"id","r1"}},
        normalized, changed, error) && normalized["uri"] == source.resource.uri && !normalized.contains("id"), error);
    require(!common_agent_runtime_apply_safe_tool_defaults_to_json(request, "resource.read",
        {{"id","agent-resource://different"}}, normalized, changed, error), "ordinary normalization overwrote a host binding");
    require(!common_agent_runtime_apply_host_tool_arguments_to_json(request, "resource.read",
        {{"uri",source.resource.uri},{"id","agent-resource://different"}}, normalized, changed, error),
        "conflicting resource aliases were silently collapsed");
}

void resumed_resource_review_contract() {
#ifdef LLAMA_PLAN_USE_SQLITE
    const std::string uri = "agent-resource://report/persisted";
    common_plan_state saved;
    saved.id = "resource-review-resume"; saved.status = common_plan_status::active;
    common_plan_workflow_definition graph;
    graph.start_state = "start"; graph.terminal_states = {"done"};
    common_plan_workflow_transition initial;
    initial.id = "read-report"; initial.from_state = "start"; initial.to_state = "done";
    initial.tool_name = "resource.read";
    common_plan_workflow_transition review;
    review.id = "review"; review.from_state = "done"; review.to_state = "inspect";
    review.kind = common_plan_workflow_transition_kind::model_choice; review.review_only = true;
    review.choices.push_back({"read", "Read missing report evidence", "inspect", {}});
    auto read = initial;
    read.id = "review-read-evidence"; read.from_state = "inspect"; read.model_supplies_arguments = true;
    read.argument_bindings.push_back({"/uri", "read-report", "/result/resource/uri", true});
    graph.transitions = {initial, review, read}; saved.workflow_definition = graph;
    saved.route_binding = common_plan_route_binding{}; saved.route_binding->route_id = "resource-review-route";
    common_plan_step completed{"report", "Read report", "Read retained report"};
    completed.status = common_plan_step_status::completed;
    completed.tool_call = common_plan_tool_call{"resource.read", json{{"uri",uri}}.dump()};
    completed.semantic_alias = "workflow-transition:read-report";
    saved.steps.push_back(completed);
    saved.observations.push_back({"tool:report:resource.read", "resource.read",
        json{{"ok",true},{"result",{{"resource",{{"uri",uri}}},{"content","retained text"}}}}.dump(), 1.0f});
    const auto directory = std::filesystem::temp_directory_path() /
        ("llama-resource-review-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    std::string error;
    common_plan_sqlite_store store;
    require(store.open((directory / "plans.sqlite").string(), error) && store.create(saved, error), error);
    store.close();
    require(store.open((directory / "plans.sqlite").string(), error), error);
    const auto restored = store.get(saved.id, error);
    require(restored && restored->workflow_definition, "saved review workflow was not restored: " + error);
    common_agent_route_candidate route;
    route.id = saved.route_binding->route_id; route.resolved_tools = {"resource.read"};
    route.workflow_definition = restored->workflow_definition;
    common_agent_request request; request.tool_output_format = common_agent_tool_output_format::native;
    common_agent_generation_config config;
    common_chat_tool tool{"resource.read", "Read resource",
        R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"},"offset":{"type":"integer","minimum":0}}})"};
    canonical_resource_view view;
    scripted_inference inference;
    inference.replies = {R"({"tool":"resource.read","args":{"offset":128}})"};
    auto continuation = make_agent_workflow_graph_continuation(route,
        [](const auto &, const auto &, std::string & choice, std::string & failure) {
            choice = "read"; failure.clear(); return true;
        }, [&](const auto & plan, const auto & observation, const auto & transition,
                const std::string & seed, std::string & arguments, std::string & failure) {
            return select_agent_planning_workflow_transition_arguments(inference, config, request, {tool},
                plan, observation, transition, seed, arguments, failure, &view);
        });
    common_plan_observation trigger{"review", common_agent_workflow_review_source,
        R"({"review_request":{"missing":"report details"}})", 1.0f};
    common_agent_workflow_continuation next;
    require(continuation(*restored, completed.id, trigger, next, error) && next.size() == 1,
        "reflection could not bind the saved report after store reopen: " + error);
    const auto arguments = json::parse(next.front().tool_call->arguments_json);
    require(arguments["uri"] == uri && arguments["offset"] == 128 && !arguments.contains("id") &&
        inference.requests.size() == 1, "resumed review changed the retained resource or lost the model's bounded position");
    store.close();
    std::filesystem::remove_all(directory);
#endif
}

void selection_formats() {
    common_agent_request request;
    require(request.tool_output_format == common_agent_tool_output_format::compact_dsl, "DSL default changed");
    common_agent_generation_config config;
    config.n_predict = 128;
    common_agent_route_catalog catalog;
    common_agent_route_candidate normal; normal.id = "normal-plan";
    auto workflow = normal; workflow.id = "blueprint:investigation:workflow:workflow://investigation/identity-analysis@v5";
    workflow.blueprint_definition.emplace();
    workflow.blueprint_definition->goal = "Verify an already identified record";
    workflow.blueprint_definition->success_criteria = "Record evidence is verified";
    common_plan_step entry;
    entry.objective = "Use an observed record identity";
    workflow.blueprint_definition->steps.push_back(entry);
    catalog.candidates = {normal, workflow};
    scripted_inference inference;
    inference.replies = {"open! selection.route\nroute_id: normal-plan\nconfidence: 0.9"};
    std::string error;
    const auto result = select_agent_planning_route(inference, config, request, catalog, error);
    require(error.empty() && result.route_id == normal.id, "DSL route selection failed: " + error);
    require(inference.requests.front().json_schema.empty() && !inference.requests.front().grammar.empty(),
        "DSL selector exposed native JSON instead of grammar");
    std::string route_prompt;
    for (const auto & message : inference.requests.front().messages) route_prompt += message.content;
    require(route_prompt.find("Use an observed record identity") != std::string::npos,
        "route selection omitted the workflow entry objective");
    require(route_prompt.find(workflow.blueprint_definition->goal) != std::string::npos &&
        route_prompt.find(workflow.blueprint_definition->success_criteria) != std::string::npos,
        "route selection omitted the declared outcome and completion criteria");
    request.tool_output_format = common_agent_tool_output_format::native;
    scripted_inference native;
    native.replies = {R"({"route_id":"normal-plan","confidence":0.9})"};
    require(select_agent_planning_route(native, config, request, catalog, error).route_id == normal.id && error.empty(),
        "explicit JSON selection was lost");
    require(!native.requests.front().json_schema.empty() && native.requests.front().grammar.empty(),
        "JSON selector received DSL grammar");

    scripted_inference continuation;
    continuation.replies = {R"({"route_id":"normal-plan","confidence":0.9})"};
    require(select_agent_planning_route(continuation, config, request, catalog, error, true).route_id == normal.id && error.empty(),
        "optional workflow finish selection failed");
    const auto & continuation_messages = continuation.requests.front().messages;
    require(std::any_of(continuation_messages.begin(), continuation_messages.end(), [](const auto & message) {
        return message.content.find("finish with the current evidence") != std::string::npos;
    }), "continuation choice did not describe the finish option");

    request.tool_output_format = common_agent_tool_output_format::compact_dsl;
    scripted_inference invalid;
    invalid.replies = {R"({"route_id":"normal-plan","confidence":0.9})"};
    require(!select_agent_planning_route(invalid, config, request, catalog, error).route_id && !error.empty(),
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
    common_plan_observation observation{"choice-source", "investigate.events", R"({"count":0,"items":[],"complete":false})", 1.0f};
    require(common_plan_workflow_choice_available(observation, transition.choices.front()),
        "zero/false/empty observations were confused with missing fields");
    scripted_inference choice;
    choice.replies = {"open! selection.candidate\nid: inspect"};
    std::string selected;
    require(select_agent_planning_workflow_candidate(choice, config, request, observation, transition, selected, error), error);
    require(selected == "inspect" && choice.requests.front().json_schema.empty(), "workflow choice escaped DSL");
    require(choice.requests.front().messages[1].content.find("Reuse sufficient retained evidence")!=std::string::npos,
        "workflow action choice used the record-matching acquisition prompt");

    observation.summary = R"({"ok":false,"count":0,"items":[],"complete":false})";
    require(!common_plan_workflow_choice_available(observation, transition.choices.front()),
        "failed observations enabled a choice");

    common_plan_workflow_transition graph_choice;
    graph_choice.model_instruction = "If truncated is true, report that more matches exist.";
    graph_choice.choices.push_back({"finish", "Finish without expansion", "done", {}});
    for (int i = 0; i < 5; ++i) graph_choice.choices.push_back({"result_" + std::to_string(i + 1),
        "Expand result " + std::to_string(i + 1), "expand", {"/result/nodes/" + std::to_string(i) + "/id"}});
    common_plan_observation graph_matches{"graph-search", "graph.find",
        R"({"result":{"nodes":[{"id":"a","name":"First"},{"id":"b","name":"Second"},{"id":"c","name":"Third"},{"id":"d","name":"Fourth"},{"id":"e","name":"Fifth"}],"truncated":true}})", 1.0f};
    scripted_inference graph_selector;
    graph_selector.replies = {"open! selection.candidate\nid: result_2"};
    require(select_agent_planning_workflow_candidate(graph_selector, config, request, graph_matches, graph_choice, selected, error), error);
    require(selected == "result_2" && graph_selector.requests.front().messages.back().content.find("Fifth") != std::string::npos &&
        graph_selector.requests.front().messages.back().content.find("\"truncated\":true") != std::string::npos &&
        std::any_of(graph_selector.requests.front().messages.begin(), graph_selector.requests.front().messages.end(),
            [](const auto & message) { return message.content.find("more matches") != std::string::npos; }),
        "graph choice did not expose five results and the truncation notice");
    common_chat_tool graph_find{"graph.find", "Find graph nodes",
        R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":32}}})"};
    scripted_inference graph_arguments;
    graph_arguments.replies = {"open! graph.find\nquery: Alice"};
    const auto bounded_action = select_agent_planning_workflow_action(graph_arguments, config, request, {graph_find},
        {{"graph.find", R"({"limit":5})", "workflow-graph-entry-template", false}}, {},
        common_agent_tool_output_format::compact_dsl, error);
    require(error.empty() && bounded_action.action && json::parse(bounded_action.action->arguments_json)["limit"] == 5 &&
        json::parse(bounded_action.action->arguments_json)["query"] == "Alice", "host entry limit was not retained: " + error);

    common_chat_tool lookup{"openalex.getWork", "Get a work by ID",
        R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})"};
    lookup.item_lookup_parameter = "id";
    common_agent_request lookup_request;
    lookup_request.prompt = "Find work about urban cooling";
    lookup_request.workflow_definition = common_plan_workflow_definition{};
    lookup_request.tool_output_format = common_agent_tool_output_format::native;
    scripted_inference absent;
    absent.replies = {R"({"tool":"selection.decline","args":{}})"};
    const auto unavailable = select_agent_planning_workflow_action(absent, config, lookup_request, {lookup}, {}, {},
        lookup_request.tool_output_format, error);
    require(!unavailable.action && unavailable.missing_grounded_identifier && absent.requests.size() == 1 &&
        absent.requests.front().json_schema.find("selection.decline") != std::string::npos,
        "lookup route could not explicitly decline a missing identifier");
    scripted_inference absent_dsl;
    absent_dsl.replies = {"open! selection.decline"};
    const auto unavailable_dsl = select_agent_planning_workflow_action(absent_dsl, config, lookup_request, {lookup}, {}, {},
        common_agent_tool_output_format::compact_dsl, error);
    require(!unavailable_dsl.action && unavailable_dsl.missing_grounded_identifier,
        "DSL lookup route could not decline a missing identifier");
    scripted_inference invented;
    invented.replies = {R"({"tool":"openalex.getWork","args":{"id":"W999"}})"};
    require(!select_agent_planning_workflow_action(invented, config, lookup_request, {lookup}, {}, {},
        lookup_request.tool_output_format, error).action && error.find("search first") != std::string::npos,
        "workflow accepted an invented lookup identity");
    lookup_request.prompt = "Read OpenAlex work W999";
    scripted_inference supplied;
    supplied.replies = invented.replies;
    require(select_agent_planning_workflow_action(supplied, config, lookup_request, {lookup}, {}, {},
        lookup_request.tool_output_format, error).action.has_value(), "workflow rejected the user's lookup identity");
    lookup_request.prompt = "Read the selected work";
    scripted_inference bound;
    bound.replies = {R"({"tool":"openalex.getWork","args":{}})"};
    require(select_agent_planning_workflow_action(bound, config, lookup_request, {lookup},
        {{"openalex.getWork", R"({"id":"W999"})", "workflow-transition", false}}, {},
        lookup_request.tool_output_format, error).action.has_value(), "workflow rejected a host-bound lookup identity");
    require(bound.requests.front().json_schema.find("selection.decline") == std::string::npos,
        "host-bound lookup identity was presented as missing");
    scripted_inference conflicting_then_fixed;
    conflicting_then_fixed.replies = {R"({"tool":"openalex.getWork","args":{"id":"W123"}})",
        R"({"tool":"openalex.getWork","args":{}})"};
    const auto corrected = select_agent_planning_workflow_action(conflicting_then_fixed, config, lookup_request, {lookup},
        {{"openalex.getWork", R"({"id":"W999"})", "workflow-transition", false}}, {},
        lookup_request.tool_output_format, error);
    require(corrected.action && json::parse(corrected.action->arguments_json)["id"] == "W999" &&
        conflicting_then_fixed.requests.size() == 2, "host-bound identity conflict was not repaired: " + error);
    scripted_inference model_bound;
    model_bound.replies = bound.replies;
    require(!select_agent_planning_workflow_action(model_bound, config, lookup_request, {lookup},
        {{"openalex.getWork", R"({"id":"W999"})", "workflow-action", false}}, {},
        lookup_request.tool_output_format, error).action,
        "a model-authored binding was promoted into observed lookup identity");
}

void result_requirement_selection() {
    common_agent_request request;
    request.prompt = "Conversation goal: overall amount count and mean, without separate customer groups.";
    common_agent_dataset_descriptor orders;
    orders.ref.uri = "dataset://seed/orders"; orders.ref.name = "orders";
    orders.columns = {{"customer_id"}, {"amount"}};
    orders.ref.column_count = orders.columns.size();
    request.available_datasets = {orders};
    common_plan_state plan;
    plan.workflow_definition = common_plan_workflow_definition{};
    plan.workflow_definition->workflow_ref = "workflow://dataset/analysis";
    plan.workflow_definition->workflow_revision = "v4";
    plan.workflow_definition->result_contracts = {{"statistics.describe","statistics.describe.v1"}};
    request.workflow_definition = plan.workflow_definition;
    common_plan_step step;
    step.id = "describe"; step.objective = "Describe the selected amount column.";
    // The proposed action is deliberately wrong. It must not become the
    // evidence used to infer what the user requested.
    step.tool_call = common_plan_tool_call{"statistics.describe", R"({"dataset":"dataset://seed/orders","columns":["amount"],"group_by":["customer_id"]})"};
    request.tool_argument_bindings = {{"statistics.describe",step.tool_call->arguments_json,"workflow-action",false}};
    common_chat_tool tool{"statistics.describe","Describe numeric columns.","{}"};
    common_agent_generation_config config; config.n_predict = 128;
    scripted_inference dsl;
    dsl.replies = {"open! selection.analysis_result\ndataset: d1\ncolumns: amount\nmeasures: count, mean\ngroup_by: []\ncoverage: complete"};
    std::string requirement, error;
    require(select_agent_planning_result_requirement(dsl, config, request, plan, step, {tool}, requirement, error), error);
    const auto value = json::parse(requirement);
    require(value["value"]["dataset"] == orders.ref.uri && value["value"]["group_by"].empty(), "result interpretation copied wrong grouping");
    require(dsl.requests.size() == 1 && dsl.requests[0].json_schema.empty() && !dsl.requests[0].grammar.empty(), "result selection bypassed DSL default");
    require(dsl.requests[0].messages.back().content.find(request.prompt) != std::string::npos &&
        dsl.requests[0].messages.back().content.find("[Host-fixed arguments]\n{}") != std::string::npos,
        "result selection lost task context or treated a proposed action as a host instruction");
    request.tool_output_format = common_agent_tool_output_format::native;
    scripted_inference native;
    native.replies = {R"({"dataset":"d1","columns":["amount"],"measures":["count","mean"],"group_by":[],"coverage":"complete"})"};
    require(select_agent_planning_result_requirement(native, config, request, plan, step, {tool}, requirement, error), error);
    require(!native.requests[0].json_schema.empty() && native.requests[0].grammar.empty(), "explicit native mode received DSL");
    for (const auto interrupted : {true, false}) {
        scripted_inference stopped;
        stopped.replies = native.replies;
        stopped.status = interrupted ? common_agent_generation_status::cancelled : common_agent_generation_status::completed;
        stopped.stop = interrupted ? common_agent_generation_stop_reason::cancelled : common_agent_generation_stop_reason::limit;
        common_agent_generated_text_result generated;
        require(!select_agent_planning_result_requirement(stopped, config, request, plan, step, {tool}, requirement, error, &generated) &&
            stopped.requests.size() == 1 && requirement.empty() && generated.stop_reason == stopped.stop,
            "cancellation or output exhaustion became an ordinary selector repair/fallback");
    }
    scripted_inference unauthorized;
    require(!select_agent_planning_result_requirement(unauthorized, config, request, plan, step, {}, requirement, error) &&
        unauthorized.requests.empty(), "result selection widened the authorized operation view");
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
    const auto selected = select_agent_planning_workflow_action(inference, config, request,
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
    const auto native_action = select_agent_planning_workflow_action(native, config, request,
        {tool}, {}, {}, common_agent_tool_output_format::native, error);
    require(error.empty() && native_action.action && !native.requests.front().json_schema.empty() &&
        native.requests.front().grammar.empty() && json::parse(native_action.action->arguments_json)["dataset"] == orders.ref.uri,
        "explicit JSON mode did not preserve the same host dataset identity");

    scripted_inference repair;
    repair.replies = {
        "open! statistics.describe\ndataset: d2\ncolumns: amount\ngroup_by: null_count",
        "open! statistics.describe\ndataset: d2\ncolumns: amount",
    };
    const auto repaired = select_agent_planning_workflow_action(repair, config, request,
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
    const auto corrected = select_agent_planning_workflow_action(cross_dataset, config, request,
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
    const auto named = select_agent_planning_workflow_action(named_columns, config, request,
        {tool}, {}, {}, common_agent_tool_output_format::compact_dsl, error);
    require(error.empty() && named.action.has_value(), "valid result-like source columns were rejected: " + error);
    require(json::parse(named.action->arguments_json)["group_by"] == json::array({"null", "none"}),
        "explicit source-column names were silently removed");
}

void workflow_argument_budget_contracts() {
    common_agent_request request;
    request.prompt = "Create a record with a detailed description";
    request.tool_output_format = common_agent_tool_output_format::native;
    common_agent_generation_config config; config.n_predict = 2048;
    common_chat_tool tool{"records.create", "Create a record",
        R"({"type":"object","required":["description"],"properties":{"description":{"type":"string","maxLength":1024}}})"};
    scripted_inference inference;
    inference.replies = {R"({"tool":"records.create","args":{"description":)",
        R"({"tool":"records.create","args":{"description":"Complete bounded description"}})"};
    inference.stops = {common_agent_generation_stop_reason::limit, common_agent_generation_stop_reason::eos};
    std::string error;
    const auto selected = select_agent_planning_workflow_action(inference, config, request, {tool}, {}, {},
        request.tool_output_format, error);
    require(selected.action && error.empty() && inference.requests.size() == 2,
        "truncated workflow arguments were accepted or not regenerated: " + error);
    require(inference.requests[0].options.n_predict > 128 &&
        inference.requests[1].options.n_predict > inference.requests[0].options.n_predict &&
        inference.requests[1].options.n_predict <= config.n_predict,
        "argument demand or bounded growth ignored the configured ceiling");
    config.n_predict = 128;
    scripted_inference capped;
    capped.replies = {"{"}; capped.stop = common_agent_generation_stop_reason::limit;
    require(!select_agent_planning_workflow_action(capped, config, request, {tool}, {}, {},
        request.tool_output_format, error).action && capped.requests.size() == 1,
        "truncation retried beyond the configured output ceiling");
    scripted_inference full_context;
    full_context.capacity=256;
    require(!select_agent_planning_workflow_action(full_context,config,request,{tool},{},{},
        request.tool_output_format,error).action && full_context.requests.empty() && !error.empty(),
        "workflow argument inference ran with no context capacity");
}

void workflow_no_progress_contracts() {
    common_chat_tool tool{"records.update", "Update a revision-pinned record",
        R"({"type":"object","required":["revision","value"],"properties":{"revision":{"type":"integer"},"value":{"type":"string"}},"x-agent-progress":{"changed":"/changed"}})"};
    common_plan_workflow_transition update;
    update.id="update"; update.tool_name=tool.name; update.model_supplies_arguments=true;
    common_plan_state plan;
    common_plan_step completed;
    completed.id="prior"; completed.status=common_plan_step_status::completed;
    completed.tool_call=common_plan_tool_call{tool.name,R"({"revision":3,"value":"saved"})"};
    plan.steps={completed};
    common_plan_observation observation{"tool:prior:records.update",tool.name,
        R"({"ok":true,"result":{"revision":3,"changed":false}})",1.0f};
    plan.observations={observation};
    common_agent_request request; request.tool_output_format=common_agent_tool_output_format::native;
    common_agent_generation_config config; config.n_predict=1024;
    std::string arguments,error;
    scripted_inference duplicate;
    duplicate.replies={R"({"tool":"records.update","args":{"value":"saved","revision":3}})"};
    require(!select_agent_planning_workflow_transition_arguments(duplicate,config,request,{tool},plan,observation,
        update,"{}",arguments,error) && error.rfind("workflow.no_progress:",0)==0 && arguments.empty(),
        "identical no-change call survived persisted evidence replay");
    scripted_inference changed;
    changed.replies={R"({"tool":"records.update","args":{"revision":3,"value":"new"}})"};
    require(select_agent_planning_workflow_transition_arguments(changed,config,request,{tool},plan,observation,
        update,"{}",arguments,error),"different requested state was blocked: "+error);
    scripted_inference new_revision;
    new_revision.replies={R"({"tool":"records.update","args":{"revision":4,"value":"saved"}})"};
    require(select_agent_planning_workflow_transition_arguments(new_revision,config,request,{tool},plan,observation,
        update,"{}",arguments,error),"new state revision was blocked: "+error);

    auto changed_plan=plan;
    changed_plan.observations.front().summary=R"({"ok":true,"result":{"revision":3,"changed":true}})";
    scripted_inference first_repeat;
    first_repeat.replies={R"({"tool":"records.update","args":{"revision":3,"value":"saved"}})"};
    require(select_agent_planning_workflow_transition_arguments(first_repeat,config,request,{tool},changed_plan,observation,
        update,"{}",arguments,error),"a successful change was misclassified as a proven no-op: "+error);

    common_agent_route_candidate route; route.id="no-progress-route"; route.resolved_tools={tool.name,"records.inspect"};
    common_plan_workflow_definition graph;
    graph.start_state="start";graph.terminal_states={"done"};
    common_plan_workflow_transition first;first.id="first";first.from_state="start";first.to_state="choose";first.tool_name=tool.name;
    common_plan_workflow_transition choice;choice.id="choice";choice.from_state="choose";
    choice.kind=common_plan_workflow_transition_kind::model_choice;
    choice.choices={{"update","Update a changed value","update",{}},{"inspect","Inspect saved evidence","inspect",{}}};
    update.from_state="update";update.to_state="choose";
    common_plan_workflow_transition inspect;inspect.id="inspect";inspect.from_state="inspect";inspect.to_state="done";
    inspect.tool_name="records.inspect";
    graph.transitions={first,choice,update,inspect};route.workflow_definition=graph;
    plan.workflow_definition=graph;plan.route_binding=common_plan_route_binding{};plan.route_binding->route_id=route.id;
    plan.steps[0].semantic_alias="workflow-transition:first";
    size_t choices=0;
    auto continuation=make_agent_workflow_graph_continuation(route,
        [&](const auto & observed,const auto & available,std::string & selected,std::string & failure) {
            failure.clear();++choices;
            if(choices==1) selected="update";
            else {
                require(json::parse(observed.summary).contains("host_recovery"),"no-progress repair lost host feedback");
                require(std::none_of(available.choices.begin(),available.choices.end(),[](const auto & c){return c.id=="update";}),
                    "reconsideration exposed the rejected choice again");
                selected="inspect";
            }
            return true;
        },[](const auto &,const auto &,const auto & transition,const auto &,std::string & args,std::string & failure) {
            if(transition.id=="update") {failure="workflow.no_progress: unchanged saved state";return false;}
            failure.clear();args="{}";return true;
        });
    common_agent_workflow_continuation steps;
    require(continuation(plan,"prior",observation,steps,error) && choices==2 && steps.size()==1 &&
        steps.front().tool_call->name=="records.inspect","no-progress workflow did not reconsider its next action: "+error);
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
    common_tool_catalog catalog;common_tool_bootstrap_result bootstrap;
    require(catalog.bootstrap("minimal", bootstrap, error), error);
    const auto collect = catalog.find_definition("investigate.collect");
    require(collect != nullptr, "collection contract is missing");
    auto collect_schema = json::parse(collect->input_schema_json);
    require(common_agent_project_dataset_model_schema(collect_schema, inventory, json::object(), error), error);
    require(collect_schema["properties"]["identifier_field"]["enum"] == json::array({"amount", "region"}),
        "collection did not expose the selected dataset's logical columns");
    require(!common_agent_validate_dataset_column_arguments(collect_schema,
        json{{"dataset",dataset.ref.uri},{"identifier_field","invented"}}, inventory, error),
        "collection accepted an invented logical column");
    require(common_agent_validate_dataset_column_arguments(collect_schema,
        json{{"dataset",dataset.ref.uri},{"identifier_field","region"}}, inventory, error), error);
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
    {
        // A host-bound report URI must validate against the execution view,
        // while the model-facing resource schema deliberately requires an id.
        common_chat_tool read{"resource.read", "Read resource",
            R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})"};
        common_plan_workflow_transition transition;
        transition.id = "read-report"; transition.tool_name = read.name;
        common_agent_request request;
        common_agent_generation_config config;
        common_plan_state plan;
        common_plan_observation observation{"report", "investigate.report.compose", R"({"ok":true})", 1.0f};
        scripted_inference inference;
        canonical_resource_view view;
        std::string arguments, error;
        require(select_agent_planning_workflow_transition_arguments(inference, config, request, {read},
            plan, observation, transition, R"({"uri":"agent-resource://report/1"})", arguments, error, &view) &&
            json::parse(arguments).at("uri") == "agent-resource://report/1" && inference.requests.empty(),
            "host-bound report resource was validated against the model id projection: " + error);
    }
    // A linked workflow's entry uses this same transition materializer.
    // Host-owned operation and identity do not require another generation.
    {
        common_chat_tool read{"records.read", "Read the selected record",
            R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"},"detail":{"type":"string"}}})"};
        common_plan_workflow_transition entry;
        entry.id = "read"; entry.tool_name = read.name;
        common_agent_request request;
        request.tool_argument_bindings = {{read.name, R"({"id":"observed-record"})", "workflow-continuation", false}};
        common_agent_generation_config config;
        common_plan_state plan;
        common_plan_observation observation{"source", "records.search", R"({"ok":true})", 1.0f};
        scripted_inference inference;
        std::string arguments, error;
        require(select_agent_planning_workflow_transition_arguments(inference, config, request, {read},
            plan, observation, entry, "{}", arguments, error) &&
            json::parse(arguments)["id"] == "observed-record" && inference.requests.empty(),
            "host-bound workflow entry unexpectedly required inference: " + error);
        request.tool_argument_bindings.clear();
        require(!select_agent_planning_workflow_transition_arguments(inference, config, request, {read},
            plan, observation, entry, "{}", arguments, error) && inference.requests.empty(),
            "incomplete deterministic entry silently requested model arguments");
        request.tool_argument_bindings = {{read.name, R"({"id":"observed-record"})", "workflow-continuation", false}};
        require(!select_agent_planning_workflow_transition_arguments(inference, config, request, {read},
            plan, observation, entry, R"({"id":"different-record"})", arguments, error) && inference.requests.empty(),
            "conflicting host bindings were silently replaced");
        entry.model_supplies_arguments = true;
        inference.replies = {R"({"tool":"records.read","args":{"detail":"summary"}})"};
        request.tool_output_format = common_agent_tool_output_format::native;
        require(select_agent_planning_workflow_transition_arguments(inference, config, request, {read},
            plan, observation, entry, "{}", arguments, error) && inference.requests.size() == 1 &&
            json::parse(arguments)["id"] == "observed-record" && json::parse(arguments)["detail"] == "summary",
            "explicit model arguments or host identity were lost: " + error);
    }
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
    common_agent_workflow_continuation steps;
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
    read.tool_call = common_plan_tool_call{"investigate.case.open", R"({"case_id":"case"})"}; plan.steps.push_back(read);
    plan.observations.push_back({"tool:read:investigate.case.open", "investigate.case.open", R"({"ok":true,"result":{"case":{"case_id":"case","revision":2}}})", 1.0f});
    auto publish = read; publish.id = "publish"; publish.tool_call = common_plan_tool_call{"investigate.collect", R"({"case_id":"case","expected_revision":2})"};
    plan.steps.push_back(publish);
    plan.observations.push_back({"tool:publish:investigate.collect", "investigate.collect", R"({"ok":true,"result":{"case_id":"case","revision":3}})", 1.0f});
    common_chat_tool events{"investigate.events", "Read events", R"({"type":"object","required":["case_id","case_revision"],"properties":{"case_id":{"type":"string"},"case_revision":{"type":"integer"}},"additionalProperties":false})"};
    common_plan_workflow_transition next; next.tool_name = events.name;
    scripted_inference inference; common_agent_generation_config config; std::string args;
    require(select_agent_planning_workflow_transition_arguments(inference, config, request, {events}, plan,
        plan.observations.back(), next, "{}", args, error), error);
    require(json::parse(args)["case_revision"] == 3 && inference.requests.empty(),
        "deterministic continuation failed to use its successful publication baseline");
    request.investigation_context->revision = 3;
    require(select_agent_planning_workflow_transition_arguments(inference, config, request, {events}, plan,
        plan.observations.back(), next, "{}", args, error), "resume failed to replay its own revision chain: " + error);
    request.investigation_context_json=R"({"case_id":"case","revision":3,"processing_status":"query_complete"})";
    common_plan_workflow_transition review_choice;
    review_choice.choices.push_back({"reuse","Use retained evidence","done",{}});
    request.tool_output_format=common_agent_tool_output_format::native;
    scripted_inference reviewer;reviewer.replies={R"({"id":"reuse"})"};
    std::string selected;
    require(select_agent_planning_workflow_candidate(reviewer,config,request,plan.observations.back(),review_choice,selected,error),error);
    require(reviewer.requests.front().messages.back().content.find("query_complete")!=std::string::npos &&
        reviewer.requests.front().messages.back().content.find("supersedes older case snapshots")!=std::string::npos,
        "workflow selector omitted the separate current case projection");
    auto concurrent = read; concurrent.id = "concurrent-read"; plan.steps.push_back(concurrent);
    plan.observations.push_back({"tool:concurrent-read:investigate.case.open", "investigate.case.open", R"({"ok":true,"result":{"case":{"case_id":"case","revision":4}}})", 1.0f});
    require(!select_agent_planning_workflow_transition_arguments(inference, config, request, {events}, plan,
        plan.observations.back(), next, "{}", args, error), "a read silently rebased the workflow onto an external revision");
}

void automatic_resume_capacity() {
    std::string error;
    common_plan_policy_config limits;limits.max_observations=1;
    common_plan_in_memory_store plans{common_plan_policy(limits)};
    common_memory_in_memory_store memory;
    require(plans.open("",error) && memory.open("",error),error);
    common_plan_state plan;plan.id="exhausted";plan.status=common_plan_status::active;
    plan.namespace_id="local";plan.session_id="session";plan.turn_id="old";plan.scope=common_plan_scope::session;
    common_plan_workflow_definition workflow;workflow.start_state="start";workflow.terminal_states={"done"};
    common_plan_workflow_transition transition;transition.id="read";transition.from_state="start";
    transition.to_state="done";transition.tool_name="repository.read";workflow.transitions={transition};
    plan.workflow_definition=workflow;
    require(plans.create(plan,error),error);
    scripted_inference inference;common_agent_generation_config config;
    common_agent_orchestration_config orchestration;orchestration.prompt="Continue the case";
    common_agent_scope scope;scope.namespace_id="local";scope.session_id="session";
    scope.turn_id="new";scope.plan_scope=common_plan_scope::session;
    std::string current=plan.id;
    std::vector<common_blueprint_candidate> blueprints;
    common_agent_route_candidate route;bool selected=false;
    std::vector<common_memory_hit> memories;std::vector<common_agent_event> events;
    std::vector<common_runtime_trace_entry> trace;
    common_agent_orchestration_runtime_context context{inference,config,orchestration,memory,current,scope,plans,
        blueprints,nullptr,nullptr,route,selected,memories,events,trace};
    require(maybe_auto_select_plan(context,error) && current.empty() && inference.requests.empty(),
        "automatic selection resumed an exhausted plan or offered it to the model");
    current=plan.id;scope.turn_id="old";
    require(maybe_auto_select_plan(context,error) && current==plan.id,"same accepted turn forked into a new plan");
    scope.turn_id="new";orchestration.agent_plan="explicit";
    require(maybe_auto_select_plan(context,error) && current==plan.id,"explicit plan pin was silently replaced");
    // A new turn may select unfinished work in an exhausted plan. The old
    // journal remains unchanged and route selection starts afresh.
    common_plan_operation observed;observed.kind=common_plan_operation_kind::record_observation;
    observed.plan_id=plan.id;observed.expected_version=plan.version;observed.host_authorized=true;
    observed.observation=common_plan_observation{"review","response_review",
        R"({"accepted":false,"checklist":{"blocking":[{"description":"Inspect saved report"}]}})",1.0f};
    require(plans.apply(observed,plan,error),error);
    orchestration.agent_plan="auto";current.clear();context.prior_work_checkpoint=&orchestration.prior_work_checkpoint;
    context.tool_output_format=common_agent_tool_output_format::native;
    inference.replies={R"({"decision":"resume","plan_id":"exhausted","confidence":1.0})"};
    require(maybe_auto_select_plan(context,error) && current.empty() && orchestration.prior_work_checkpoint,
        "unfinished exhausted plan did not yield a linked successor: "+error);
    const auto checkpoint=json::parse(orchestration.prior_work_checkpoint->summary);
    require(checkpoint["source_plan_id"]==plan.id && checkpoint["source_plan_version"]==plan.version &&
        checkpoint["review"]["accepted"]==false && plans.get(plan.id,error)->version==plan.version,
        "successor changed the source journal or lost its unfinished review");
    context.tool_output_format=common_agent_tool_output_format::compact_dsl;
    orchestration.prior_work_checkpoint.reset();inference.requests.clear();
    inference.replies={"open! selection.plan\ndecision: resume\nplan_id: exhausted\nconfidence: 1"};
    require(maybe_auto_select_plan(context,error) && current.empty() && orchestration.prior_work_checkpoint,
        "DSL selection failed to retain prior work: "+error);
}

void review_read_progress_contracts() {
    common_agent_request request; request.tool_output_format=common_agent_tool_output_format::native;
    common_agent_generation_config config; scripted_inference inference;
    common_chat_tool read{"resource.read","Read a resource",R"({"type":"object","properties":{"uri":{"type":"string"},"offset":{"type":"integer"}},"required":["uri"],"additionalProperties":false})"};
    canonical_resource_view view; view.tools={read};
    common_plan_state plan; plan.id="read-progress";
    common_plan_step completed;completed.id="read";completed.status=common_plan_step_status::completed;
    completed.tool_call=common_plan_tool_call{read.name,R"({"uri":"resource://retained","offset":0})"};
    plan.steps={completed};
    plan.observations.push_back({"tool:read:resource.read",read.name,R"({"ok":true,"result":{"content":"retained"}})",1.0f});
    common_plan_workflow_transition transition;transition.tool_name=read.name;
    common_plan_observation review{"review",common_agent_workflow_review_source,R"({"ok":true,"review_request":{"guidance":["Inspect remaining evidence"]}})",1.0f};
    std::string args,error;
    const auto select=[&](const std::string & seed) {
        return select_agent_planning_workflow_transition_arguments(inference,config,request,{read},plan,review,
            transition,seed,args,error,&view);
    };
    require(!select(R"({"uri":"resource://retained","offset":0})") &&
        error.rfind("workflow.no_progress:",0)==0,"review replayed an identical successful read");
    require(select(R"({"uri":"resource://retained","offset":128})"),"review blocked an unread byte position: "+error);
    review.summary=R"({"ok":true})";
    require(select(R"({"uri":"resource://retained","offset":0})"),"ordinary read acquired review-only restrictions");
    review.summary=R"({"ok":true,"review_request":{}})";
    auto mutation=completed;mutation.id="mutation";mutation.tool_call=common_plan_tool_call{"workspace.update","{}"};
    plan.steps.push_back(mutation);
    require(select(R"({"uri":"resource://retained","offset":0})"),"review blocked a read after a state mutation: "+error);
    plan.steps.pop_back();plan.route_binding=common_plan_route_binding{};plan.route_binding->workflow_step_start_index=1;
    require(select(R"({"uri":"resource://retained","offset":0})"),"prior route read blocked a new authorized route");
    require(inference.requests.empty(),"deterministic review-read guard invoked inference");
}

void review_entry_contracts() {
    std::string error;
    common_agent_route_candidate route;route.id="review-route";route.resolved_tools={"resource.read"};
    common_plan_workflow_definition graph;graph.start_state="start";graph.terminal_states={"done"};
    common_plan_workflow_transition initial;initial.id="initial";initial.from_state="start";
    initial.to_state="done";initial.tool_name="resource.read";
    common_plan_workflow_transition entry;entry.id="review";entry.from_state="done";entry.to_state="inspect";
    entry.kind=common_plan_workflow_transition_kind::model_choice;entry.review_only=true;
    entry.choices.push_back({"read","Read missing evidence","inspect",{}});
    auto read=initial;read.id="read-more";read.from_state="inspect";
    graph.transitions={initial,entry,read};route.workflow_definition=graph;
    common_plan_state plan;plan.id="review-plan";plan.status=common_plan_status::active;
    plan.workflow_definition=graph;plan.route_binding=common_plan_route_binding{};plan.route_binding->route_id=route.id;
    common_plan_step done{"original","Read","Read original range"};done.status=common_plan_step_status::completed;
    done.tool_call=common_plan_tool_call{"resource.read","{}"};done.semantic_alias="workflow-transition:initial";
    plan.steps={done};
    common_plan_observation observed{"tool:original:resource.read","resource.read",R"({"ok":true,"result":{"content":"known"}})",1.0f};
    plan.observations={observed};
    size_t selections=0;
    auto provider=make_agent_workflow_graph_continuation(route,[&](const auto &,const auto &,std::string & id,std::string & failure) {
        ++selections;id="read";failure.clear();return true;
    });
    common_agent_workflow_continuation next;
    require(provider(plan,done.id,observed,next,error) && next.empty() && selections==0,
        "normal terminal workflow entered review automatically");
    observed.source=common_agent_workflow_review_source;
    require(provider(plan,done.id,observed,next,error) && next.size()==1 && selections==1,error);
    require(next.front().tool_call->name=="resource.read","review changed its declared operation");
    next.front().status=common_plan_step_status::completed;plan.steps.push_back(next.front());
    require(common_plan_workflow_is_terminal(plan,graph),"review aliases did not replay through their declared graph");
    plan.status=common_plan_status::completed;
    require(!provider(plan,done.id,observed,next,error),"review reopened an accepted completed plan");

    // Discovery has no selected case until a real host identity is available.
    common_agent_request request;request.tool_output_format=common_agent_tool_output_format::native;
    common_chat_tool open{"investigate.case.open","Open selected record",R"({"type":"object","properties":{"case_id":{"type":"string"}},"required":["case_id"],"additionalProperties":false})"};
    common_plan_workflow_transition select;select.tool_name=open.name;select.model_supplies_arguments=true;
    scripted_inference inference;inference.replies={R"({"tool":"investigate.case.open","args":{"case_id":"listed-case"}})"};
    common_plan_observation listed{"listed","investigate.case.list",R"({"ok":true,"result":{"cases":[{"case_id":"listed-case"}]}})",1.0f};
    common_agent_generation_config config;std::string arguments;
    require(select_agent_planning_workflow_transition_arguments(inference,config,request,{open},{},listed,select,"{}",arguments,error) &&
        json::parse(arguments)["case_id"]=="listed-case","discovery manufactured an empty selected-case binding: "+error);
}

void issued_cursor_contracts() {
    common_agent_request request;request.tool_output_format=common_agent_tool_output_format::native;
    common_agent_generation_config config;
    common_chat_tool page{"records.page","Read a bounded page",
        R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string"},"cursor":{"type":"string"}},"x-agent-pagination":{"cursor":"cursor","next_cursor":"next_cursor","identity_fields":["query"]}})"};
    common_plan_workflow_transition transition;transition.tool_name=page.name;transition.model_supplies_arguments=true;
    common_plan_state plan;plan.id="pages";
    common_plan_step first{"first","Read","Read records"};first.status=common_plan_step_status::completed;
    first.tool_call=common_plan_tool_call{page.name,R"({"query":"known"})"};plan.steps={first};
    common_plan_observation observed{"tool:first:records.page",page.name,
        R"({"ok":true,"result":{"next_cursor":"issued-page-2"}})",1.0f};
    plan.observations={observed};
    std::string args,error;
    scripted_inference valid;valid.replies={R"({"tool":"records.page","args":{"cursor":"issued-page-2"}})"};
    require(select_agent_planning_workflow_transition_arguments(valid,config,request,{page},plan,observed,
        transition,R"({"query":"known"})",args,error) && json::parse(args)["cursor"]=="issued-page-2",
        "compatible issued cursor was rejected: "+error);
    scripted_inference forged;forged.replies={R"({"tool":"records.page","args":{"cursor":"query=known"}})",
        R"({"tool":"records.page","args":{}})"};
    require(select_agent_planning_workflow_transition_arguments(forged,config,request,{page},plan,observed,
        transition,R"({"query":"known"})",args,error) && !json::parse(args).contains("cursor") &&
        forged.requests.size()==2,"forged cursor did not receive bounded first-page repair: "+error);
    scripted_inference transplant;transplant.replies={R"({"tool":"records.page","args":{"cursor":"issued-page-2"}})",
        R"({"tool":"records.page","args":{"cursor":"issued-page-2"}})"};
    require(!select_agent_planning_workflow_transition_arguments(transplant,config,request,{page},plan,observed,
        transition,R"({"query":"different"})",args,error) && args.empty(),"cursor crossed its query identity");
}

void package_contracts() {
    std::ifstream input(std::string(LLAMA_AGENT_SOURCE_DIR) + "/docs/examples/agent-investigation-workflows-v1.json");
    std::stringstream text; text << input.rdbuf(); common_agent_bootstrap_package package; std::string error;
    require(common_agent_package_parse_json(text.str(), package, error), error);
    require(package.version == "v11" && package.workflows.size() == 9, "new investigation methods missing");
    for (const auto & workflow : package.workflows) require(!workflow.definition.transitions.empty(), "method still has no executable graph");
    const auto identity = std::find_if(package.workflows.begin(), package.workflows.end(), [](const auto & workflow) {
        return workflow.definition.workflow_ref == "workflow://investigation/identity-analysis";
    });
    require(identity != package.workflows.end(), "identity workflow missing");
    {
        common_agent_request entry_request;
        entry_request.tool_output_format = common_agent_tool_output_format::native;
        entry_request.workflow_definition = identity->definition;
        entry_request.prompt = "Continue after the earlier case creation failed.";
        common_chat_tool open{"investigate.case.open", "Open a case", R"({"type":"object","properties":{}})"};
        common_chat_tool create{"investigate.case.create", "Create a case", R"({"type":"object","properties":{}})"};
        scripted_inference selection;
        selection.replies = {R"("investigate.case.create")", R"({"tool":"investigate.case.create","args":{}})"};
        common_agent_generation_config generation;
        const auto action = select_agent_planning_workflow_action(selection, generation, entry_request,
            {open, create}, {}, {}, entry_request.tool_output_format, error);
        require(action.action && action.action->tool_name == "investigate.case.create", error);
        std::string prompt;
        for (const auto & message : selection.requests.front().messages) prompt += message.content + "\n";
        require(prompt.find("Open the explicitly selected existing case") != std::string::npos &&
            prompt.find("continuation after an earlier case-creation failure") != std::string::npos,
            "workflow entry conditions were absent from the operation choice");
    }
    const auto choice = std::find_if(identity->definition.transitions.begin(), identity->definition.transitions.end(),
        [](const auto & transition) { return transition.id == "choose-next"; });
    require(choice != identity->definition.transitions.end() && choice->from_state == "assess",
        "identity analysis result is not the direct next-choice observation");
    require(std::none_of(identity->definition.transitions.begin(), identity->definition.transitions.end(),
        [](const auto & transition) { return transition.id == "inspect-result"; }),
        "identity analysis result is still replaced by a case summary");
    const auto actor=std::find_if(package.workflows.begin(),package.workflows.end(),[](const auto & item) {
        return item.definition.workflow_ref=="workflow://investigation/actor-relationship";
    });
    require(actor!=package.workflows.end(),"actor workflow missing");
    const auto & definition=actor->definition;
    const auto listed=std::find_if(definition.transitions.begin(),definition.transitions.end(),[](const auto & t){return t.id=="choose-listed-case";});
    require(listed!=definition.transitions.end(),"case list has no empty-result decision");
    common_plan_observation empty{"empty","investigate.case.list",R"({"ok":true,"result":{"cases":[]}})",1.0f};
    require(!common_plan_workflow_choice_available(empty,listed->choices.front()),"empty case list offered an existing-case lookup");
    require(common_plan_workflow_choice_available(empty,listed->choices.back()),"empty case list lost its clarification path");
    const auto source_choice=std::find_if(definition.transitions.begin(),definition.transitions.end(),[](const auto & t){return t.id=="choose-case-source";});
    require(source_choice!=definition.transitions.end(),"source discovery has no empty-result decision");
    empty.summary=R"({"ok":true,"result":{"sources":[]}})";
    require(!common_plan_workflow_choice_available(empty,source_choice->choices.front()) &&
        common_plan_workflow_choice_available(empty,source_choice->choices.back()),
        "empty authorized source list offered case creation");

    const auto review_entry=std::find_if(definition.transitions.begin(),definition.transitions.end(),[](const auto & t){return t.review_only;});
    require(review_entry!=definition.transitions.end(),"investigation workflow lacks declared review entry");

    std::vector<common_chat_tool> available={
        {"investigate.case.items","Read items",R"({"type":"object"})"},
        {"investigate.case.open","Open case",R"({"type":"object"})"},
        {"investigate.statement.record","Record text",R"({"type":"object","x-agent-investigation-source-kinds":["document","log_file"]})"},
        {"investigate.graph.expand","Expand retained events",R"({"type":"object"})"}};
    std::vector<std::string> names;for(const auto & tool:available)names.push_back(tool.name);
    common_agent_request search;
    search.investigation_source_kinds=std::vector<common_investigation_source_kind>{common_investigation_source_kind::search_connector};
    require(!common_agent_workflow_path_eligible(search,definition,"text-underlay",names,available,error) && error.empty(),
        "text path passed source eligibility through its read-only prefix");
    require(common_agent_workflow_path_eligible(search,definition,"event-underlay",names,available,error),error);
    search.investigation_source_kinds=std::vector<common_investigation_source_kind>{common_investigation_source_kind::document};
    require(common_agent_workflow_path_eligible(search,definition,"text-underlay",names,available,error),
        "document actor analysis lost its existing text path");
    for (const auto & blueprint : package.blueprints) require(!blueprint.workflow_bindings.front().phase_bindings.empty(), "blueprint phases are unbound");
    std::string exported;
    require(common_agent_package_to_json(package, exported, error), error);
    common_agent_bootstrap_package restored;
    require(common_agent_package_parse_json(exported, restored, error), error);
    require(std::any_of(restored.workflows.front().definition.transitions.begin(), restored.workflows.front().definition.transitions.end(),
        [](const auto & t){return t.review_only;}), "package codec dropped review authority");
    require(restored.blueprints.front().workflow_bindings.front().phase_bindings == package.blueprints.front().workflow_bindings.front().phase_bindings,
        "package roundtrip dropped phase references");
    const auto report=std::find_if(restored.workflows.begin(),restored.workflows.end(),[](const auto & item) {
        return item.definition.workflow_ref=="workflow://investigation/report-composition";
    });
    require(report!=restored.workflows.end() && report->definition.workflow_revision=="v12",
        "report scope repair revision missing");
    const auto & report_graph=report->definition;
    const auto scope=std::find_if(report_graph.transitions.begin(),report_graph.transitions.end(),[](const auto & t) {
        return t.id=="report-update-case-scope";
    });
    require(scope!=report_graph.transitions.end() && scope->review_mutation_allowed &&
        scope->tool_name=="investigate.case.update_scope", "package codec lost report scope repair authority");
    common_agent_route_candidate report_route;report_route.id="report-repair";report_route.workflow_definition=report_graph;
    for (const auto & t:report_graph.transitions) if(!t.tool_name.empty()) report_route.resolved_tools.push_back(t.tool_name);
    common_plan_state report_plan;report_plan.id="report-repair";report_plan.status=common_plan_status::active;
    report_plan.workflow_definition=report_graph;report_plan.route_binding=common_plan_route_binding{};
    report_plan.route_binding->route_id=report_route.id;
    // Start at the same terminal report read that reflection repairs in production.
    for (const auto & id:{"open-case","inspect-underlay","choose-report-preparation=compose,compose-report","read-report"}) {
        common_plan_step step;step.id="old:"+std::string(id);step.status=common_plan_step_status::completed;
        step.semantic_alias="workflow-transition:"+std::string(id);report_plan.steps.push_back(step);
    }
    auto repair=make_agent_workflow_graph_continuation(report_route,
        [](const auto &,const auto & t,std::string & choice,std::string & failure) {
            failure.clear(); choice=t.review_only ? "repair-work" : "update-scope";return true;
        },[](const auto &,const auto &,const auto &,const std::string & seed,std::string & args,std::string & failure) {
            args=seed;failure.clear();return true;
        });
    common_plan_observation trigger{"review",common_agent_workflow_review_source,R"({"ok":true,"review_request":{"guidance":["Save requested scope"]}})",1.0f};
    common_agent_workflow_continuation admitted;
    const auto advance=[&](const std::string & expected) {
        require(repair(report_plan,report_plan.steps.back().id,trigger,admitted,error) && admitted.size()==1,error);
        require(admitted.front().tool_call && admitted.front().tool_call->name==expected,
            "report repair did not follow the declared preparation graph");
        auto step=admitted.front();step.status=common_plan_step_status::completed;
        report_plan.steps.push_back(step);
        trigger={"tool:"+step.id+":"+expected,expected,R"({"ok":true,"result":{"resource":{"uri":"resource://report"},"markdown_uri":"resource://report"}})",1.0f};
        report_plan.observations.push_back(trigger);
    };
    advance("investigate.case.open");
    advance("investigate.case.items");
    advance("investigate.case.update_scope");
    advance("investigate.case.open");
    advance("investigate.case.items");
    // The saved scope now satisfies the request: choose compose, not another mutation.
    repair=make_agent_workflow_graph_continuation(report_route,
        [](const auto &,const auto &,std::string & choice,std::string & failure) {
            failure.clear();choice="compose";return true;
        },[](const auto &,const auto &,const auto &,const std::string & seed,std::string & args,std::string & failure) {
            args=seed;failure.clear();return true;
        });
    advance("investigate.report.compose");
    advance("resource.read");
    require(common_plan_workflow_is_terminal(report_plan,report_graph),"scope repair did not return to terminal review");
    auto invalid_mutation=json::parse(exported);
    auto & bad_transition=invalid_mutation["workflows"][0]["transitions"][0];
    bad_transition["review_mutation_allowed"]=true;
    bad_transition["kind"]="model_choice";
    bad_transition["choices"]=json::array({{{"id","choose"},{"description","Choose"},{"to",bad_transition["to"]}}});
    require(!common_agent_package_parse_json(invalid_mutation.dump(),restored,error),
        "review mutation authorization was accepted on a model choice");

    auto invalid_review = json::parse(exported);
    for (auto & t : invalid_review["workflows"][0]["transitions"]) if (t.value("review_only",false)) {
        t["review_only"]=false;break;
    }
    require(!common_agent_package_parse_json(invalid_review.dump(),restored,error),
        "ordinary execution was allowed to leave a terminal workflow state");
    invalid_review=json::parse(exported);
    for (auto & t : invalid_review["workflows"][0]["transitions"]) if (t.value("review_only",false)) {
        t["from"]="start";break;
    }
    require(!common_agent_package_parse_json(invalid_review.dump(),restored,error),
        "review authority was accepted on a normal execution entry");
    auto invalid = json::parse(exported);
    invalid["blueprints"][0]["workflow_bindings"][0]["phase_bindings"]["invented"] = {"open-case"};
    require(!common_agent_package_parse_json(invalid.dump(), restored, error), "invalid phase mapping was imported");
}
}
int main(int argc, char ** argv) {
    require(argc == 2, "choose formats, dataset-handles, continuation, blueprint or package");
    const std::string group = argv[1];
    if (group == "formats") { selection_formats(); workflow_argument_budget_contracts(); }
    else if (group == "dataset-handles") { workflow_action_dataset_handles(); dataset_projection_contracts(); result_requirement_selection(); }
    else if (group == "continuation") { resource_argument_contracts(); resumed_resource_review_contract(); transitions_and_resume(); automatic_resume_capacity(); review_read_progress_contracts(); review_entry_contracts(); issued_cursor_contracts(); workflow_no_progress_contracts(); }
    else if (group == "blueprint") blueprint_and_publication_context();
    else if (group == "package") package_contracts();
    else require(false, "unknown contract group");
}
