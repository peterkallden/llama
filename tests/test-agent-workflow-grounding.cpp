#include "agent/grounding/workflow-grounding.h"
#include "agent/thinking/deliberation-policy.h"
#include "plan/plan-workflow-verification.h"
#include "plan/plan-policy.h"
#include "plan/plan-workflow-state.h"

#include <cstdlib>
#include <iostream>

using json = nlohmann::ordered_json;

// Evaluate checks and setup in release builds as well.
#define CHECK_CONTRACT(condition) do { \
    if (!(condition)) { \
        std::cerr << "contract failed at line " << __LINE__ << ": " << #condition << '\n'; \
        std::exit(1); \
    } \
} while (false)

int main() {
    std::string error;
    common_plan_workflow_definition definition;
    const auto spec = json::parse(R"({"preferred_thinking_mode":"deliberate","require_declared_claims":true,"verification_checks":[
      {"id":"geo","transition":"query","phase":"before_tool","operation":"required","left":{"source":"arguments","pointer":"/geo"},"clarification_question":"Which geography?"},
      {"id":"availability","transition":"query","phase":"before_tool","operation":"dimension_membership","left":{"source":"arguments"},"right":{"source":"handoff","pointer":"/value/result/dataset","dataset_rows":true},"dimension_mapping":{"geo":"geo","time":"TIME_PERIOD"}},
      {"id":"coverage","transition":"inspect","phase":"after_tool","operation":"coverage","left":{"source":"result","pointer":"/result/dataset","dataset_rows":true},"right":{"source":"transition_arguments","transition":"query","pointer":"/arguments"},"dimension_mapping":{"geo":"geo","time":"time"}}
    ]})");
    CHECK_CONTRACT(common_plan_workflow_grounding_from_json(spec, definition, error));
    definition.workflow_ref = "workflow://test/grounding";
    definition.workflow_revision = "v1";
    json exported = json::object();
    common_plan_workflow_grounding_to_json(definition, exported);
    common_plan_workflow_definition restored;
    CHECK_CONTRACT(common_plan_workflow_grounding_from_json(exported, restored, error));
    CHECK_CONTRACT(restored.preferred_thinking_mode == "deliberate" && restored.require_declared_claims);
    CHECK_CONTRACT(restored.verification_checks.size() == 3);
    auto invalid = spec; invalid["preferred_thinking_mode"] = "invented";
    CHECK_CONTRACT(!common_plan_workflow_grounding_from_json(invalid, restored, error));
    invalid = spec; invalid["verification_checks"][0]["left"]["pointer"] = "not-a-pointer";
    CHECK_CONTRACT(!common_plan_workflow_grounding_from_json(invalid, restored, error));

    common_agent_deliberation_policy policy;
    CHECK_CONTRACT(resolve_common_agent_deliberation_policy(common_agent_thinking_request::auto_select, policy, error));
    CHECK_CONTRACT(resolve_common_agent_workflow_thinking_preference(policy, "research").allowed);
    policy.requested_mode = common_agent_thinking_request::deliberate;
    CHECK_CONTRACT(!resolve_common_agent_workflow_thinking_preference(policy, "research").allowed);
    policy.requested_mode = common_agent_thinking_request::auto_select;
    policy.maximum_mode = common_agent_thinking_mode::deliberate;
    CHECK_CONTRACT(!resolve_common_agent_workflow_thinking_preference(policy, "research").allowed);
    policy.maximum_mode = common_agent_thinking_mode::research;
    policy.max_escalations = 0;
    CHECK_CONTRACT(!resolve_common_agent_workflow_thinking_preference(policy, "research").allowed);
    policy.mode = common_agent_thinking_mode::research;
    CHECK_CONTRACT(!resolve_common_agent_workflow_thinking_preference(policy, "deliberate").escalation_requested);

    common_plan_state plan;
    plan.id = "grounding"; plan.version = 1;
    plan.workflow_definition = definition;
    plan.route_binding = common_plan_route_binding{};
    plan.route_binding->continuation_source_fingerprint = "source-hash";
    common_plan_step query;
    query.id = "query"; query.status = common_plan_step_status::active;
    query.semantic_alias = "workflow-transition:query";
    query.tool_call = common_plan_tool_call{"api.getData", "{}"};
    plan.steps = {query};
    common_plan_observation handoff;
    handoff.id = "workflow_handoff:source-hash:" + definition.workflow_ref;
    handoff.source = "workflow_handoff";
    handoff.summary = R"({"source_fingerprint":"source-hash","target_revision":"v1","value":{"result":{"dataset":"dataset://constraints"}}})";
    plan.observations.push_back(handoff);
    auto constraints = json::array({
        json{{"artifact_kind","Value"},{"constraint_region",0},{"region_include","true"},{"dimension_id","geo"},{"code","SE"}},
        json{{"artifact_kind","Value"},{"constraint_region",0},{"region_include","true"},{"dimension_id","TIME_PERIOD"},{"code","2020"}}
    });
    auto data = json::array({json{{"geo","SE"},{"geo_label","Sweden"},{"time","2020"},
        {"unit","PER_KM2"},{"unit_label","Persons per square kilometre"},{"value",25.2}}});
    common_agent_request request;
    request.grounding_dataset_reader = [&](const std::string & uri, json & rows, std::string & read_error) {
        read_error.clear();
        if (uri == "dataset://constraints") rows = constraints;
        else if (uri == "dataset://data") rows = data;
        else { read_error = "unavailable"; return false; }
        return true;
    };
    const auto arguments = R"({"geo":["SE"],"time":["2020"]})";
    CHECK_CONTRACT(common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", arguments, "{}").passed);
    CHECK_CONTRACT(!common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", R"({"geo":["NO"],"time":["2020"]})", "{}").passed);
    const auto missing = common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", R"({"time":["2020"]})", "{}");
    CHECK_CONTRACT(!missing.passed && missing.clarification_question == "Which geography?");
    constraints[0]["key_include"] = "false";
    CHECK_CONTRACT(!common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", arguments, "{}").passed);
    constraints[0].erase("key_include");
    plan.route_binding->continuation_source_fingerprint = "different";
    CHECK_CONTRACT(!common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", arguments, "{}").passed);
    plan.route_binding->continuation_source_fingerprint = "source-hash";

    // Different regions cannot be merged into an invented SE/2021 tuple.
    constraints.push_back(json{{"artifact_kind","Value"},{"constraint_region",1},{"region_include","true"},{"dimension_id","geo"},{"code","NO"}});
    constraints.push_back(json{{"artifact_kind","Value"},{"constraint_region",1},{"region_include","true"},{"dimension_id","TIME_PERIOD"},{"code","2021"}});
    CHECK_CONTRACT(!common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", R"({"geo":["SE"],"time":["2021"]})", "{}").passed);

    plan.steps.front().status = common_plan_step_status::completed;
    plan.observations.push_back({"tool:query:api.getData", "api.getData", "{}", 1.0f});
    plan.observations.push_back({"workflow_arguments:query", "workflow_arguments",
        json{{"arguments",json::parse(arguments)}}.dump(), 1.0f});
    common_plan_step inspect;
    inspect.id = "inspect"; inspect.status = common_plan_step_status::active;
    inspect.semantic_alias = "workflow-transition:inspect";
    inspect.tool_call = common_plan_tool_call{"dataset.inspect","{}"};
    plan.steps.push_back(inspect);
    const auto inspected = R"({"result":{"dataset":"dataset://data"}})";
    CHECK_CONTRACT(common_agent_verify_workflow(request, plan, common_plan_verification_phase::after_tool,
        "inspect", "{}", inspected).passed);
    data[0]["geo"] = "NO";
    CHECK_CONTRACT(!common_agent_verify_workflow(request, plan, common_plan_verification_phase::after_tool,
        "inspect", "{}", inspected).passed);
    data[0]["geo"] = "SE";
    plan.steps.back().status = common_plan_step_status::completed;
    common_plan_observation output{"tool:inspect:dataset.inspect","dataset.inspect",inspected,1.0f};
    common_agent_dataset_ref dataset; dataset.uri = "dataset://data";
    dataset.source_resource_uri = "resource://source";
    output.dataset_refs = {dataset}; plan.observations.push_back(output);
    const auto handles = common_agent_workflow_claim_handles(request, plan);
    CHECK_CONTRACT(handles.size() == 1 && handles[0]["dimensions"].size() == 2);
    std::string answer;
    json claims;
    CHECK_CONTRACT(common_agent_parse_grounded_draft(
        "answer: Sweden had 25.2 persons per square kilometre in 2020.\nclaim: E1\nvalue: 25.2\nunit: Persons per square kilometre\ndimension: geo SE\ndimension: time 2020\n",
        false, answer, claims, error));
    std::vector<common_agent_research_verification_claim> verified;
    CHECK_CONTRACT(common_agent_verify_declared_claims(claims, handles, verified).passed && verified.size() == 1);
    CHECK_CONTRACT(verified.front().dataset_uri == "dataset://data");
    CHECK_CONTRACT(common_agent_parse_grounded_draft(json{{"answer",answer},{"claims",claims}}.dump(), true, answer, claims, error));
    claims[0]["value"] = 99;
    CHECK_CONTRACT(!common_agent_verify_declared_claims(claims, handles, verified).passed);

    // Model plan edits cannot manufacture verification observations.
    common_plan_operation forged;
    forged.plan_id = plan.id; forged.expected_version = plan.version;
    forged.kind = common_plan_operation_kind::record_observation;
    forged.observation = common_plan_observation{"forged","workflow_verification","{}",1.0f};
    common_plan_policy plan_policy;
    CHECK_CONTRACT(!plan_policy.validate(plan, forged).allowed);
    forged.host_authorized = true;
    CHECK_CONTRACT(plan_policy.validate(plan, forged).allowed);
    const auto require = [](bool condition, const char * message) {
        if (!condition) { std::cerr << message << '\n'; std::exit(1); }
    };
    forged.host_authorized = false;
    forged.observation->source = "response_review";
    require(!plan_policy.validate(plan, forged).allowed,
        "model-created review evidence could clear a rejected response");
    common_plan_workflow_definition flexible;
    require(common_plan_workflow_grounding_from_json(json::parse(R"({
      "declared_claim_tools":["statistics.describe"],
      "verification_checks":[{"id":"selected","tool":"statistics.describe","phase":"after_tool","operation":"required",
        "left":{"source":"result","pointer":"/result/selected_columns"}}]
    })"), flexible, error), "one-action operation selector was rejected");
    json flexible_json = json::object();
    common_plan_workflow_grounding_to_json(flexible, flexible_json);
    common_plan_workflow_definition flexible_restored;
    require(common_plan_workflow_grounding_from_json(flexible_json, flexible_restored, error) &&
        flexible_restored.declared_claim_tools == flexible.declared_claim_tools &&
        flexible_restored.verification_checks[0].tool_name == "statistics.describe",
        "operation verification or conditional claim assurance was lost on roundtrip");
    common_plan_state analyzed;
    analyzed.workflow_definition = flexible_restored;
    common_plan_step described;
    described.id = "describe";
    described.semantic_alias = "execute-operation";
    described.status = common_plan_step_status::completed;
    described.tool_call = common_plan_tool_call{"statistics.describe", R"({"dataset":"dataset://orders","columns":["amount"]})"};
    analyzed.steps = {described};
    const auto raw = R"({"ok":true,"result":{"dataset":"dataset://orders","selected_columns":["amount"],"columns":[{"name":"amount","count":3,"mean":13.333333333333334}],"scan_truncated":false}})";
    analyzed.observations.push_back({"tool:describe:statistics.describe", "statistics.describe", raw, 1.0f});
    require(common_agent_verify_workflow(request, analyzed, common_plan_verification_phase::after_tool,
        described.id, described.tool_call->arguments_json, raw).passed,
        "legacy one-action workflow bypassed or rejected a valid result check");
    require(!common_agent_verify_workflow(request, analyzed, common_plan_verification_phase::after_tool,
        described.id, described.tool_call->arguments_json, R"({"result":{"selected_columns":[]}})").passed,
        "missing applicable output passed a required operation condition");
    require(common_plan_workflow_observation(analyzed, "execute-operation") != nullptr,
        "stable legacy action alias could not resolve its observation");
    require(common_agent_workflow_requires_claims(analyzed), "declared numeric operation lost its claim requirement");
    const auto fields = common_agent_workflow_claim_handles({}, analyzed);
    require(fields.size() == 2 && fields[0]["kind"] == "result_field" &&
        !fields[0].contains("unit") && !fields[0].contains("dimensions"),
        "inline results invented a dataset-cell unit or required a dataset reader");
    require(common_agent_parse_grounded_draft("answer: Three observations.\nclaim: E1\nvalue: 3", false, answer, claims, error) &&
        common_agent_verify_declared_claims(claims, fields, verified).passed &&
        verified[0].source_location == "/result/columns/0/count",
        "inline scalar claim lost its exact evidence position");
    claims[0]["value"] = 0;
    require(!common_agent_verify_declared_claims(claims, fields, verified).passed,
        "a fabricated zero matched the executed count");
    claims[0]["value"] = 3;
    claims[0]["unit"] = "invented";
    require(!common_agent_verify_declared_claims(claims, fields, verified).passed,
        "result-field verification invented a measurement unit");
    require(common_agent_grounded_draft_grammar(fields).find("claim: E1") != std::string::npos,
        "inline scalar handles did not reach the compact sampling contract");
    analyzed.observations.back().summary = R"({"result":{"dataset":"dataset://orders","columns":[{"name":"amount","count":4,"mean":null}]}})";
    const auto changed_fields = common_agent_workflow_claim_handles({}, analyzed);
    require(changed_fields.size() == 1 && changed_fields[0]["value"] == 4,
        "null statistic became a numerical zero or stale result survived");
    claims[0].erase("unit");
    require(!common_agent_verify_declared_claims(claims, changed_fields, verified).passed,
        "old scalar claim matched a different evidence generation");

    common_plan_step optional_read;
    optional_read.id = "metadata";
    optional_read.status = common_plan_step_status::completed;
    optional_read.tool_call = common_plan_tool_call{"dataset.inspect", "{}"};
    analyzed.steps.push_back(optional_read);
    analyzed.observations.push_back({"tool:metadata:dataset.inspect", "dataset.inspect", R"({"result":{"name":"orders"}})", 1.0f});
    require(common_agent_workflow_requires_claims(analyzed) &&
        common_agent_workflow_claim_handles({}, analyzed)[0]["value"] == 4,
        "an unrelated successful read silently removed the declared evidence requirement");

    common_plan_state document_plan;
    common_plan_workflow_definition document_workflow;
    require(common_plan_workflow_grounding_from_json(json::parse(R"({"verification_checks":[
      {"id":"document","tool":"document.open","phase":"after_tool","operation":"required",
       "left":{"source":"result","pointer":"/result/document_id"}}
    ]})"), document_workflow, error), "document result declaration was rejected");
    document_plan.workflow_definition = document_workflow;
    require(common_agent_verify_workflow({}, document_plan, common_plan_verification_phase::after_tool,
        "open", "{}", R"({"result":{"document_id":"doc-1"}})", "document.open").passed &&
        !common_agent_verify_workflow({}, document_plan, common_plan_verification_phase::after_tool,
        "open", "{}", R"({"result":{}})", "document.open").passed &&
        !common_agent_workflow_requires_claims(document_plan),
        "generic operation verification assumed a dataset or mandatory numerical answer");
    return 0;
}
