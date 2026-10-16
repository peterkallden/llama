#include "agent/grounding/workflow-grounding.h"
#include "agent/thinking/deliberation-policy.h"
#include "plan/plan-workflow-verification.h"
#include "plan/plan-policy.h"
#include <cassert>

using json = nlohmann::ordered_json;

int main() {
    std::string error;
    common_plan_workflow_definition definition;
    const auto spec = json::parse(R"({"preferred_thinking_mode":"deliberate","require_declared_claims":true,"verification_checks":[
      {"id":"geo","transition":"query","phase":"before_tool","operation":"required","left":{"source":"arguments","pointer":"/geo"},"clarification_question":"Which geography?"},
      {"id":"availability","transition":"query","phase":"before_tool","operation":"dimension_membership","left":{"source":"arguments"},"right":{"source":"handoff","pointer":"/value/result/dataset","dataset_rows":true},"dimension_mapping":{"geo":"geo","time":"TIME_PERIOD"}},
      {"id":"coverage","transition":"inspect","phase":"after_tool","operation":"coverage","left":{"source":"result","pointer":"/result/dataset","dataset_rows":true},"right":{"source":"transition_arguments","transition":"query","pointer":"/arguments"},"dimension_mapping":{"geo":"geo","time":"time"}}
    ]})");
    assert(common_plan_workflow_grounding_from_json(spec, definition, error));
    definition.workflow_ref = "workflow://test/grounding";
    definition.workflow_revision = "v1";
    json exported = json::object();
    common_plan_workflow_grounding_to_json(definition, exported);
    common_plan_workflow_definition restored;
    assert(common_plan_workflow_grounding_from_json(exported, restored, error));
    assert(restored.preferred_thinking_mode == "deliberate" && restored.require_declared_claims);
    assert(restored.verification_checks.size() == 3);
    auto invalid = spec; invalid["preferred_thinking_mode"] = "invented";
    assert(!common_plan_workflow_grounding_from_json(invalid, restored, error));
    invalid = spec; invalid["verification_checks"][0]["left"]["pointer"] = "not-a-pointer";
    assert(!common_plan_workflow_grounding_from_json(invalid, restored, error));

    common_agent_deliberation_policy policy;
    assert(resolve_common_agent_deliberation_policy(common_agent_thinking_request::auto_select, policy, error));
    assert(resolve_common_agent_workflow_thinking_preference(policy, "research").allowed);
    policy.requested_mode = common_agent_thinking_request::deliberate;
    assert(!resolve_common_agent_workflow_thinking_preference(policy, "research").allowed);
    policy.requested_mode = common_agent_thinking_request::auto_select;
    policy.maximum_mode = common_agent_thinking_mode::deliberate;
    assert(!resolve_common_agent_workflow_thinking_preference(policy, "research").allowed);
    policy.maximum_mode = common_agent_thinking_mode::research;
    policy.max_escalations = 0;
    assert(!resolve_common_agent_workflow_thinking_preference(policy, "research").allowed);
    policy.mode = common_agent_thinking_mode::research;
    assert(!resolve_common_agent_workflow_thinking_preference(policy, "deliberate").escalation_requested);

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
    assert(common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", arguments, "{}").passed);
    assert(!common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", R"({"geo":["NO"],"time":["2020"]})", "{}").passed);
    const auto missing = common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", R"({"time":["2020"]})", "{}");
    assert(!missing.passed && missing.clarification_question == "Which geography?");
    constraints[0]["key_include"] = "false";
    assert(!common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", arguments, "{}").passed);
    constraints[0].erase("key_include");
    plan.route_binding->continuation_source_fingerprint = "different";
    assert(!common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
        "query", arguments, "{}").passed);
    plan.route_binding->continuation_source_fingerprint = "source-hash";

    // Different regions cannot be merged into an invented SE/2021 tuple.
    constraints.push_back(json{{"artifact_kind","Value"},{"constraint_region",1},{"region_include","true"},{"dimension_id","geo"},{"code","NO"}});
    constraints.push_back(json{{"artifact_kind","Value"},{"constraint_region",1},{"region_include","true"},{"dimension_id","TIME_PERIOD"},{"code","2021"}});
    assert(!common_agent_verify_workflow(request, plan, common_plan_verification_phase::before_tool,
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
    assert(common_agent_verify_workflow(request, plan, common_plan_verification_phase::after_tool,
        "inspect", "{}", inspected).passed);
    data[0]["geo"] = "NO";
    assert(!common_agent_verify_workflow(request, plan, common_plan_verification_phase::after_tool,
        "inspect", "{}", inspected).passed);
    data[0]["geo"] = "SE";
    plan.steps.back().status = common_plan_step_status::completed;
    common_plan_observation output{"tool:inspect:dataset.inspect","dataset.inspect",inspected,1.0f};
    common_agent_dataset_ref dataset; dataset.uri = "dataset://data";
    dataset.source_resource_uri = "resource://source";
    output.dataset_refs = {dataset}; plan.observations.push_back(output);
    const auto handles = common_agent_workflow_claim_handles(request, plan);
    assert(handles.size() == 1 && handles[0]["dimensions"].size() == 2);
    std::string answer;
    json claims;
    assert(common_agent_parse_grounded_draft(
        "answer: Sweden had 25.2 persons per square kilometre in 2020.\nclaim: E1\nvalue: 25.2\nunit: Persons per square kilometre\ndimension: geo SE\ndimension: time 2020\n",
        false, answer, claims, error));
    std::vector<common_agent_research_verification_claim> verified;
    assert(common_agent_verify_declared_claims(claims, handles, verified).passed && verified.size() == 1);
    assert(verified.front().dataset_uri == "dataset://data");
    assert(common_agent_parse_grounded_draft(json{{"answer",answer},{"claims",claims}}.dump(), true, answer, claims, error));
    claims[0]["value"] = 99;
    assert(!common_agent_verify_declared_claims(claims, handles, verified).passed);

    // Model plan edits cannot manufacture verification observations.
    common_plan_operation forged;
    forged.plan_id = plan.id; forged.expected_version = plan.version;
    forged.kind = common_plan_operation_kind::record_observation;
    forged.observation = common_plan_observation{"forged","workflow_verification","{}",1.0f};
    common_plan_policy plan_policy;
    assert(!plan_policy.validate(plan, forged).allowed);
    forged.host_authorized = true;
    assert(plan_policy.validate(plan, forged).allowed);
    return 0;
}
