#include "agent/thinking/reflection-json.h"

#include <cassert>
#include <nlohmann/json.hpp>

namespace {

bool same_json_object(const std::string & lhs, const std::string & rhs) {
    return nlohmann::json::parse(lhs, nullptr, false) ==
           nlohmann::json::parse(rhs, nullptr, false);
}

} // namespace

int main() {
    common_reflection_result result;
    std::string error;
    assert(common_reflection_parse_json(
        R"({"decision":"revise","assurance_action":"escalate_research","issues":[]})",
        result, error));
    assert(result.next_action == common_agent_reflection_next_action::escalate_research);
    assert(common_reflection_parse_json(R"({"decision":"revise","ready_to_answer":false,"revision_guidance":["cite evidence"],"operations":[{"kind":"complete_step","step_id":"lookup","reason_summary":"evidence received"},{"kind":"add_step","reason_summary":"procedure adds verification","evidence_ids":[],"step":{"id":"reopen","title":"Reopen store","objective":"Verify persistence","depends_on":[],"required_evidence":[],"source_memory_ids":["procedure-1"]}}]})", result, error));
    assert(result.decision == common_reflection_decision::revise);
    assert(result.proposed_plan_operations.size() == 2);
    assert(result.proposed_plan_operations[1].kind == common_plan_operation_kind::add_step);
    assert(result.proposed_plan_operations[1].step->source_memory_ids[0] == "procedure-1");
    assert(common_reflection_parse_json(R"({"decision":"revise","ready_to_answer":false,"confidence":0.7,"revision_guidance":["repair the failed branch"],"complete":["lookup"],"activate":["retry"],"retry":["retry-2"],"reset":["fetch"],"next_action":"re-run verification","add_steps":[{"tool":"repository.search","args":{"query":"planner"}},{"mode":"reasoning","source_memory_ids":["procedure-1"]}],"replace_steps":[{"step_id":"fetch","title":"Retry fetch","objective":"Retry with corrected parameters","tool":"repository.search","args":{"query":"planner repair"}}]})", result, error));
    assert(result.proposed_plan_operations.size() == 8);
    assert(result.proposed_plan_operations[0].kind == common_plan_operation_kind::complete_step);
    assert(*result.proposed_plan_operations[0].step_id == "lookup");
    assert(result.proposed_plan_operations[1].kind == common_plan_operation_kind::activate_step);
    assert(*result.proposed_plan_operations[1].step_id == "retry");
    assert(result.proposed_plan_operations[2].kind == common_plan_operation_kind::activate_step);
    assert(*result.proposed_plan_operations[2].step_id == "retry-2");
    assert(result.proposed_plan_operations[3].kind == common_plan_operation_kind::reset_step);
    assert(*result.proposed_plan_operations[3].step_id == "fetch");
    assert(result.proposed_plan_operations[4].kind == common_plan_operation_kind::set_next_action);
    assert(*result.proposed_plan_operations[4].value == "re-run verification");
    assert(result.proposed_plan_operations[5].kind == common_plan_operation_kind::add_step);
    assert(result.proposed_plan_operations[5].step->id == "repair_1");
    assert(result.proposed_plan_operations[5].step->tool_call && result.proposed_plan_operations[5].step->tool_call->name == "repository.search");
    assert(result.proposed_plan_operations[6].kind == common_plan_operation_kind::add_step);
    assert(result.proposed_plan_operations[6].step->id == "repair_2");
    assert(result.proposed_plan_operations[6].step->depends_on == std::vector<std::string>{"repair_1"});
    assert(result.proposed_plan_operations[7].kind == common_plan_operation_kind::replace_step);
    assert(*result.proposed_plan_operations[7].step_id == "fetch");
    assert(result.proposed_plan_operations[7].step->tool_call && result.proposed_plan_operations[7].step->tool_call->name == "repository.search");
    assert(common_reflection_parse_json(R"({"decision":"revise","operations":[{"kind":"add_constraint","reason_summary":"preserve the repository boundary","constraint":{"id":"repo-boundary","description":"Do not modify files outside the repository.","hard":true}},{"kind":"add_assumption","reason_summary":"record the working assumption","assumption":{"id":"local-model","statement":"The configured model is available locally.","confidence":0.8,"evidence_ids":["model-check"]}},{"kind":"invalidate_assumption","reason_summary":"assumption no longer holds","target_id":"old-model"}]})", result, error));
    assert(result.proposed_plan_operations.size() == 3);
    assert(result.proposed_plan_operations[0].kind == common_plan_operation_kind::add_constraint);
    assert(result.proposed_plan_operations[0].constraint && result.proposed_plan_operations[0].constraint->hard);
    assert(result.proposed_plan_operations[1].kind == common_plan_operation_kind::add_assumption);
    assert(result.proposed_plan_operations[1].assumption && result.proposed_plan_operations[1].assumption->confidence == 0.8f);
    assert(result.proposed_plan_operations[2].kind == common_plan_operation_kind::invalidate_assumption);
    assert(result.proposed_plan_operations[2].target_id && *result.proposed_plan_operations[2].target_id == "old-model");
    assert(common_reflection_parse_json(R"({"decision":"revise","add_steps":[{"tool":"memory_search","args":{"tool":{"name":"memory_search","arguments":{"query":"regression procedure","limit":"2"}}}}],"replace_steps":[{"step_id":"fetch","tool":"memory_search","args":{"tool":"memory_search","arguments":{"query":"corrected lookup","limit":"3"}}}]})", result, error));
    assert(result.proposed_plan_operations.size() == 2);
    assert(same_json_object(result.proposed_plan_operations[0].step->tool_call->arguments_json, R"({"limit":2,"query":"regression procedure"})"));
    assert(same_json_object(result.proposed_plan_operations[1].step->tool_call->arguments_json, R"({"limit":3,"query":"corrected lookup"})"));
    assert(common_reflection_parse_json(R"({"decision":"revise","add_steps":[{"tool":"memory_search","args":{"tool":"memory_search","query":"regression procedure","limit":"2"}}],"replace_steps":[{"step_id":"fetch","tool":"memory_search","args":{"tool":"memory_search","query":"corrected lookup","limit":"3"}}]})", result, error));
    assert(result.proposed_plan_operations.size() == 2);
    assert(same_json_object(result.proposed_plan_operations[0].step->tool_call->arguments_json, R"({"limit":2,"query":"regression procedure"})"));
    assert(same_json_object(result.proposed_plan_operations[1].step->tool_call->arguments_json, R"({"limit":3,"query":"corrected lookup"})"));
    assert(common_reflection_parse_json(R"({"decision":"revise","add_steps":[{"tool":"memory_get","args":"memory-1"}],"replace_steps":[]})", result, error));
    assert(result.proposed_plan_operations.size() == 1);
    assert(same_json_object(result.proposed_plan_operations[0].step->tool_call->arguments_json, R"({"id":"memory-1"})"));
    assert(common_reflection_parse_json(R"({"decision":"accept","ready_to_answer":true,"confidence":0.9,"revision_guidance":[],"learning_hint":{"category":"tool_precondition","statement":"Verify a repository path before reading it.","expected_reuse":0.8},"operations":[]})", result, error));
    assert(result.learning_hint && result.learning_hint->category == "tool_precondition");
    const std::string explicit_tool_call_reflection =
        "{\"decision\":\"revise\",\"operations\":[{\"kind\":\"add_step\",\"step\":"
        "{\"id\":\"repair-fetch\",\"title\":\"Repair fetch\",\"objective\":\"Retry the fetch with normalized args\","
        "\"tool_call\":{\"name\":\"memory_search\",\"arguments_json\":\"{\\\"tool\\\":\\\"memory_search\\\",\\\"query\\\":\\\"reopened evidence\\\",\\\"limit\\\":\\\"4\\\"}\"}}}]}";
    assert(common_reflection_parse_json(explicit_tool_call_reflection, result, error));
    assert(result.proposed_plan_operations.size() == 1);
    assert(result.proposed_plan_operations[0].step->tool_call);
    assert(same_json_object(result.proposed_plan_operations[0].step->tool_call->arguments_json, R"({"limit":4,"query":"reopened evidence"})"));
    assert(common_reflection_parse_json(R"({"decision":"revise","add_steps":[{"mode":"reasoning"},{"mode":"reasoning"}]})", result, error));
    assert(result.proposed_plan_operations.size() == 2);
    assert(result.proposed_plan_operations[0].step->id == "repair_1");
    assert(result.proposed_plan_operations[1].step->id == "repair_2");
    assert(result.proposed_plan_operations[1].step->depends_on == std::vector<std::string>{"repair_1"});
    assert(!common_reflection_parse_json(R"({"decision":"accept","learning_hint":{"category":"x","statement":"","expected_reuse":2}})", result, error));
    assert(!common_reflection_parse_json("not json", result, error));
    assert(!common_reflection_parse_json(R"({"decision":"unsafe"})", result, error));
    assert(!common_reflection_parse_json(R"({"decision":"accept","operations":[{"kind":"remove_step"}]})", result, error));
    assert(!common_reflection_parse_json(R"({"decision":"revise","operations":[{"kind":"add_assumption","assumption":{"id":"a","statement":"x","confidence":2}}]})", result, error));
    assert(!common_reflection_parse_json(R"({"decision":"revise","operations":[{"kind":"invalidate_assumption"}]})", result, error));
    assert(!common_reflection_parse_json(R"({"decision":"revise","complete":"lookup"})", result, error));
    return 0;
}
