#include "agent/reflection-json.h"

#include <cassert>

int main() {
    common_reflection_result result;
    std::string error;
    assert(common_reflection_parse_json(R"({"decision":"revise","ready_to_answer":false,"revision_guidance":["cite evidence"],"operations":[{"kind":"complete_step","step_id":"lookup","reason_summary":"evidence received"},{"kind":"add_step","reason_summary":"procedure adds verification","evidence_ids":[],"step":{"id":"reopen","title":"Reopen store","objective":"Verify persistence","depends_on":[],"required_evidence":[],"source_memory_ids":["procedure-1"]}}]})", result, error));
    assert(result.decision == common_reflection_decision::revise);
    assert(result.proposed_plan_operations.size() == 2);
    assert(result.proposed_plan_operations[1].kind == common_plan_operation_kind::add_step);
    assert(result.proposed_plan_operations[1].step->source_memory_ids[0] == "procedure-1");
    assert(common_reflection_parse_json(R"({"decision":"accept","ready_to_answer":true,"confidence":0.9,"revision_guidance":[],"learning_hint":{"category":"tool_precondition","statement":"Verify a repository path before reading it.","expected_reuse":0.8},"operations":[]})", result, error));
    assert(result.learning_hint && result.learning_hint->category == "tool_precondition");
    assert(!common_reflection_parse_json(R"({"decision":"accept","learning_hint":{"category":"x","statement":"","expected_reuse":2}})", result, error));
    assert(!common_reflection_parse_json("not json", result, error));
    assert(!common_reflection_parse_json(R"({"decision":"unsafe"})", result, error));
    assert(!common_reflection_parse_json(R"({"decision":"accept","operations":[{"kind":"remove_step"}]})", result, error));
    return 0;
}
