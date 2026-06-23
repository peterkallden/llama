#include "plan/plan-json.h"

#include <cassert>

int main() {
    common_plan_state plan;
    plan.id = "p";
    std::vector<common_plan_operation> operations;
    std::string error;

    const auto compact = R"({"goal":"inspect bindings","steps":[{"id":"search","tool":"repository_search","args":{"query":"plan bindings"}},{"id":"read","tool":{"name":"repository_read","arguments":{"path":{"$from_step":"search","$json_pointer":"/matches/0/path"}}},"after":"search"},{"id":"answer","mode":"final","after":"read"}]})";
    assert(common_plan_parse_proposal_json(compact, plan, operations, error));
    assert(operations.size() == 3);
    assert(operations[0].step->tool_call->arguments_json == R"({"query":"plan bindings"})");
    assert(operations[1].step->depends_on == std::vector<std::string>{"search"});
    assert(common_plan_step_effective_mode(*operations[2].step) == common_plan_step_mode::final_response);

    const auto normalized = R"({"goal":"calculate","steps":[{"id":"calculate","tool":"calculator","args":{"operation":"multiply","operands":[{"value":17},{"value":23}]}}]})";
    assert(common_plan_parse_proposal_json(normalized, plan, operations, error));
    assert(operations.size() == 2); // native final synthesis
    assert(operations[0].step->tool_call->arguments_json == R"({"expression":"17 * 23"})");
    assert(operations[1].step->id == "answer");

    // The prior full proposal format remains accepted for persisted or older callers.
    const auto legacy = R"({"goal":"answer","success_criteria":"clear","next_action":"draft","operations":[{"kind":"add_step","reason_summary":"tool use","evidence_ids":[],"step":{"id":"s1","title":"Calc","objective":"compute","depends_on":[],"required_evidence":[],"tool":{"name":"calculator","arguments_json":"{'operation':'multiply','operands':[{'value':17},{'value':23}]}"}}}]})";
    assert(common_plan_parse_proposal_json(legacy, plan, operations, error));
    assert(operations[0].step->tool_call->arguments_json == R"({"expression":"17 * 23"})");
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","steps":[]})", plan, operations, error));
    assert(common_plan_proposal_json_schema().find("arguments_json") == std::string::npos);
    return 0;
}
