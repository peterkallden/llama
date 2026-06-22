#include "plan/plan-json.h"

#include <cassert>

int main() {
    common_plan_state plan;
    plan.id = "p";
    std::vector<common_plan_operation> operations;
    std::string error;
    const auto input = R"({"goal":"answer","success_criteria":"clear","next_action":"draft","operations":[{"kind":"add_step","reason_summary":"procedure evidence","evidence_ids":[],"step":{"id":"s1","title":"Draft","objective":"answer","depends_on":[],"required_evidence":[],"source_memory_ids":["procedure-1"],"tool":{"name":"web_search","arguments_json":"{"query":"llama.cpp"}"}}}]})";
    assert(common_plan_parse_proposal_json(input, plan, operations, error));
    assert(operations.size() == 1 && !operations[0].step->generated_from_memory);
    assert(operations[0].step->source_memory_ids == std::vector<std::string>{"procedure-1"});
    assert(operations[0].step->tool_call && operations[0].step->tool_call->name == "web_search");
    const auto relaxed = R"({"goal":"answer","success_criteria":"clear","next_action":"draft","operations":[{"kind":"add_step","reason_summary":"tool use","evidence_ids":[],"step":{"id":"s1","title":"Calc","objective":"compute","depends_on":[],"required_evidence":[],"tool":{"name":"calculator","arguments_json":"{'operation':'multiply','operands':[{'value':17},{'value':23}]}"}}]})";
    assert(common_plan_parse_proposal_json(relaxed, plan, operations, error));
    assert(operations[0].step->tool_call->arguments_json == R"({"expression":"17 * 23"})");
    const auto multi_step = R"({"goal":"answer","success_criteria":"verified","next_action":"calculate","operations":[{"kind":"add_step","reason_summary":"calculate","evidence_ids":[],"step":{"id":"calculate","title":"Calculate","objective":"calculate","depends_on":[],"required_evidence":[],"tool":{"name":"calculator","arguments_json":"{\"expression\":\"17 * 23\"}"}}},{"kind":"add_step","reason_summary":"answer","evidence_ids":[],"step":{"id":"answer","title":"Answer","objective":"answer","depends_on":["calculate"],"required_evidence":[]}}]})";
    assert(common_plan_parse_proposal_json(multi_step, plan, operations, error, 6));
    assert(operations.size() == 2 && operations[1].step->depends_on == std::vector<std::string>{"calculate"});
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","operations":[]})", plan, operations, error));
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","success_criteria":"y","next_action":"z","operations":[{"kind":"complete_plan"}]})", plan, operations, error));
    assert(common_plan_proposal_json_schema().find("source_memory_ids") != std::string::npos);
    return 0;
}
