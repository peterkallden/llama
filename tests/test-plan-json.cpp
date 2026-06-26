#include "plan/plan-json.h"
#include "plan/plan-context.h"

#include <cassert>

int main() {
    common_plan_state plan;
    plan.id = "p";
    std::vector<common_plan_operation> operations;
    std::string error;

    const auto compact = R"({"purpose":"inspect the current implementation","goal":"inspect bindings","steps":[{"id":"search","contribution":"find the relevant implementation","tool":"repository_search","args":{"query":"plan bindings"}},{"id":"read","tool":{"name":"repository_read","arguments":{"path":{"$from_step":"search","$json_pointer":"/matches/0/path"}}},"after":"search"},{"id":"answer","mode":"final","after":"read"}]})";
    assert(common_plan_parse_proposal_json(compact, plan, operations, error));
    assert(operations.size() == 3);
    assert(operations[0].step->tool_call->arguments_json == R"({"query":"plan bindings"})");
    assert(plan.purpose == "inspect the current implementation");
    assert(operations[0].step->intended_contribution == "find the relevant implementation");
    assert(operations[1].step->depends_on == std::vector<std::string>{"search"});
    assert(common_plan_step_effective_mode(*operations[2].step) == common_plan_step_mode::final_response);

    const auto normalized = R"({"goal":"calculate","steps":[{"id":"calculate","tool":"calculator","args":{"operation":"multiply","operands":[{"value":17},{"value":23}]}}]})";
    assert(common_plan_parse_proposal_json(normalized, plan, operations, error));
    assert(operations.size() == 2); // native final synthesis
    assert(operations[0].step->tool_call->arguments_json == R"({"expression":"17 * 23"})");
    assert(operations[1].step->id == "answer");

    const auto repaired_integer = R"({"goal":"inspect","steps":[{"id":"search","tool":"repository_search","args":{"query":"plan","max_results":"16"}}]})";
    assert(common_plan_parse_proposal_json(repaired_integer, plan, operations, error));
    assert(operations[0].step->tool_call->arguments_json == R"({"max_results":16,"query":"plan"})");

    const auto wrapped_tool_arguments = R"({"goal":"inspect memory","steps":[{"id":"search","tool":"memory_search","args":{"tool":{"name":"memory_search","arguments":{"query":"regression procedure","limit":"2"}}}}]})";
    assert(common_plan_parse_proposal_json(wrapped_tool_arguments, plan, operations, error));
    assert(operations[0].step->tool_call->arguments_json == R"({"limit":2,"query":"regression procedure"})");

    const auto compact_without_ids = R"({"goal":"inspect","steps":[{"tool":"repository_search","args":{"query":"planner"}},{"tool":{"name":"repository_read","arguments":{"path":{"$from_step":"step_1","$json_pointer":"/matches/0/path"}}}},{"mode":"reasoning"}]})";
    assert(common_plan_parse_proposal_json(compact_without_ids, plan, operations, error));
    assert(operations.size() == 4); // native final synthesis
    assert(operations[0].step->id == "step_1");
    assert(operations[1].step->id == "step_2");
    assert(operations[2].step->id == "step_3");
    assert(operations[1].step->depends_on == std::vector<std::string>{"step_1"});
    assert(operations[2].step->depends_on == std::vector<std::string>{"step_2"});
    assert(operations[3].step->id == "answer");
    assert(operations[3].step->depends_on == std::vector<std::string>({"step_1", "step_2", "step_3"}));

    // The prior full proposal format remains accepted for persisted or older callers.
    const auto legacy = R"({"goal":"answer","success_criteria":"clear","next_action":"draft","operations":[{"kind":"add_step","reason_summary":"tool use","evidence_ids":[],"step":{"id":"s1","title":"Calc","objective":"compute","depends_on":[],"required_evidence":[],"tool":{"name":"calculator","arguments_json":"{'operation':'multiply','operands':[{'value':17},{'value':23}]}"}}}]})";
    assert(common_plan_parse_proposal_json(legacy, plan, operations, error));
    assert(operations[0].step->tool_call->arguments_json == R"({"expression":"17 * 23"})");
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","steps":[]})", plan, operations, error));
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","steps":[{"id":"invalid","mode":"tool"}]})", plan, operations, error));
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","steps":[{"id":"search","tool":"repository_search","args":{"query":"x"}},{"id":"search","mode":"final"}]})", plan, operations, error));
    assert(error == "duplicate step id");
    assert(!common_plan_parse_proposal_json(R"({"goal":"x","steps":[{"id":"answer","tool":"calculator","args":{"expression":"6 * 7"}}]})", plan, operations, error));
    assert(error == "native final step id conflicts with proposed step");
    assert(common_plan_proposal_json_schema().find("arguments_json") == std::string::npos);
    assert(common_plan_proposal_json_schema().find(R"("required":["id"])") == std::string::npos);
    assert(common_plan_proposal_json_schema().find("oneOf") == std::string::npos);

    common_plan_state context_plan;
    context_plan.goal = "Use only relevant evidence";
    common_plan_step search{"search", "Search", "Find evidence"};
    search.result_summary = "matching file found";
    context_plan.steps.push_back(search);
    common_plan_step reasoning{"reason", "Reason", "Use the selected search result."};
    reasoning.depends_on = {"search"};
    context_plan.steps.push_back(reasoning);
    context_plan.observations.push_back({"tool:search", "tool:search", "matching file found", 1.0f, {}, 0});
    context_plan.observations.push_back({"tool:other", "tool:other", "unrelated result", 1.0f, {}, 0});
    const auto step_context = common_plan_render_step_context(context_plan, context_plan.steps.back());
    assert(step_context.find("matching file found") != std::string::npos);
    assert(step_context.find("unrelated result") == std::string::npos);
    return 0;
}
