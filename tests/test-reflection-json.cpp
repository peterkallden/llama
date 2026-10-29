#include "agent/thinking/reflection-json.h"


#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {

bool same_json_object(const std::string & lhs, const std::string & rhs) {
    return nlohmann::json::parse(lhs, nullptr, false) ==
           nlohmann::json::parse(rhs, nullptr, false);
}

} // namespace

// Evaluate checks and setup in release builds as well.
#define CHECK_CONTRACT(condition) do { \
    if (!(condition)) { \
        std::cerr << "contract failed at line " << __LINE__ << ": " << #condition << '\n'; \
        std::exit(1); \
    } \
} while (false)

int main() {
    common_reflection_result result;
    std::string error;
    const std::string host_step = "tool:agent-blueprint:" + std::string(96, 'a') + ":workflow:action:investigate.source.attach";
    if (!common_reflection_parse_compact_dsl("reflect retry", result, error, {}, 8, {}, host_step) ||
            result.proposed_plan_operations.size() != 1 || result.proposed_plan_operations.front().step_id != host_step) {
        std::cerr << "host-bound long retry identity failed: " << error << '\n'; return 1;
    }
    if (!common_reflection_parse_compact_dsl("reflect retry A", result, error, {}, 8, {}, {}, {}, {{"A", host_step}}) ||
            result.proposed_plan_operations.front().step_id != host_step ||
            common_reflection_parse_compact_dsl("reflect retry B", result, error, {}, 8, {}, {}, {}, {{"A", host_step}}) ||
            common_reflection_parse_compact_dsl("reflect retry " + host_step, result, error) ||
            common_reflection_parse_json(nlohmann::json{{"decision", "revise"}, {"retry", {host_step}}}.dump(), result, error)) {
        std::cerr << "host retry binding changed model ID validation" << '\n'; return 1;
    }
    if (!common_reflection_parse_compact_dsl("reflect retry\nactivate host-retry-1", result, error, {}, 8, {}, host_step) ||
            result.proposed_plan_operations.size() != 2 || result.proposed_plan_operations[0].step_id != "host-retry-1" ||
            result.proposed_plan_operations[1].step_id != host_step) {
        std::cerr << "host retry alias rewrote a model-written reference" << '\n'; return 1;
    }
    CHECK_CONTRACT(common_reflection_parse_compact_dsl("reflect accept", result, error));
    CHECK_CONTRACT(result.decision == common_reflection_decision::accept);
    CHECK_CONTRACT(result.proposed_plan_operations.empty());
    CHECK_CONTRACT(common_reflection_parse_compact_dsl(
        "reflect revise\nguidance \"count is missing\"", result, error));
    CHECK_CONTRACT(result.decision == common_reflection_decision::revise);
    CHECK_CONTRACT(result.revision_guidance == std::vector<std::string>{"count is missing"});
    CHECK_CONTRACT(common_reflection_parse_compact_dsl(
        "reflect retry", result, error, {}, 8, {}, "failed-search"));
    CHECK_CONTRACT(result.decision == common_reflection_decision::revise);
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 1);
    CHECK_CONTRACT(*result.proposed_plan_operations[0].step_id == "failed-search");
    const std::vector<std::pair<std::string, std::string>> retry_choices = {
        {"A", "failed-search"}, {"B", "failed-fetch"},
    };
    CHECK_CONTRACT(common_reflection_parse_compact_dsl(
        "reflect retry B", result, error, {}, 8, {}, {}, {}, retry_choices));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 1);
    CHECK_CONTRACT(*result.proposed_plan_operations[0].step_id == "failed-fetch");
    CHECK_CONTRACT(!common_reflection_parse_compact_dsl(
        "reflect retry C", result, error, {}, 8, {}, {}, {}, retry_choices));

    const common_chat_tool memory_search_tool{
        "memory.search", "Search memory.",
        R"({"type":"object","additionalProperties":false,"required":["query"],"properties":{"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":10}}})",
        R"({"type":"object"})",
    };
    CHECK_CONTRACT(common_reflection_parse_compact_dsl(
        "reflect revise\nquery: \"corrected lookup\"\nlimit: 3\n",
        result, error, "failed-search", 8, "memory.search", {}, {memory_search_tool}));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 1);
    CHECK_CONTRACT(result.proposed_plan_operations[0].step->tool_call->name == "memory.search");
    CHECK_CONTRACT(*result.proposed_plan_operations[0].step_id == "failed-search");
    CHECK_CONTRACT(same_json_object(result.proposed_plan_operations[0].step->tool_call->arguments_json,
        R"({"query":"corrected lookup","limit":3})"));
    CHECK_CONTRACT(!common_reflection_parse_compact_dsl(
        "reflect revise\nunknown: value\n",
        result, error, "failed-search", 8, "memory.search", {}, {memory_search_tool}));

    CHECK_CONTRACT(common_reflection_parse_json(
        R"({"decision":"revise","assurance_action":"escalate_research","issues":[]})",
        result, error));
    CHECK_CONTRACT(result.next_action == common_agent_reflection_next_action::escalate_research);
    CHECK_CONTRACT(common_reflection_parse_json(R"({"decision":"revise","ready_to_answer":false,"revision_guidance":["cite evidence"],"operations":[{"kind":"complete_step","step_id":"lookup","reason_summary":"evidence received"},{"kind":"add_step","reason_summary":"procedure adds verification","evidence_ids":[],"step":{"id":"reopen","title":"Reopen store","objective":"Verify persistence","depends_on":[],"required_evidence":[],"source_memory_ids":["procedure-1"]}}]})", result, error));
    CHECK_CONTRACT(result.decision == common_reflection_decision::revise);
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 2);
    CHECK_CONTRACT(result.proposed_plan_operations[1].kind == common_plan_operation_kind::add_step);
    CHECK_CONTRACT(result.proposed_plan_operations[1].step->source_memory_ids[0] == "procedure-1");
    CHECK_CONTRACT(common_reflection_parse_json(R"({"decision":"revise","ready_to_answer":false,"confidence":0.7,"revision_guidance":["repair the failed branch"],"complete":["lookup"],"activate":["retry"],"retry":["retry-2"],"reset":["fetch"],"next_action":"re-run verification","add_steps":[{"tool":"repository.search","args":{"query":"planner"}},{"mode":"reasoning","source_memory_ids":["procedure-1"]}],"replace_steps":[{"step_id":"fetch","title":"Retry fetch","objective":"Retry with corrected parameters","tool":"repository.search","args":{"query":"planner repair"}}]})", result, error));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 8);
    CHECK_CONTRACT(result.proposed_plan_operations[0].kind == common_plan_operation_kind::complete_step);
    CHECK_CONTRACT(*result.proposed_plan_operations[0].step_id == "lookup");
    CHECK_CONTRACT(result.proposed_plan_operations[1].kind == common_plan_operation_kind::activate_step);
    CHECK_CONTRACT(*result.proposed_plan_operations[1].step_id == "retry");
    CHECK_CONTRACT(result.proposed_plan_operations[2].kind == common_plan_operation_kind::activate_step);
    CHECK_CONTRACT(*result.proposed_plan_operations[2].step_id == "retry-2");
    CHECK_CONTRACT(result.proposed_plan_operations[3].kind == common_plan_operation_kind::reset_step);
    CHECK_CONTRACT(*result.proposed_plan_operations[3].step_id == "fetch");
    CHECK_CONTRACT(result.proposed_plan_operations[4].kind == common_plan_operation_kind::set_next_action);
    CHECK_CONTRACT(*result.proposed_plan_operations[4].value == "re-run verification");
    CHECK_CONTRACT(result.proposed_plan_operations[5].kind == common_plan_operation_kind::add_step);
    CHECK_CONTRACT(result.proposed_plan_operations[5].step->id == "repair_1");
    CHECK_CONTRACT(result.proposed_plan_operations[5].step->tool_call && result.proposed_plan_operations[5].step->tool_call->name == "repository.search");
    CHECK_CONTRACT(result.proposed_plan_operations[6].kind == common_plan_operation_kind::add_step);
    CHECK_CONTRACT(result.proposed_plan_operations[6].step->id == "repair_2");
    CHECK_CONTRACT(result.proposed_plan_operations[6].step->depends_on == std::vector<std::string>{"repair_1"});
    CHECK_CONTRACT(result.proposed_plan_operations[7].kind == common_plan_operation_kind::replace_step);
    CHECK_CONTRACT(*result.proposed_plan_operations[7].step_id == "fetch");
    CHECK_CONTRACT(result.proposed_plan_operations[7].step->tool_call && result.proposed_plan_operations[7].step->tool_call->name == "repository.search");
    CHECK_CONTRACT(common_reflection_parse_json(R"({"decision":"revise","operations":[{"kind":"add_constraint","reason_summary":"preserve the repository boundary","constraint":{"id":"repo-boundary","description":"Do not modify files outside the repository.","hard":true}},{"kind":"add_assumption","reason_summary":"record the working assumption","assumption":{"id":"local-model","statement":"The configured model is available locally.","confidence":0.8,"evidence_ids":["model-check"]}},{"kind":"invalidate_assumption","reason_summary":"assumption no longer holds","target_id":"old-model"}]})", result, error));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 3);
    CHECK_CONTRACT(result.proposed_plan_operations[0].kind == common_plan_operation_kind::add_constraint);
    CHECK_CONTRACT(result.proposed_plan_operations[0].constraint && result.proposed_plan_operations[0].constraint->hard);
    CHECK_CONTRACT(result.proposed_plan_operations[1].kind == common_plan_operation_kind::add_assumption);
    CHECK_CONTRACT(result.proposed_plan_operations[1].assumption && result.proposed_plan_operations[1].assumption->confidence == 0.8f);
    CHECK_CONTRACT(result.proposed_plan_operations[2].kind == common_plan_operation_kind::invalidate_assumption);
    CHECK_CONTRACT(result.proposed_plan_operations[2].target_id && *result.proposed_plan_operations[2].target_id == "old-model");
    CHECK_CONTRACT(common_reflection_parse_json(R"({"decision":"revise","add_steps":[{"tool":"memory.search","args":{"tool":{"name":"memory.search","arguments":{"query":"regression procedure","limit":"2"}}}}],"replace_steps":[{"step_id":"fetch","tool":"memory.search","args":{"tool":"memory.search","arguments":{"query":"corrected lookup","limit":"3"}}}]})", result, error));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 2);
    CHECK_CONTRACT(same_json_object(result.proposed_plan_operations[0].step->tool_call->arguments_json, R"({"limit":2,"query":"regression procedure"})"));
    CHECK_CONTRACT(same_json_object(result.proposed_plan_operations[1].step->tool_call->arguments_json, R"({"limit":3,"query":"corrected lookup"})"));
    CHECK_CONTRACT(common_reflection_parse_json(R"({"decision":"revise","add_steps":[{"tool":"memory.search","args":{"tool":"memory.search","query":"regression procedure","limit":"2"}}],"replace_steps":[{"step_id":"fetch","tool":"memory.search","args":{"tool":"memory.search","query":"corrected lookup","limit":"3"}}]})", result, error));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 2);
    CHECK_CONTRACT(same_json_object(result.proposed_plan_operations[0].step->tool_call->arguments_json, R"({"limit":2,"query":"regression procedure"})"));
    CHECK_CONTRACT(same_json_object(result.proposed_plan_operations[1].step->tool_call->arguments_json, R"({"limit":3,"query":"corrected lookup"})"));
    CHECK_CONTRACT(common_reflection_parse_json(R"({"decision":"revise","add_steps":[{"tool":"memory.get","args":"memory-1"}],"replace_steps":[]})", result, error));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 1);
    CHECK_CONTRACT(same_json_object(result.proposed_plan_operations[0].step->tool_call->arguments_json, R"({"id":"memory-1"})"));
    CHECK_CONTRACT(common_reflection_parse_json(R"({"decision":"accept","ready_to_answer":true,"confidence":0.9,"revision_guidance":[],"learning_hint":{"category":"tool_precondition","statement":"Verify a repository path before reading it.","expected_reuse":0.8},"operations":[]})", result, error));
    CHECK_CONTRACT(result.learning_hint && result.learning_hint->category == "tool_precondition");
    const std::string explicit_tool_call_reflection =
        "{\"decision\":\"revise\",\"operations\":[{\"kind\":\"add_step\",\"step\":"
        "{\"id\":\"repair-fetch\",\"title\":\"Repair fetch\",\"objective\":\"Retry the fetch with normalized args\","
        "\"tool_call\":{\"name\":\"memory.search\",\"arguments_json\":\"{\\\"tool\\\":\\\"memory.search\\\",\\\"query\\\":\\\"reopened evidence\\\",\\\"limit\\\":\\\"4\\\"}\"}}}]}";
    CHECK_CONTRACT(common_reflection_parse_json(explicit_tool_call_reflection, result, error));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 1);
    CHECK_CONTRACT(result.proposed_plan_operations[0].step->tool_call);
    CHECK_CONTRACT(same_json_object(result.proposed_plan_operations[0].step->tool_call->arguments_json, R"({"limit":4,"query":"reopened evidence"})"));
    CHECK_CONTRACT(common_reflection_parse_json(R"({"decision":"revise","add_steps":[{"mode":"reasoning"},{"mode":"reasoning"}]})", result, error));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 2);
    CHECK_CONTRACT(result.proposed_plan_operations[0].step->id == "repair_1");
    CHECK_CONTRACT(result.proposed_plan_operations[1].step->id == "repair_2");
    CHECK_CONTRACT(result.proposed_plan_operations[1].step->depends_on == std::vector<std::string>{"repair_1"});
    CHECK_CONTRACT(!common_reflection_parse_json(R"({"decision":"accept","learning_hint":{"category":"x","statement":"","expected_reuse":2}})", result, error));
    CHECK_CONTRACT(!common_reflection_parse_json("not json", result, error));
    CHECK_CONTRACT(!common_reflection_parse_json(R"({"decision":"unsafe"})", result, error));
    CHECK_CONTRACT(!common_reflection_parse_json(R"({"decision":"accept","operations":[{"kind":"remove_step"}]})", result, error));
    CHECK_CONTRACT(!common_reflection_parse_json(R"({"decision":"revise","operations":[{"kind":"add_assumption","assumption":{"id":"a","statement":"x","confidence":2}}]})", result, error));
    CHECK_CONTRACT(!common_reflection_parse_json(R"({"decision":"revise","operations":[{"kind":"invalidate_assumption"}]})", result, error));
    CHECK_CONTRACT(!common_reflection_parse_json(R"({"decision":"revise","complete":"lookup"})", result, error));
    CHECK_CONTRACT(!common_reflection_parse_json(R"({"decision":"revise","replace_steps":[{"tool":"memory.search","args":{"query":"repair"}}]})", result, error));
    CHECK_CONTRACT(common_reflection_parse_json(
        R"({"decision":"revise","replace_steps":[{"tool":"memory.search","args":{"query":"repair"}}]})",
        result, error, 8, "failed-search"));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 1);
    CHECK_CONTRACT(*result.proposed_plan_operations[0].step_id == "failed-search");
    CHECK_CONTRACT(!common_reflection_parse_json(
        R"({"decision":"revise","replace_steps":[{"step_id":"","tool":"memory.search","args":{"query":"repair"}}]})",
        result, error, 8, "failed-search"));
    CHECK_CONTRACT(!common_reflection_parse_json(
        R"({"decision":"revise","replace_steps":[{"step_id":"other-step","tool":"memory.search","args":{"query":"repair"}}]})",
        result, error, 8, "failed-search"));

    CHECK_CONTRACT(common_reflection_parse_compact_dsl(
        "reflect decision=revise\nguidance \"repair the failed lookup\"\n"
        "replace | open! memory.search query=\"corrected lookup\" limit=3\n",
        result, error, "failed-search"));
    CHECK_CONTRACT(result.decision == common_reflection_decision::revise);
    CHECK_CONTRACT(result.revision_guidance == std::vector<std::string>{"repair the failed lookup"});
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 1);
    CHECK_CONTRACT(*result.proposed_plan_operations[0].step_id == "failed-search");
    CHECK_CONTRACT(result.proposed_plan_operations[0].step->tool_call->name == "memory.search");
    CHECK_CONTRACT(same_json_object(
        result.proposed_plan_operations[0].step->tool_call->arguments_json,
        R"({"limit":3,"query":"corrected lookup"})"));
    CHECK_CONTRACT(!common_reflection_parse_compact_dsl(
        "reflect decision=revise\nreplace other-step | open! memory.search query=\"wrong target\" limit=3\n",
        result, error, "failed-search"));
    CHECK_CONTRACT(common_reflection_parse_compact_dsl(
        "reflect decision=revise\ncomplete lookup\n"
        "add | open! memory.search query=\"follow-up\" limit=2\n",
        result, error));
    CHECK_CONTRACT(result.proposed_plan_operations.size() == 2);
    CHECK_CONTRACT(*result.proposed_plan_operations[0].step_id == "lookup");
    CHECK_CONTRACT(result.proposed_plan_operations[1].kind == common_plan_operation_kind::add_step);
    CHECK_CONTRACT(common_reflection_parse_compact_dsl(
        "reflect decision=revise\nassurance_action escalate_research\n"
        "learning_hint category=\"tool_precondition\" statement=\"Check the schema before querying.\" expected_reuse=0.8\n",
        result, error));
    CHECK_CONTRACT(result.next_action == common_agent_reflection_next_action::escalate_research);
    CHECK_CONTRACT(result.learning_hint && result.learning_hint->category == "tool_precondition");
    const auto require = [](bool condition, const char * message) {
        if (!condition) { std::cerr << message << '\n'; std::exit(1); }
    };
    const std::string review_dsl = "reflect revise\nassurance_action revise_plan\nguidance \"Fetch the missing evidence.\"";
    common_reflection_result dsl_review, json_review;
    require(common_reflection_parse_compact_dsl(review_dsl, dsl_review, error), "compact evidence repair review was rejected");
    require(common_reflection_parse_json(
        R"({"decision":"revise","assurance_action":"revise_plan","revision_guidance":["Fetch the missing evidence."]})",
        json_review, error), "JSON evidence repair review was rejected");
    require(dsl_review.next_action == common_agent_reflection_next_action::revise_plan &&
        dsl_review.next_action == json_review.next_action && dsl_review.revision_guidance == json_review.revision_guidance,
        "DSL and JSON changed the meaning of the same repair decision");
    require(common_reflection_parse_compact_dsl("reflect revise", result, error) &&
        result.next_action == common_agent_reflection_next_action::revise_response,
        "legacy bare revision became implicit acceptance");
    require(!common_reflection_parse_json(R"({"decision":"revise","ready_to_answer":true})", result, error),
        "negative review contradicted readiness without a controlled error");
    require(!common_reflection_parse_json(R"({"decision":"accept","assurance_action":"revise_plan"})", result, error),
        "acceptance allowed a contradictory plan-repair verdict");
    require(common_reflection_review_grammar().find("revise_plan") != std::string::npos &&
        common_reflection_review_grammar().find("guidance") != std::string::npos,
        "compact sampling grammar removed actionable evidence repair");
    return 0;
}
