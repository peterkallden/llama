#include "agent/synthesis-state.h"
#include "agent/review-checklist.h"
#include "agent/protocol/agent-jsonl.h"
#include "pocs/agent/smoke/support/synthesis-review-scenario.h"

using json=nlohmann::ordered_json;
int main() {
    using namespace synthesis_review_scenario;
    common_plan_state plan;plan.id="state";
    common_agent_synthesis_state research;
    research.evidence=common_agent_synthesis_evidence::partial;research.partial_allowed=true;
    auto state=common_agent_resolve_synthesis_state(plan,research);
    require(!state.mandatory_remaining && state.partial_allowed && state.work==common_agent_synthesis_work::bounded,
        "partial evidence was confused with unfinished work");
    common_plan_step pending{"pending","Acquire","Read required evidence"};
    pending.tool_call=common_plan_tool_call{"catalog.get","{}"};plan.steps={pending};
    state=common_agent_resolve_synthesis_state(plan,research);
    require(state.mandatory_remaining && !state.partial_allowed,"mandatory work allowed premature partial synthesis");
    plan.steps.clear();plan.workflow_definition=common_plan_workflow_definition{};
    auto & workflow=*plan.workflow_definition;
    workflow.workflow_ref="workflow://test/recovery";workflow.workflow_revision="v1";
    workflow.start_state="done";workflow.terminal_states={"done"};
    common_plan_workflow_transition unused;unused.id="unused";unused.from_state="other";unused.to_state="done";
    workflow.transitions={unused};workflow.continuation_links.resize(1);
    state=common_agent_resolve_synthesis_state(plan,research);
    require(state.mandatory_remaining,"unknown continuation resolution permitted synthesis");
    const json resolution{{"workflow",workflow.workflow_ref},{"revision","v1"},{"segment",0},
        {"terminal_state","done"},{"outcome","not_applicable"},{"reason","use_limit_reached"},
        {"evidence_fingerprint",common_agent_workflow_evidence_fingerprint(plan)}};
    plan.observations.push_back({"resolution","workflow_continuation_resolution",resolution.dump(),1.0f});
    require(common_agent_resolve_synthesis_state(plan,research).partial_allowed,"bounded recovery was not retained");
    plan.observations.push_back({"tool:new:catalog.get","catalog.get","{}",1.0f});
    require(common_agent_resolve_synthesis_state(plan,research).mandatory_remaining,"stale closure survived new source evidence");
    common_agent_review_checklist checklist;checklist.round=1;
    checklist.blocking.push_back({"review-1-1","Missing limitation","Disclose unavailable text",{"tool:source"}});
    common_agent_review_checklist restored;
    require(common_agent_review_checklist_from_json(common_agent_review_checklist_json(checklist),restored) &&
        restored.blocking.front().id=="review-1-1","review checklist did not round trip");
    common_reflection_result review;review.structured_review=true;review.decision=common_reflection_decision::accept;
    common_agent_review_checklist next;
    std::string error;
    require(!common_agent_advance_review_checklist(restored,review,plan,"draft",next,error),
        "accept silently discarded an unaccounted blocker");
    review.resolved_issue_ids={"invented"};
    require(!common_agent_advance_review_checklist(restored,review,plan,"draft",next,error),"invented issue identity was accepted");
    review.resolved_issue_ids={"review-1-1"};review.advisory_notes={"Change the heading"};
    require(common_agent_advance_review_checklist(restored,review,plan,"draft",next,error) && next.blocking.empty(),
        "advisory style preference blocked an otherwise accepted report");
    review.decision=common_reflection_decision::revise;review.next_action=common_agent_reflection_next_action::revise_response;
    require(!common_agent_advance_review_checklist(restored,review,plan,"draft",next,error),
        "advisory-only review demanded response revision");
    common_agent_review_checklist empty_checklist;
    common_reflection_result unfinished;
    unfinished.decision=common_reflection_decision::accept;
    unfinished.task_outcome="remaining";
    require(!common_agent_advance_review_checklist(empty_checklist,unfinished,plan,"draft",next,error),
        "completed plan slice concealed remaining user work");
    unfinished.decision=common_reflection_decision::revise;
    unfinished.next_action=common_agent_reflection_next_action::revise_plan;
    require(common_agent_advance_review_checklist(empty_checklist,unfinished,plan,"draft",next,error),
        "remaining work could not request replanning");
    unfinished.decision=common_reflection_decision::accept;
    unfinished.next_action=common_agent_reflection_next_action::accept;
    unfinished.task_outcome="blocked";
    require(common_agent_advance_review_checklist(empty_checklist,unfinished,plan,"draft",next,error),
        "explicit blocked outcome could not preserve an honest bounded result");
    common_plan_state failed_plan;
    common_plan_step failed_step{"lookup","Search","Find code"};
    failed_step.tool_call=common_plan_tool_call{"repository.search",R"({"query":"needle"})"};
    failed_step.status=common_plan_step_status::failed;
    failed_plan.steps.push_back(failed_step);
    failed_plan.observations.push_back({"tool:lookup:repository.search","repository.search",
        R"({"failure":{"retryable":false}})",1.0f});
    common_reflection_result retry;
    retry.decision=common_reflection_decision::revise;
    retry.next_action=common_agent_reflection_next_action::revise_plan;
    common_plan_operation reset;
    reset.kind=common_plan_operation_kind::reset_step;reset.step_id="lookup";
    retry.proposed_plan_operations.push_back(reset);
    require(!common_agent_review_step_allows_retry(failed_plan,"lookup") &&
        !common_agent_advance_review_checklist(empty_checklist,retry,failed_plan,"draft",next,error),
        "non-retryable timeout was blindly reset");
    failed_plan.observations.push_back({"tool:lookup:repository.search:attempt:2","repository.search",
        R"({"failure":{"retryable":true}})",1.0f});
    require(common_agent_review_step_allows_retry(failed_plan,"lookup") &&
        common_agent_advance_review_checklist(empty_checklist,retry,failed_plan,"draft",next,error),
        "latest retryable host failure did not supersede previous failure");
    run(false);
    const auto rejected=run(true);
    common_agent_jsonl_turn_result wire;
    wire.error=rejected.error;wire.unverified_draft=rejected.unverified_draft;
    const auto encoded=common_agent_jsonl_make_turn_result(wire);
    require(!encoded["ok"].get<bool>() && encoded["response"]=="" &&
        encoded["unverified_draft"]["review_status"]=="unverified", "wire result promoted a rejected draft to a successful response");
    wire.unverified_draft.reset();
    require(!common_agent_jsonl_make_turn_result(wire).contains("unverified_draft"),"legacy wire result acquired a fabricated draft");
    run(false,true);run(false,false,true);
}
