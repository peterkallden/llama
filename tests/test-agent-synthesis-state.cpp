#include "agent/synthesis-state.h"
#include "agent/review-checklist.h"
#include "agent/protocol/agent-jsonl.h"
#include "agent/thinking/reflection-references.h"
#include "agent/thinking/reflection-json.h"
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
    research.stop_reason="execution_budget_exhausted";
    state=common_agent_resolve_synthesis_state(plan,research);
    require(state.mandatory_remaining && state.partial_allowed && state.work==common_agent_synthesis_work::pending,
        "budget finalization hid unfinished work or prohibited a qualified partial answer");
    require(common_agent_render_synthesis_state(state,"final_review").find("task_outcome blocked")!=std::string::npos,
        "budget finalization omitted the separate task-outcome contract");
    research.stop_reason.clear();
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
    review.decision=common_reflection_decision::revise;
    review.next_action=common_agent_reflection_next_action::revise_plan;
    require(common_agent_advance_review_checklist(restored,review,plan,"draft",next,error) &&
        next.blocking.size()==1 && next.blocking.front().id=="review-1-1",
        "plan repair lost an omitted prior blocker");
    review.structured_review=false;
    require(common_agent_advance_review_checklist(restored,review,plan,"draft",next,error) &&
        next.blocking.size()==1,"legacy plan repair discarded its prior blocker");
    review.structured_review=true;review.decision=common_reflection_decision::accept;
    review.next_action=common_agent_reflection_next_action::accept;
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
    // Partial review is bound to exact inspected text, never to an entire
    // report merely because it was saved or because some excerpts passed.
    common_plan_state report_plan;report_plan.id="report-review";
    common_plan_step inspected;inspected.id="read-report";inspected.status=common_plan_step_status::completed;
    inspected.tool_call=common_plan_tool_call{"resource.read",R"({"uri":"resource://report-r3"})"};
    report_plan.steps.push_back(inspected);
    report_plan.observations.push_back({"tool:read-report:resource.read","resource.read",
        R"({"ok":true,"result":{"resource":{"uri":"resource://report-r3"},"content":"Timeline: authentication observed. Coverage remains limited."}})",1.0f});
    common_reflection_result partial;partial.structured_review=true;partial.task_outcome="satisfied";
    partial.parts.push_back({"tool:read-report:resource.read","Timeline: authentication observed.","reviewed",
        "Matches the saved observation",{"tool:read-report:resource.read"},"resource://invented","invented"});
    partial.parts.push_back({"tool:read-report:resource.read","Coverage remains limited.","requires_work",
        "Additional collection is optional for this report-update request",{}});
    common_agent_review_checklist progress;
    require(common_agent_advance_review_checklist({},partial,report_plan,"saved report",progress,error) &&
        progress.parts.size()==2 && progress.parts.front().resource_uri=="resource://report-r3" &&
        progress.parts.front().content_fingerprint!="invented" && progress.blocking.empty(),
        "qualified coverage blocked a saved update or model-supplied identity was trusted: "+error);
    common_agent_review_checklist loaded_progress;
    require(common_agent_review_checklist_from_json(common_agent_review_checklist_json(progress),loaded_progress) &&
        loaded_progress.parts.front().content_fingerprint==progress.parts.front().content_fingerprint,
        "partial review lost its content binding after round trip");
    auto fabricated=partial;fabricated.parts.front().excerpt="No such claim was read";
    require(!common_agent_advance_review_checklist({},fabricated,report_plan,"saved report",next,error),
        "unread text received a reviewed verdict");
    fabricated=partial;fabricated.parts.front().evidence_ids={"invented"};
    require(!common_agent_advance_review_checklist({},fabricated,report_plan,"saved report",next,error),
        "invented evidence approved a report excerpt");
    fabricated=partial;fabricated.parts.front().evidence_ids.clear();
    require(!common_agent_advance_review_checklist({},fabricated,report_plan,"saved report",next,error),
        "reviewed excerpt lacked supporting evidence");
    auto stale=report_plan;stale.observations.front().summary=R"({"failure":{"code":"read_failed"}})";
    require(!common_agent_advance_review_checklist({},partial,stale,"saved report",next,error),
        "failed read received a reviewed verdict");
    common_reflection_result new_review;new_review.structured_review=true;
    require(common_agent_advance_review_checklist(progress,new_review,report_plan,"changed draft",next,error) &&
        next.parts.size()==2,"unchanged report excerpts were invalidated by a chat wording change");
    auto changed_content=report_plan;
    changed_content.observations.front().summary=R"({"ok":true,"result":{"resource":{"uri":"resource://report-r3"},"content":"Different content"}})";
    require(common_agent_advance_review_checklist(progress,new_review,changed_content,"changed draft",next,error) &&
        next.parts.empty(),"changed report content retained stale excerpt assessments");
    auto with_blocker=progress;with_blocker.blocking.push_back({"unsupported","Unsupported conclusion","Remove it",{}});
    require(!common_agent_advance_review_checklist(with_blocker,new_review,report_plan,"saved report",next,error),
        "partially reviewed text laundered an unresolved blocker");
    common_reflection_evidence_references references(report_plan);
    auto compact=partial;compact.parts.front().observation_id="e1";compact.parts.front().evidence_ids={"e1"};
    compact.parts.back().observation_id="e1";
    require(references.restore(compact,error) && compact.parts.front().observation_id==report_plan.observations.front().id,
        "excerpt handle did not restore its canonical identity");
    require(references.restore(compact,error),"canonical issued evidence identity was rejected");
    compact.parts.front().observation_id="unknown";
    require(!references.restore(compact,error),"unknown excerpt identity bypassed restoration");
    common_agent_review_progress paused{report_plan.id,progress.draft_fingerprint,progress.evidence_fingerprint,
        false,true,progress.parts,with_blocker.blocking};
    common_agent_jsonl_turn_result paused_wire;paused_wire.review_progress=paused;
    const auto paused_json=common_agent_jsonl_make_turn_result(paused_wire);
    require(!paused_json["ok"].get<bool>() && paused_json["review_progress"]["paused"]==true &&
        paused_json["review_progress"]["parts"].size()==2,"paused partial review became a successful answer");
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
