#include "agent/agent-runtime.h"
#include "agent/method-phases.h"
#include "agent/review-checklist.h"
#include "agent/thinking/reflection-references.h"
#include "plan/plan-context.h"
#include "plan/plan-in-memory.h"
#include "pocs/agent/smoke/support/method-phase-scenario.h"
#include <stdexcept>
#include <algorithm>

using json=nlohmann::ordered_json;
int main() {
    using namespace method_phase_scenario;
    std::string error;
    const auto saved=std::optional<common_plan_state>(run());
    run(true); // Rejected continuation is corrected before any side effect.
    run(false,true);
    run(false,false,true);
    // Known phase identities, evidence references and dependencies remain host-owned.
    auto pending=*saved;
    pending.observations.erase(std::remove_if(pending.observations.begin(),pending.observations.end(),
        [](const auto & o){return o.source=="method_phase_review";}),pending.observations.end());
    json progress;
    require(!common_agent_method_review(pending,{{pending.id+":knowledge","satisfied","Premature",{}}},progress,error),
        "phase completed ahead of its prerequisite");
    require(common_agent_method_review(pending,{{pending.id+":orient","remaining","Further inspection needed",{}},
        {pending.id+":knowledge","not_needed","No external lookup needed yet",{}}},progress,error) &&
        !progress.contains(pending.id+":knowledge"),
        "provisional downstream review prevented continuation or retained a premature decision");
    require(common_agent_method_review(pending,{{pending.id+":knowledge","satisfied","Documented",{}},
        {pending.id+":orient","satisfied","Source inspected",{}}},progress,error) &&
        progress.contains(pending.id+":knowledge"),"valid verdicts depended on array order");
    require(!common_agent_method_review(pending,{{pending.id+":orient","satisfied","Unknown source",{"invented"}}},progress,error),
        "phase accepted invented evidence");
    require(!common_agent_method_review(pending,{{"invented-phase","blocked","Not available",{}}},progress,error),
        "phase identity was model-created");
    const auto ready=common_agent_method_ready_phases(pending);
    require(ready.size()==1 && ready[0]["phase_id"]==pending.id+":orient",
        "ready frontier exposed a phase before its prerequisite");
    common_agent_request selection;selection.method_phase_choices={{"p1",pending.id+":orient"}};
    std::string selected;
    auto selection_json=std::string(R"({"method_phase":"p1","goal":"Inspect source","steps":[{"mode":"reasoning"}]})");
    require(common_agent_method_parse_selection(selection,false,selection_json,selected,error) &&
        selected==pending.id+":orient" && !json::parse(selection_json).contains("method_phase"),error);
    auto selection_dsl=std::string("phase p1\nplan goal=\"Inspect source\"\nstep reasoning");
    require(common_agent_method_parse_selection(selection,true,selection_dsl,selected,error) &&
        selection_dsl.rfind("plan ",0)==0,error);
    auto invalid_choice=std::string(R"({"method_phase":"p2"})");
    const auto unmodified=invalid_choice;
    require(!common_agent_method_parse_selection(selection,false,invalid_choice,selected,error) &&
        invalid_choice==unmodified && error.find("p1")!=std::string::npos,
        "invalid phase choice changed input or lost actionable feedback");
    common_reflection_result premature;premature.decision=common_reflection_decision::accept;
    premature.task_outcome="satisfied";common_agent_review_checklist checklist,next;
    require(!common_agent_advance_review_checklist(checklist,premature,pending,"draft",next,error),
        "finished inspection slice bypassed unresolved method phases");

    // No-tool and blocked phases are valid decisions, not fabricated execution.
    require(common_agent_method_review(pending,{{pending.id+":orient","not_needed","The user supplied sufficient evidence",{}},
        {pending.id+":knowledge","blocked","External lookup unavailable",{}}},progress,error),error);
    pending.observations.push_back({"bounded-review","method_phase_review",
        json{{"schema","method-phase-review.v1"},{"phases",progress}}.dump(),0.0f});
    premature.task_outcome="blocked";
    require(common_agent_advance_review_checklist(checklist,premature,pending,"bounded report",next,error),error);
    premature.task_outcome="satisfied";
    require(!common_agent_advance_review_checklist(checklist,premature,pending,"false success",next,error),
        "blocked phase became a satisfied user outcome");
    common_plan_state repair=*saved;
    common_plan_step failed{"build","Build","Verify changed source"};
    failed.status=common_plan_step_status::failed;
    failed.tool_call=common_plan_tool_call{"development.build",R"({"source_snapshot_ref":"before"})"};
    failed.tool_call->result_requirement_json=R"({"kind":"verification"})";
    failed.depends_on={saved->steps.front().id};failed.required_evidence={"verified-source"};
    repair.steps.push_back(failed);
    auto other_failed=failed;other_failed.id="other-build";
    other_failed.tool_call->arguments_json=R"({"source_snapshot_ref":"other-before"})";
    repair.steps.push_back(other_failed);
    common_plan_state slice;auto corrected=failed;corrected.id="retry-build";
    corrected.optional=true;corrected.required_evidence.clear();corrected.depends_on.clear();
    corrected.tool_call->arguments_json=R"({"source_snapshot_ref":"after"})";
    slice.steps={corrected};std::vector<common_plan_operation> operations;
    require(common_agent_method_append_slice(repair,slice,operations,error,{{"retry-build","build"}}) && operations.size()==1 &&
        operations.front().kind==common_plan_operation_kind::replace_step && operations.front().step->id=="build" &&
        !operations.front().step->optional && operations.front().step->required_evidence==failed.required_evidence &&
        operations.front().step->depends_on==failed.depends_on &&
        operations.front().step->tool_call->result_requirement_json==failed.tool_call->result_requirement_json,
        "corrected verification did not preserve its failed node and history");
    // Compare the canonical call after resolving slice-local references.
    common_plan_state binding_plan=repair;
    common_plan_step old_source{"old-source","Source","Existing producer"};
    binding_plan.steps.push_back(old_source);
    binding_plan.steps[binding_plan.steps.size()-3].tool_call->arguments_json=
        R"({"source":{"$from_step":"old-source","$json_pointer":"/uri"}})";
    common_plan_state binding_slice;
    common_plan_step replay{"retry","Retry","Retry failed operation"};
    replay.tool_call=common_plan_tool_call{"development.build",R"({"source":{"$from_step":"local-source","$json_pointer":"/uri"}})"};
    common_plan_step local_source{"local-source","Source","Repair producer"};
    local_source.tool_call=common_plan_tool_call{"repository.read","{}"};
    old_source.status=common_plan_step_status::failed;
    old_source.tool_call=common_plan_tool_call{"repository.read",R"({"path":"old"})"};
    binding_plan.steps.back()=old_source;
    local_source.tool_call->arguments_json=R"({"path":"new"})";
    binding_slice.steps={local_source,replay};
    require(!common_agent_method_append_slice(binding_plan,binding_slice,operations,error,
        {{"local-source","old-source"},{"retry","build"}}) && error.find("blindly replay")!=std::string::npos,
        "slice-local binding bypassed the failed-call replay check");
    slice.steps.front().tool_call->arguments_json=R"({ "source_snapshot_ref" : "before" })";
    require(!common_agent_method_append_slice(repair,slice,operations,error,{{"retry-build","build"}}),
        "whitespace-only argument change bypassed blind retry rejection");
    require(!common_agent_method_append_slice(repair,slice,operations,error),
        "omitting the repair reference bypassed blind retry rejection");
    slice.steps[0].tool_call->arguments_json=R"({"source_snapshot_ref":"after"})";
    // Reusing a tool does not imply replacing any previous failed operation.
    auto fresh=slice.steps.front();fresh.id="fresh";
    fresh.depends_on={slice.steps.front().id};slice.steps.push_back(fresh);
    require(common_agent_method_append_slice(repair,slice,operations,error) && operations.size()==2 &&
        operations[0].kind==common_plan_operation_kind::add_step &&
        operations[1].kind==common_plan_operation_kind::add_step &&
        operations[0].step->id!=operations[1].step->id &&
        operations[1].step->depends_on==std::vector<std::string>{operations[0].step->id},
        "same-tool new work collapsed into the failed execution identity");
    slice.steps[0].tool_call->arguments_json=R"({"source_snapshot_ref":"after"})";
    slice.steps[1].tool_call->arguments_json=R"({"source_snapshot_ref":"later"})";
    require(!common_agent_method_append_slice(repair,slice,operations,error,
        {{"retry-build","build"},{"fresh","build"}}),"two repairs acquired the same execution identity");
    require(!common_agent_method_append_slice(repair,slice,operations,error,{{"retry-build","missing"}}),
        "repair accepted an unknown target");
    auto cycle=slice;cycle.steps.resize(1);cycle.steps[0].depends_on={cycle.steps[0].id};
    require(!common_agent_method_append_slice(repair,cycle,operations,error,{{"retry-build","build"}}) &&
        error.find("dependency cycle")!=std::string::npos,"explicit repair accepted a real dependency cycle");
    common_plan_state bound_slice;
    common_plan_step producer{"source","Source","Resolve fresh source"};
    producer.tool_call=common_plan_tool_call{"repository.snapshot","{}"};
    common_plan_step consumer{"use","Use","Read bound result"};consumer.depends_on={"source"};
    consumer.tool_call=common_plan_tool_call{"resource.read",R"({"uri":{"$from_step":"source","$json_pointer":"/uri"}})"};
    bound_slice.steps={consumer,producer};
    require(common_agent_method_append_slice(*saved,bound_slice,operations,error) && operations.size()==2 &&
        operations.front().step->tool_call->name=="repository.snapshot" &&
        operations.back().step->depends_on==std::vector<std::string>{operations.front().step->id} &&
        operations.back().step->tool_call->arguments_json.find(operations.front().step->id)!=std::string::npos,
        "phase slicing lost deferred argument identity or dependency order");
    common_plan_state reopened=*saved;
    require(common_agent_method_review(reopened,{{reopened.id+":orient","remaining","Source changed; inspect again",{}}},progress,error) &&
        !progress.contains(reopened.id+":knowledge"),"reopened prerequisite retained stale descendant completion");
    auto reversed=*saved;
    for (auto & observation:reversed.observations) {
        if (observation.source!="blueprint_outline") continue;
        auto phases=json::parse(observation.summary);
        phases.push_back({{"id",reversed.id+":later"},{"depends_on",json::array({reversed.id+":knowledge"})}});
        std::reverse(phases.begin(),phases.end());observation.summary=phases.dump();
    }
    auto earlier=json::object();
    earlier[reversed.id+":orient"]={{"status","satisfied"}};
    earlier[reversed.id+":knowledge"]={{"status","satisfied"}};
    earlier[reversed.id+":later"]={{"status","satisfied"}};
    reversed.observations.push_back({"reverse-review","method_phase_review",
        json{{"schema","method-phase-review.v1"},{"method_revision",reversed.source_revision},{"phases",earlier}}.dump(),0.0f});
    require(common_agent_method_review(reversed,{{reversed.id+":orient","remaining","Reinspect source",{}}},progress,error) &&
        !progress.contains(reversed.id+":knowledge") && !progress.contains(reversed.id+":later"),
        "reopening failed to invalidate transitive decisions in reverse outline order");
    const auto before=common_agent_synthesis_evidence_fingerprint(*saved);
    reopened.observations.push_back({"extra-review","method_phase_review",
        json{{"schema","method-phase-review.v1"},{"phases",progress}}.dump(),0.0f});
    require(common_agent_synthesis_evidence_fingerprint(reopened)==before,
        "model phase decisions were promoted to new source evidence");
    // Completed tool evidence lives in observations, not step.result_summary.
    auto repeated=pending;repeated.steps.clear();repeated.observations.clear();
    common_plan_step prior{"one","Inspect","Read evidence"};
    prior.status=common_plan_step_status::completed;
    prior.method_phase_id=pending.id+":orient";prior.method_turn_id="turn-1";
    prior.semantic_alias="model-alias";
    prior.tool_call=common_plan_tool_call{"repository.search",R"({"query":"symbol"})"};
    repeated.steps={prior,prior};repeated.steps.back().id="two";
    repeated.observations.push_back({"tool:one:repository.search","repository.search",R"({"matches":["same"]})",1.0f});
    repeated.observations.push_back({"tool:two:repository.search","repository.search",R"({"matches":["same"]})",1.0f});
    common_plan_operation repeat;repeat.step=prior;
    const auto read_only=[](const std::string &) {return true;};
    require(!common_agent_method_validate_progress(repeated,{repeat},pending.id+":orient","turn-1",read_only,error),
        "new observation IDs bypassed unchanged-call detection");
    repeated.observations.back().summary=R"({"matches":["changed"]})";
    require(common_agent_method_validate_progress(repeated,{repeat},pending.id+":orient","turn-1",read_only,error),
        "changed source result was incorrectly rejected");
    repeated.observations.back().summary=R"({"matches":["same"],"result_truncated":true})";
    require(common_agent_method_validate_progress(repeated,{repeat},pending.id+":orient","turn-1",read_only,error),
        "truncated output was treated as proof of identical results");
    repeated.observations.back().summary=R"({"matches":["same"]})";
    require(common_agent_method_validate_progress(repeated,{repeat},pending.id+":orient","turn-2",read_only,error),
        "a new user turn could not explicitly repeat an operation");
    require(common_agent_method_validate_progress(repeated,{repeat},pending.id+":orient","turn-1",
        [](const std::string &) {return false;},error),"progress guard incorrectly classified a write as a read");
    require(common_agent_method_validate_progress(repeated,{repeat},pending.id+":orient","",read_only,error),
        "missing turn identity incorrectly blocked an explicit refresh");
    auto reasoning=repeated;
    for (auto & step:reasoning.steps) {
        step.tool_call.reset();step.mode=common_plan_step_mode::reasoning;
    }
    reasoning.observations={{"reasoning:one","reasoning",R"({"analysis":"unchanged"})",1.0f},
        {"reasoning:two","reasoning",R"({"analysis":"unchanged"})",1.0f}};
    auto reason_op=repeat;reason_op.step=reasoning.steps.front();
    require(!common_agent_method_validate_progress(reasoning,{reason_op},pending.id+":orient","turn-1",read_only,error),
        "identical reasoning loop was not recognized");
    reasoning.observations.back().summary=R"({"analysis":"new conclusion"})";
    require(common_agent_method_validate_progress(reasoning,{reason_op},pending.id+":orient","turn-1",read_only,error),
        "new reasoning outcome was incorrectly treated as a duplicate");
    repeat.step->tool_call->arguments_json=R"({"query":"symbol","offset":10})";
    require(common_agent_method_validate_progress(repeated,{repeat},pending.id+":orient","turn-1",read_only,error),
        "pagination was treated as a repeated call");
    common_plan_state legacy;legacy.id="legacy";
    require(!common_agent_method_has_remaining(legacy) && common_agent_render_method_progress(legacy).empty(),
        "legacy plans acquired new method requirements");

    common_plan_state evidence;
    evidence.observations={{"tool:long:composite:step:repository.read","repository.read","source",0.0f},
        {"e1","memory","collision",0.0f}};
    common_reflection_evidence_references refs(evidence);
    require(refs.project("\"tool:long:composite:step:repository.read\" e1")=="\"e1\" e2",
        "reference projection cascaded replacements or lost stable order");
    require(refs.project("tool:long:composite:step:repository.read:attempt:2")==
        "tool:long:composite:step:repository.read:attempt:2","projection matched only a prefix of an identity");
    const std::string host_view="  evidence: \"tool:long:composite:step:repository.read\"\n"
        "  content: \"e1 tool:long:composite:step:repository.read\"\n"
        "{\"evidence_ids\":[\"tool:long:composite:step:repository.read\"],\"reason\":\"e1\"}\n";
    const auto projected=refs.project_context(host_view);
    require(projected.find("  evidence: \"e1\"")!=std::string::npos &&
        projected.find("  content: \"e1 tool:long:composite:step:repository.read\"")!=std::string::npos &&
        projected.find("\"evidence_ids\":[\"e1\"]")!=std::string::npos &&
        projected.find("\"reason\":\"e1\"")!=std::string::npos,
        "reference projection changed source content or missed host-owned fields");
    common_reflection_result review;
    review.phase_reviews={{"phase","remaining","Need more evidence",{"e1"}}};
    review.issues.push_back({});review.issues.back().evidence_ids={"e2"};
    require(refs.restore(review,error) && review.phase_reviews[0].evidence_ids[0]==evidence.observations[0].id &&
        review.issues[0].evidence_ids[0]=="e1" && evidence.observations[0].id=="tool:long:composite:step:repository.read",
        "review references changed durable identities or provenance");
    review.phase_reviews[0].evidence_ids={"e1"};review.issues[0].evidence_ids={"e999"};
    require(!refs.restore(review,error) && review.phase_reviews[0].evidence_ids==std::vector<std::string>{"e1"},
        "unknown reference was guessed or a rejected restore partially mutated the review");
}
