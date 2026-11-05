#pragma once
#include "agent/agent-runtime.h"
#include "plan/plan-in-memory.h"
#include <stdexcept>

namespace synthesis_review_scenario {
inline void require(bool condition,const std::string & error) {
    if(!condition) throw std::runtime_error(error);
}
class planner final:public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &,std::string & error) override {
        error.clear(); common_plan_proposal proposal;
        proposal.plan.id="review-scenario";proposal.plan.status=common_plan_status::active;
        proposal.plan.goal="Report established evidence and its limits";proposal.plan.success_criteria="A qualified report";
        common_plan_step step{"answer","Answer","Synthesize established evidence"};
        step.mode=common_plan_step_mode::final_response;step.status=common_plan_step_status::active;
        proposal.plan.steps={step};proposal.plan.active_step_id=step.id;return proposal;
    }
};
class executor final:public common_action_executor {
public:
    size_t calls=0;
    bool fail_generation=false;
    std::string generate_draft(const common_agent_request & request,const common_plan_state &,
        const std::vector<std::string> & guidance,std::string & error) override {
        require(request.synthesis_state && !request.synthesis_state->mandatory_remaining &&
            request.synthesis_state->partial_allowed,"draft did not receive the bounded-partial contract");
        require(calls==0 || (!guidance.empty() && !request.review_checklist.blocking.empty()),
            "revision lost its prior blocker");
        ++calls;
        if(fail_generation) {error="transport unavailable";return {};}
        error.clear();return calls==1 ? "An abstract-based report." :
            "A partial abstract-based report. Full text was unavailable; no full-text analysis was performed.";
    }
};
class reviewer final:public common_reflection_engine {
public:
    size_t calls=0;
    bool reject_all=false;
    common_reflection_result evaluate(const common_agent_request & request,const common_plan_state &,
        const std::string &,std::string & error) override {
        ++calls;error.clear();common_reflection_result result;
        require(request.synthesis_state && request.synthesis_state->partial_allowed,"review lost partial-report permission");
        result.structured_review=true;
        if(calls==1 || reject_all) {
            result.decision=common_reflection_decision::revise;
            result.next_action=common_agent_reflection_next_action::revise_response;
            common_reflection_issue issue;
            issue.description="The access limitation must be explicit";
            issue.correction="State that full text was unavailable and was not analyzed";
            if(calls>1) {
                require(request.review_checklist.blocking.size()==1,"next review did not retain the checklist");
                issue.previous_issue_id=request.review_checklist.blocking.front().id;
            }
            result.issues={issue};
        } else {
            require(request.review_checklist.blocking.size()==1,"next review did not receive its previous issue");
            result.decision=common_reflection_decision::accept;result.ready_to_answer=true;
            result.resolved_issue_ids={request.review_checklist.blocking.front().id};
            result.advisory_notes={"A shorter heading would also be acceptable"};
        }
        return result;
    }
};
inline common_agent_result run(bool reject_all,bool fail_generation=false,bool resume=false) {
    common_plan_in_memory_store store;std::string error;require(store.open("",error),error);
    planner p;executor e;reviewer r;r.reject_all=reject_all;e.fail_generation=fail_generation;
    common_agent_runtime runtime(store,p,e,r);
    common_agent_request request;request.prompt="Write a partial report from the available abstract";
    request.max_iterations=6;request.max_reflection_rounds=2;request.enable_memory=false;
    common_agent_synthesis_state state;
    state.evidence=common_agent_synthesis_evidence::partial;state.partial_allowed=true;
    state.stop_reason="source_access_blocked";request.synthesis_state=state;
    if(resume) request.max_reflection_rounds=1;
    auto result=runtime.run(request);
    if(resume) {
        require(result.unverified_draft && result.response.empty(),"first execution did not retain its unfinished review");
        common_agent_runtime restarted(store,p,e,r);
        request.plan_id="review-scenario";request.max_reflection_rounds=2;
        result=restarted.run(request);
    }
    if(fail_generation) {
        require(!result.error.empty() && !result.unverified_draft && r.calls==0,
            "generation transport failure was relabelled as a complete review draft");
        return result;
    }
    require(r.calls==2 && e.calls==2,"review did not perform exactly one bounded revision");
    if(reject_all) {
        require(!result.error.empty() && result.response.empty() && result.unverified_draft &&
            result.unverified_draft->issues.size()==1,"rejected response was not retained as a separate draft");
        const auto saved=store.get("review-scenario",error);
        require(saved && saved->status!=common_plan_status::completed,"rejected draft completed the plan");
    } else require(result.error.empty() && !result.response.empty() && !result.unverified_draft,
        "corrected partial report was not accepted");
    return result;
}
}
