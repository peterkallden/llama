#pragma once
#include "method-phase-scenario.h"
#include "agent/synthesis-state.h"
#include "plan/plan-context.h"
#include <iostream>

namespace prior_work_scenario {
inline void run(common_plan_store & store, bool trace = false) {
    using namespace method_phase_scenario;
    std::string error;
        common_plan_state method;method.id="resume-method";method.kind=common_plan_kind::blueprint;
        method.source_revision="v1";method.phase_review_required=true;method.goal="Inspect and analyse source";
        method.steps={{"inspect","Inspect","Inspect source"},{"analyse","Analyse","Analyse source"}};
        method.steps[1].depends_on={"inspect"};
        common_plan_state parent;parent.id="old-turn";parent.status=common_plan_status::active;
        parent.source_revision=method.source_revision;parent.phase_review_required=true;
        parent.derived_from_plan_id=method.id;parent.namespace_id="local";parent.session_id="session";
        parent.project_id="project";
        common_plan_step read;read.id="saved-read";read.status=common_plan_step_status::completed;
        read.tool_call=common_plan_tool_call{"repository.read","{}"};parent.steps.push_back(read);
        parent.observations.push_back(common_plan_make_blueprint_outline(method,parent.id));
        parent.observations.push_back({"tool:saved-read:repository.read","repository.read",
            R"({"ok":true,"result":{"path":"source.h","sha256":"sha256:known","lines":[{"line":1,"text":"source"}]}})",1.0f});
        parent.observations.push_back({"phase-review","method_phase_review",
            R"({"schema":"method-phase-review.v1","method_revision":"v1","phases":{"old-turn:inspect":{"status":"satisfied","reason":"Source inspected","evidence_ids":["tool:saved-read:repository.read"]}}})",0.0f});
        require(store.create(parent,error),error);parent=*store.get(parent.id,error);
        class resume_planner final:public common_planner {
        public:
            common_plan_proposal create_plan(const common_agent_request & request,std::string & error) override {
                error.clear();
                require(request.method_phase_choices.size()==1 && request.method_phase_choices.front().second=="new-turn:analyse",
                    "successor restarted inspection instead of selecting the remaining phase");
                require(request.prompt.find("source.h")!=std::string::npos,"successor planning lost prior source context");
                common_plan_proposal result;result.selected_method_phase=request.method_phase_choices.front().second;
                result.plan.id="new-turn";result.plan.status=common_plan_status::active;
                common_plan_step step{"analyse-step","Analyse","Analyse already inspected source"};
                step.mode=common_plan_step_mode::reasoning;step.status=common_plan_step_status::active;
                result.plan.steps.push_back(step);result.plan.active_step_id=step.id;return result;
            }
        } planner;
        class resume_executor final:public common_action_executor {
        public:
            std::string generate_reasoning(const common_agent_request & request,const common_plan_state & plan,
                    const common_plan_step &,std::string & error) override {
                error.clear();require(request.prior_evidence_view &&
                    !request.prior_evidence_view(plan).observations.empty(),"reasoning lost inherited evidence");
                return "Analysed the inspected source";
            }
            std::string generate_draft(const common_agent_request &,const common_plan_state &,
                    const std::vector<std::string> &,std::string & error) override {error.clear();return "Analysis completed";}
        } executor;
        class resume_reviewer final:public common_reflection_engine {
        public:
            common_reflection_result evaluate(const common_agent_request &,const common_plan_state & plan,
                    const std::string &,std::string & error) override {
                error.clear();require(std::any_of(plan.observations.begin(),plan.observations.end(),[](const auto & item){
                    return item.id=="tool:saved-read:repository.read";}),"review lost inherited evidence");
                common_reflection_result result;result.decision=common_reflection_decision::accept;
                result.task_outcome="satisfied";result.resolved_issue_ids={"prior-issue"};result.phase_reviews={{"new-turn:analyse","satisfied","Analysis grounded in retained source",{"tool:saved-read:repository.read"}}};
                return result;
            }
        } reviewer;
        phase_tools tools;common_agent_runtime runtime(store,planner,executor,reviewer,&tools);
        common_agent_request request;request.prompt="Continue source analysis";request.plan_id="new-turn";
        request.namespace_id=parent.namespace_id;request.session_id=parent.session_id;request.project_id=parent.project_id;
        request.turn_id="new";request.blueprint_definition=method;request.blueprint_plan_id=method.id;
        request.enable_memory=false;request.max_iterations=3;request.max_reflection_rounds=2;
        request.prior_evidence_is_current=[](const auto & observation){return observation.source=="repository.read";};
        common_agent_review_checklist previous_checklist;previous_checklist.round=1;
        previous_checklist.blocking.push_back({"prior-issue","Finish analysis","Analyse retained source",{}});
        request.prior_work_checkpoint=common_plan_observation{"prior","prior_work_checkpoint",
            json{{"schema","prior-work.v2"},{"source_plan_id",parent.id},{"source_plan_version",parent.version},
                {"evidence_fingerprint",common_agent_synthesis_evidence_fingerprint(parent)},
                {"review",{{"accepted",false},{"checklist",common_agent_review_checklist_json(previous_checklist)}}}}.dump(),1.0f};
        const auto result=runtime.run(request);
        if (trace) for (const auto & event : result.trace)
            std::cout << json{{"stage",common_runtime_trace_stage_name(event.stage)},
                {"detail",event.detail},{"plan_id",event.plan_id}}.dump() << '\n';
        require(result.error.empty(),"successor runtime failed: "+result.error);
        require(tools.executed.empty(),"successor replayed a completed source read");
        const auto child=store.get("new-turn",error);
        require(child && !common_agent_method_has_remaining(*child),"successor failed to retain completed method phases");
        require(store.get(parent.id,error)->version==parent.version,"successor mutated predecessor journal");

}
}
