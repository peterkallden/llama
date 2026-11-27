#pragma once
#include "agent/agent-runtime.h"
#include "agent/method-phases.h"
#include "plan/plan-in-memory.h"
#include <algorithm>
#include <stdexcept>

namespace method_phase_scenario {
using json=nlohmann::ordered_json;
static void require(bool ok,const std::string & message) { if(!ok) throw std::runtime_error(message); }

class phase_planner final:public common_planner {
public:
    size_t calls=0;
    bool reject_first_continuation=false;
    bool reject_all_continuations=false;
    bool fail_continuation_transport=false;
    common_plan_proposal create_plan(const common_agent_request & request,std::string & error) override {
        error.clear();++calls;
        if(calls>1) require(request.prompt.find("method-phase-context.v1")!=std::string::npos,
            "continuation planner lost retained method progress");
        if ((reject_first_continuation || reject_all_continuations) && calls==3) require(request.prompt.find("Host rejected the previous continuation")!=std::string::npos &&
            request.prompt.find("unknown argument binding")!=std::string::npos,
            "continuation retry lost the specific host validation error");
        common_plan_proposal proposal;
        if (fail_continuation_transport && calls>1) {
            error="synthetic planner transport failure";
            proposal.generation=common_agent_generated_text_result{};return proposal;
        }
        if(calls>1) {
            require(request.method_phase_choices.size()==1,"planner did not receive the dependency-ready phase frontier");
            proposal.selected_method_phase=request.method_phase_choices.front().second;
        }
        proposal.plan.id="method-test";proposal.plan.status=common_plan_status::active;
        proposal.plan.goal="Inspect local code and applicable primary documentation";
        common_plan_step step{"lookup","Inspect evidence","Read the phase's authorized source"};
        step.status=common_plan_step_status::active;
        step.tool_call=common_plan_tool_call{calls==1?"repository.read":"web.fetch","{}"};
        if ((reject_first_continuation && calls==2) || (reject_all_continuations && calls>1))
            step.tool_call->arguments_json=R"({"url":{"$from_step":"invented","$json_pointer":"/url"}})";
        step.selected_tool=step.tool_call->name;
        if(calls>1) step.semantic_alias="source-choice";
        proposal.plan.steps={step};proposal.plan.active_step_id=step.id;
        return proposal;
    }
};
class phase_executor final:public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &,const common_plan_state &,
        const std::vector<std::string> &,std::string & error) override {error.clear();return "Inspected evidence and stated limits.";}
};
class phase_tools final:public common_agent_tool_runtime {
public:
    mutable std::vector<std::string> executed;
    bool is_read_only(const std::string &) const override {return true;}
    bool is_policy_gated(const std::string &) const override {return false;}
    bool validate(const common_agent_tool_call & call,std::string & error) const override {
        error.clear();return call.name=="repository.read" || call.name=="web.fetch";
    }
    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        executed.push_back(call.name);return common_tool_execution_result::success(R"({"content":"verified host source"})");
    }
};
class phase_reviewer final:public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &,const common_plan_state & plan,
            const std::string &,std::string & error) override {
        error.clear();common_reflection_result result;
        std::string local,external;
        for(const auto & observation:plan.observations) {
            if(observation.source=="repository.read") local=observation.id;
            if(observation.source=="web.fetch") external=observation.id;
        }
        require(!local.empty(),"method review preceded local source execution");
        if(external.empty()) {
            result.decision=common_reflection_decision::revise;
            result.next_action=common_agent_reflection_next_action::revise_plan;
            result.task_outcome="remaining";
            common_plan_operation guidance;guidance.kind=common_plan_operation_kind::set_next_action;
            guidance.value="Inspect the applicable primary source";result.proposed_plan_operations.push_back(guidance);
            result.phase_reviews={{plan.id+":orient","satisfied","Local source inspected",{local}},
                {plan.id+":knowledge","remaining","Need primary documentation",{local}}};
        } else {
            result.decision=common_reflection_decision::accept;result.task_outcome="satisfied";
            result.phase_reviews={{plan.id+":knowledge","satisfied","Applicable primary source inspected",{external}}};
        }
        return result;
    }
};
inline common_plan_state run(bool reject_first_continuation=false,bool reject_all_continuations=false,bool fail_continuation_transport=false,
        size_t execution_limit=12, size_t tool_limit=2, size_t storage_limit=16) {
    std::string error;
    common_plan_policy_config storage; storage.max_steps=storage_limit;
    common_plan_in_memory_store store{common_plan_policy(storage)};require(store.open("",error),error);
    phase_planner planner;phase_executor executor;phase_reviewer reviewer;phase_tools tools;
    planner.reject_first_continuation=reject_first_continuation;
    planner.reject_all_continuations=reject_all_continuations;
    planner.fail_continuation_transport=fail_continuation_transport;
    common_agent_runtime runtime(store,planner,executor,reviewer,&tools);
    common_plan_state method;method.kind=common_plan_kind::blueprint;method.id="development-method";
    method.goal="Inspect local code and documentation";method.success_criteria="Both sources inspected";
    method.source_revision="test@v1";method.phase_review_required=true;
    method.steps={{"orient","Orient","Inspect local source"},{"knowledge","Knowledge","Inspect applicable documentation"}};
    method.steps[1].depends_on={"orient"};
    common_agent_request request;request.prompt=method.goal;request.blueprint_definition=method;
    request.enable_memory=false;request.max_iterations=execution_limit;request.max_tool_batches=tool_limit;
    request.max_reflection_rounds=1; // Phase continuation must not consume the final verdict budget.
    auto result=runtime.run(request);
    if (storage_limit==2) {
        require(result.plan_storage_exhausted && result.remaining_continuation_budget.has_value(),
            "capacity stop did not retain its remaining task budget");
        require(result.remaining_continuation_budget->executions_remaining<execution_limit + method.steps.size()*2 &&
            result.remaining_continuation_budget->tool_calls_remaining<tool_limit,
            "capacity handoff replenished consumed task budget");
        return *store.get("method-test",error);
    }
    if (execution_limit == 1 || tool_limit == 1) {
        require(result.limit_reached && !result.error.empty() && result.response.empty(),
            "exhausted continuation did not report a resumable limit");
        require(tools.executed == std::vector<std::string>{"repository.read"} &&
            planner.calls == (execution_limit == 1 ? 1u : 2u),
            "exhausted budget executed work or planned a slice with no execution round");
        auto stopped=store.get("method-test",error);
        require(stopped && common_agent_method_has_remaining(*stopped) &&
            std::none_of(stopped->steps.begin(),stopped->steps.end(),[](const auto & step) {
                return step.tool_call && step.tool_call->name=="web.fetch";
            }), "unrunnable continuation mutated the saved plan");
        return *stopped;
    }
    if (reject_all_continuations || fail_continuation_transport) {
        require(!result.error.empty() && planner.calls==(fail_continuation_transport?2u:3u) &&
            tools.executed==std::vector<std::string>{"repository.read"},
            "invalid continuation was executed, transport retried or validation repair was unbounded");
        auto stopped=store.get("method-test",error);
        require(stopped && common_agent_method_has_remaining(*stopped),
            "rejected continuation closed unresolved method work");
        return *stopped;
    }
    require(result.error.empty() && !result.response.empty(),"real runtime method continuation failed: "+result.error);
    require(planner.calls==(reject_first_continuation?3u:2u) && tools.executed==std::vector<std::string>{"repository.read","web.fetch"},
        "method did not execute exactly one local slice followed by one external slice");
    auto saved=store.get("method-test",error);require(saved && saved->phase_review_required &&
        !common_agent_method_has_remaining(*saved),"method progress did not persist in the ordinary plan");
    const auto continuation=std::find_if(saved->steps.begin(),saved->steps.end(),[](const auto & step) {
        return step.tool_call && step.tool_call->name=="web.fetch";
    });
    require(continuation!=saved->steps.end() && continuation->semantic_alias=="source-choice" &&
        continuation->method_phase_id==saved->id+":knowledge" && continuation->method_turn_id==request.turn_id,
        "method provenance replaced the model alias or was not retained in the plan");
    require(common_agent_render_method_progress(*saved).find("Applicable primary source")!=std::string::npos,
        "retained context lost phase provenance");

    return *saved;
}
} // namespace method_phase_scenario
