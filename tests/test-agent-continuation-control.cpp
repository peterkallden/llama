#include "agent/continuation-budget.h"
#include "agent/continuation-control.h"
#include "agent/method-phases.h"
#include "pocs/agent/smoke/support/method-phase-scenario.h"

using namespace method_phase_scenario;

namespace {
class failing_tools final : public common_agent_tool_runtime {
public:
    mutable size_t calls = 0;
    bool retryable = true;
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }
    bool validate(const common_agent_tool_call &, std::string & error) const override { error.clear(); return true; }
    common_tool_execution_result execute(const common_agent_tool_call &) const override {
        ++calls;
        return common_tool_execution_result::failure("test.network",common_tool_failure_class::network,retryable,"Source unavailable");
    }
};
class blocked_reviewer final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &,
            const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result review;
        review.decision=common_reflection_decision::revise;
        review.next_action=common_agent_reflection_next_action::fail_bounded;
        review.task_outcome="blocked";
        review.revision_guidance={"Source unavailable; retain unfinished inspection."};
        return review;
    }
};
class retry_reviewer final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request & request, const common_plan_state & plan,
            const std::string &, std::string & error) override {
        error.clear();
        require(request.continuation_budget && request.continuation_budget->tool_calls_remaining == 0,
            "failed call did not spend the host tool budget");
        common_reflection_result review;
        review.decision=common_reflection_decision::revise;
        review.next_action=common_agent_reflection_next_action::revise_plan;
        common_plan_operation retry;
        retry.kind=common_plan_operation_kind::reset_step;retry.step_id=plan.steps.front().id;
        review.proposed_plan_operations.push_back(retry);
        return review;
    }
};
class two_tool_planner final : public common_planner {
public:
    bool workflow = false;
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "two-tool-budget"; proposal.plan.goal = "Inspect both sources";
        proposal.plan.status = common_plan_status::active;
        common_plan_step first{"first", "First", "Read first source"};
        first.status = common_plan_step_status::active;
        first.tool_call = common_plan_tool_call{"repository.read", "{}"};
        first.selected_tool = first.tool_call->name;
        common_plan_step second{"second", "Second", "Read second source"};
        second.depends_on = {first.id};
        second.tool_call = common_plan_tool_call{"web.fetch", "{}"};
        second.selected_tool = second.tool_call->name;
        proposal.plan.steps = {first, second}; proposal.plan.active_step_id = first.id;
        if (workflow) {
            common_plan_workflow_definition definition;
            definition.workflow_ref="workflow://budget";definition.workflow_revision="v1";
            definition.family="test";definition.graph_revision="v1";
            definition.start_state="start";definition.terminal_states={"done"};
            common_plan_workflow_transition transition;transition.id="first";transition.from_state="start";
            transition.to_state="done";transition.tool_name="repository.read";
            definition.transitions.push_back(transition);proposal.plan.workflow_definition=definition;
        }
        return proposal;
    }
};

class review_workflow_planner final : public common_planner {
public:
    bool allow_review=true;
    bool allow_mutation=false;
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear(); common_plan_proposal proposal;
        auto & plan=proposal.plan; plan.id="review-workflow"; plan.status=common_plan_status::active;
        plan.goal="Inspect source and resolve evidence review";
        common_plan_step first{"initial","Read","Inspect initial source"};
        first.tool_call=common_plan_tool_call{"repository.read","{}"};first.selected_tool="repository.read";
        first.status=common_plan_step_status::active;first.semantic_alias="workflow-transition:initial";
        plan.steps={first};plan.active_step_id=first.id;
        plan.route_binding=common_plan_route_binding{};plan.route_binding->route_id="review-route";
        common_plan_workflow_definition graph;
        graph.workflow_ref="workflow://test/review";graph.workflow_revision="v1";graph.family="test";
        graph.graph_revision="v1";graph.start_state="start";graph.terminal_states={"done"};
        common_plan_workflow_transition initial;initial.id="initial";initial.from_state="start";
        initial.to_state="done";initial.tool_name="repository.read";
        common_plan_workflow_transition entry;entry.id="review-entry";entry.from_state="done";entry.to_state="extra";
        entry.kind=common_plan_workflow_transition_kind::model_choice;entry.review_only=true;
        entry.choices.push_back({"inspect","Inspect missing evidence","extra",{}});
        auto extra=initial;extra.id="extra";extra.from_state="extra";extra.tool_name="web.fetch";
        extra.review_mutation_allowed=allow_mutation;
        graph.transitions={initial,entry,extra};
        if (!allow_review) graph.transitions={initial};
        plan.workflow_definition=graph;
        return proposal;
    }
};
// Exercise write classification without weakening the ordinary fixture.
class mutation_tools final : public common_agent_tool_runtime {
public:
    phase_tools delegate;
    bool is_read_only(const std::string & name) const override { return name != "web.fetch"; }
    bool is_policy_gated(const std::string & name) const override { return name == "web.fetch"; }
    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        return delegate.validate(call,error);
    }
    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        return delegate.execute(call);
    }
};
class evidence_reviewer final : public common_reflection_engine {
public:
    size_t calls=0;
    common_reflection_result evaluate(const common_agent_request & request,const common_plan_state & plan,
            const std::string &,std::string & error) override {
        ++calls;error.clear();common_reflection_result review;
        const bool inspected=std::any_of(plan.observations.begin(),plan.observations.end(),
            [](const auto & o){return o.source=="web.fetch";});
        if (inspected) {
            review.decision=common_reflection_decision::accept;
            for (const auto & issue : request.review_checklist.blocking) review.resolved_issue_ids.push_back(issue.id);
            return review;
        }
        review.decision=common_reflection_decision::revise;
        review.next_action=common_agent_reflection_next_action::revise_plan;
        review.task_outcome="remaining";review.revision_guidance={"Read the remaining evidence"};
        common_plan_operation description;description.kind=common_plan_operation_kind::set_next_action;
        description.value="Inspect missing evidence before answering";
        review.proposed_plan_operations={description};return review;
    }
};
}

int main() {
    if (common_agent_continuation_execution_limit(4,3)!=4 ||
            common_agent_continuation_execution_limit(4,12)!=13 ||
            common_agent_continuation_execution_limit(9,3)!=9 ||
            common_agent_continuation_execution_limit(0,0)!=2) return 1;

    std::string error;
    common_agent_continuation_budget budget{1, 1, 1, 1};
    common_plan_operation op;
    op.kind = common_plan_operation_kind::add_step;
    op.step = common_plan_step{"inspect", "Inspect", "Inspect source"};
    op.step->tool_call = common_plan_tool_call{"repository.read", R"({"path":"a.cpp"})"};
    require(common_agent_admit_continuation(budget, {op}, error), error);
    auto exhausted = budget;
    exhausted.executions_remaining = 0;
    require(!common_agent_admit_continuation(exhausted, {op}, error), "last-round work was admitted");
    exhausted = budget; exhausted.tool_calls_remaining = 0;
    require(!common_agent_admit_continuation(exhausted, {op}, error), "tool exhaustion was ignored");
    exhausted = budget; exhausted.revisions_remaining = 0;
    require(!common_agent_admit_continuation(exhausted, {op}, error), "revision exhaustion was ignored");
    exhausted = budget; exhausted.new_steps_remaining = 0;
    require(!common_agent_admit_continuation(exhausted, {op}, error), "step exhaustion was ignored");
    op.kind = common_plan_operation_kind::replace_step;
    op.step_id = "inspect";
    require(common_agent_admit_continuation(exhausted, {op}, error), "replacement consumed a new step slot");
    auto binding = op; binding.kind = common_plan_operation_kind::revise_step;
    auto no_execution = budget; no_execution.executions_remaining = 0;
    require(!common_agent_admit_continuation(no_execution, {binding}, error), "tool binding bypassed execution admission");
    auto activate = op; activate.kind = common_plan_operation_kind::activate_step;
    require(common_agent_admit_continuation(budget, {op, activate}, error), "same work was counted twice");
    auto reasoning = op; reasoning.step->tool_call.reset(); reasoning.step->mode = common_plan_step_mode::reasoning;
    exhausted = budget; exhausted.tool_calls_remaining = 0;
    require(!common_agent_admit_continuation(exhausted, {reasoning, op}, error),
        "earlier non-tool operation hid a tool call on the same step");
    require(common_agent_admit_continuation({}, {}, error), "a verdict without new work required execution budget");

    common_plan_policy_config limits;
    limits.max_observations=2; limits.max_workflow_control_observations=8;
    common_plan_state capacity_plan;capacity_plan.id="capacity";
    capacity_plan.status=common_plan_status::active;
    common_plan_workflow_definition workflow;
    workflow.start_state="start";workflow.terminal_states={"done"};
    common_plan_workflow_transition terminal;terminal.id="read";terminal.from_state="start";
    terminal.to_state="done";terminal.tool_name="repository.read";
    workflow.transitions.push_back(terminal);capacity_plan.workflow_definition=workflow;
    for(size_t i=0;i<5;++i) capacity_plan.observations.push_back({"control-"+std::to_string(i),"workflow_arguments","{}",1.0f});
    const auto capacity=common_plan_remaining_capacity(capacity_plan,limits);
    require(capacity.observations==2 && capacity.workflow_control==3,"host control records consumed the evidence budget");
    require(!common_agent_workflow_storage_available(capacity_plan,limits,error),"control budget was not reserved before dispatch");
    common_plan_policy policy(limits);
    common_plan_operation fake;fake.kind=common_plan_operation_kind::record_observation;
    fake.plan_id=capacity_plan.id;fake.expected_version=capacity_plan.version;
    fake.observation=common_plan_observation{"invented","workflow_choice","{}",1.0f};
    require(!policy.validate(capacity_plan,fake).allowed,"model-authored control record bypassed host authority");

    common_plan_state plan;
    plan.id = "recovery"; plan.phase_review_required = true;
    auto failed = *op.step; failed.status = common_plan_step_status::failed;
    plan.steps.push_back(failed);
    const auto observe = [&](bool retryable, const std::string & kind) {
        plan.observations.push_back({"tool:inspect:repository.read:attempt:" + std::to_string(plan.observations.size()+1),
            "repository.read", json{{"failure", {{"class",kind},{"retryable",retryable},{"code","test.failure"}}}}.dump(), 0.0f});
    };
    require(!common_agent_continuation_allows_same_call(plan,"inspect"), "missing declaration granted a retry");
    observe(false, "limit");
    require(!common_agent_continuation_allows_same_call(plan,"inspect"), "argument repair allowed blind retry");
    observe(true, "network");
    require(common_agent_continuation_allows_same_call(plan,"inspect"), "host-approved retry was denied");
    auto identical = op;
    require(common_agent_validate_continuation_retries(plan, {identical}, error),
        "eligible same-step retry was rejected");
    identical.kind = common_plan_operation_kind::add_step;
    identical.step_id.reset(); identical.step->id = "new-identity";
    require(!common_agent_validate_continuation_retries(plan, {identical}, error),
        "unchanged failed call escaped retry policy through a new step");
    identical.step->tool_call->arguments_json = R"({"path":"different.cpp"})";
    require(common_agent_validate_continuation_retries(plan, {identical}, error),
        "corrected arguments were rejected as an unchanged retry");
    common_plan_state slice; auto retry=failed; retry.id="retry"; slice.steps={retry};
    std::vector<common_plan_operation> operations;
    require(common_agent_method_append_slice(plan,slice,operations,error,{{"retry","inspect"}}), error);
    require(!common_agent_method_append_slice(plan,slice,operations,error), "retry acquired a fresh identity to evade its target");
    observe(true, "network");
    require(!common_agent_continuation_allows_same_call(plan,"inspect"), "unchanged retries were unbounded");
    identical = op;
    require(!common_agent_validate_continuation_retries(plan, {identical}, error),
        "exhausted retry bypassed ordinary plan admission");
    require(!common_agent_method_append_slice(plan,slice,operations,error,{{"retry","inspect"}}),
        "exhausted retry passed phase admission");
    plan.observations.back().summary = R"({"failure":{"class":"network","retryable":true,"call_fingerprint":"new"}})";
    plan.observations[plan.observations.size()-2].summary = R"({"failure":{"class":"network","retryable":true,"call_fingerprint":"old"}})";
    require(common_agent_continuation_allows_same_call(plan,"inspect"),
        "a changed dispatched call inherited the previous call's retry count");
    require(common_agent_recovery_for_failure(common_agent_failure_class::policy,true)==common_agent_recovery_action::blocked,
        "policy failure became retryable");
    require(common_agent_recovery_for_failure(common_agent_failure_class::not_found,false)==common_agent_recovery_action::inspect,
        "missing source lost investigation semantics");
    require(common_agent_render_continuation_context(plan,budget).find("same_call_allowed")!=std::string::npos,
        "model context omitted retry constraints");
    plan.steps.front().tool_call->arguments_json = "{";
    require(common_agent_render_continuation_context(plan,budget).find("arguments_omitted")!=std::string::npos,
        "invalid saved arguments broke continuation context rendering");
    // Real runtime path: the final execution round must not invoke a planner,
    // and an oversized tool slice must not mutate the retained plan.
    run(false,false,false,1,2);
    run(false,false,false,12,1);
    run();

    // An ordinary plan shares the same guard; a failed execution is charged
    // and its reset is rejected before changing the failed state.
    common_plan_in_memory_store store;
    require(store.open("",error),error);
    phase_planner planner; phase_executor executor; retry_reviewer reviewer; failing_tools tools;
    common_agent_runtime runtime(store,planner,executor,reviewer,&tools);
    common_agent_request request;
    request.prompt="Inspect source";request.enable_memory=false;
    request.max_iterations=3;request.max_tool_batches=1;request.max_reflection_rounds=2;
    const auto stopped=runtime.run(request);
    require(stopped.limit_reached && tools.calls==1 && stopped.response.empty(),
        "failed tool was retried beyond its execution budget");
    const auto retained=store.get("method-test",error);
    require(retained && retained->steps.front().status==common_plan_step_status::failed,
        "budget denial reset mandatory failed work");

    common_plan_in_memory_store blocked_store;require(blocked_store.open("",error),error);
    blocked_reviewer bounded_stop;failing_tools blocked_tools;blocked_tools.retryable=false;
    phase_planner blocked_planner; phase_executor blocked_executor;
    common_agent_runtime blocked_runtime(blocked_store,blocked_planner,blocked_executor,bounded_stop,&blocked_tools);
    auto blocked_request=request;blocked_request.max_tool_batches=16;
    const auto blocked=blocked_runtime.run(blocked_request);
    require(!blocked.limit_reached && blocked.response.empty() && blocked_tools.calls==1 &&
        blocked.error.find("Source unavailable; retain unfinished inspection.")!=std::string::npos &&
        blocked.review_progress && !blocked.review_progress->paused && !blocked.review_progress->accepted &&
        blocked.review_progress->blocked_reason == blocked.error &&
        common_agent_review_progress_json(*blocked.review_progress)["blocked_reason"] == blocked.error,
        "model-declared blockage lost its cause or became a budget pause: "+blocked.error);
    const auto blocked_plan=blocked_store.get("method-test",error);
    require(blocked_plan && blocked_plan->steps.front().status==common_plan_step_status::failed &&
        std::any_of(blocked_plan->observations.begin(),blocked_plan->observations.end(),[](const auto & observation) {
            if(observation.source!="response_review") return false;
            const auto saved=json::parse(observation.summary);
            return saved.value("task_outcome",std::string{})=="blocked" && !saved.value("accepted",true) &&
                saved["revision_guidance"][0]=="Source unavailable; retain unfinished inspection.";
        }),"bounded stop failed to persist its reason and mandatory failed work");

    common_plan_in_memory_store second_store;
    require(second_store.open("",error),error);
    two_tool_planner two_planner; phase_executor second_executor; phase_reviewer unused_reviewer; phase_tools second_tools;
    common_agent_runtime second_runtime(second_store,two_planner,second_executor,unused_reviewer,&second_tools);
    common_agent_request second_request;
    second_request.prompt="Inspect both sources"; second_request.enable_memory=false;
    second_request.enable_reflection=false; second_request.max_tool_batches=1;
    const auto incomplete=second_runtime.run(second_request);
    require(incomplete.limit_reached && incomplete.response.empty() && second_tools.executed.size()==1,
        "tool budget exhaustion accepted a response with mandatory work remaining");
    const auto pending=second_store.get("two-tool-budget",error);
    require(pending && pending->steps[1].status!=common_plan_step_status::completed,
        "tool budget exhaustion lost the pending step");
    common_plan_policy_config tiny;tiny.max_observations=1;
    common_plan_in_memory_store full_store{common_plan_policy(tiny)};
    require(full_store.open("",error),error);
    two_tool_planner full_planner;full_planner.workflow=true;
    phase_tools never_dispatched;
    common_agent_runtime full_runtime(full_store,full_planner,second_executor,unused_reviewer,&never_dispatched);
    const auto full=full_runtime.run(second_request);
    require(full.plan_storage_exhausted && full.limit_reached && full.response.empty() && never_dispatched.executed.empty(),
        "workflow executed before reserving space for its durable result");

    // Actual runtime: descriptive next_action must delegate to the declared
    // workflow repair, and one review round must still allow a fresh verdict.
    common_plan_in_memory_store review_store; require(review_store.open("",error),error);
    review_workflow_planner review_planner; evidence_reviewer evidence_review;
    phase_tools evidence_tools;phase_executor evidence_executor;
    common_agent_runtime evidence_runtime(review_store,review_planner,evidence_executor,evidence_review,&evidence_tools);
    common_agent_request review_request;review_request.prompt="Inspect both pieces of evidence";
    review_request.enable_memory=false;review_request.max_iterations=6;review_request.max_tool_batches=4;
    review_request.allow_policy_gated_tool_proposals=true;
    review_request.max_reflection_rounds=1;
    review_request.deliberation_policy.max_plan_revisions=2;
    size_t review_entries=0;
    review_request.workflow_continuation=[&](const auto &,const std::string & completed,
            const common_plan_observation & observation,common_agent_workflow_continuation & steps,std::string & failure) {
        failure.clear();steps.clear();
        if(observation.source!=common_agent_workflow_review_source) return true;
        ++review_entries;
        require(json::parse(observation.summary).contains("review_request"),"review request lost its guidance");
        common_plan_step read{"extra","Inspect","Read missing evidence"};
        read.tool_call=common_plan_tool_call{"web.fetch","{}"};read.selected_tool="web.fetch";
        read.depends_on={completed};read.semantic_alias="workflow-transition:review-entry=inspect,extra";
        steps.push_back(read);return true;
    };
    const auto reviewed=evidence_runtime.run(review_request);
    require(reviewed.error.empty() && !reviewed.response.empty() && evidence_review.calls==2 &&
        review_entries==1 && evidence_tools.executed==std::vector<std::string>{"repository.read","web.fetch"},
        "workflow review repeated synthesis instead of executing additional evidence: "+reviewed.error);

    // A full durable plan is a bounded pause, not an execution error that
    // discards the saved draft or silently accepts the report.
    // One source step plus the required host synthesis step fit; the
    // additional review evidence step must exceed durable capacity.
    common_plan_policy_config review_limits;review_limits.max_steps=2;
    common_plan_in_memory_store bounded_review_store{common_plan_policy(review_limits)};
    require(bounded_review_store.open("",error),error);
    review_workflow_planner bounded_review_planner;evidence_reviewer bounded_reviewer;phase_tools bounded_tools;
    common_agent_runtime bounded_runtime(bounded_review_store,bounded_review_planner,evidence_executor,bounded_reviewer,&bounded_tools);
    const auto paused_review=bounded_runtime.run(review_request);
    require(paused_review.plan_storage_exhausted && paused_review.limit_reached && paused_review.response.empty() &&
        paused_review.unverified_draft && paused_review.review_progress && paused_review.review_progress->paused &&
        !paused_review.review_progress->accepted && bounded_tools.executed==std::vector<std::string>{"repository.read"},
        "review step-capacity exhaustion lost the bounded pause: "+paused_review.error);

    // Both admission gates must preserve the explicit transition opt-in.
    for (bool authorized : {false,true}) {
        common_plan_in_memory_store mutation_store; require(mutation_store.open("",error),error);
        review_workflow_planner mutation_planner; mutation_planner.allow_mutation=authorized;
        evidence_reviewer mutation_reviewer; mutation_tools tools;
        common_agent_runtime mutation_runtime(mutation_store,mutation_planner,evidence_executor,mutation_reviewer,&tools);
        const auto outcome=mutation_runtime.run(review_request);
        if (authorized) {
            require(outcome.error.empty() && !outcome.response.empty() && mutation_reviewer.calls==2 &&
                tools.delegate.executed==std::vector<std::string>{"repository.read","web.fetch"},
                "declared review mutation did not execute and receive a fresh review: "+outcome.error);
        } else {
            require(outcome.error.find("explicit transition authorization")!=std::string::npos &&
                tools.delegate.executed==std::vector<std::string>{"repository.read"},
                "review executed a mutation without transition authorization");
        }
    }

    const auto entries_before_denial=review_entries;
    common_plan_in_memory_store no_review_store;require(no_review_store.open("",error),error);
    review_workflow_planner no_review_planner;no_review_planner.allow_review=false;
    evidence_reviewer denied_reviewer;phase_tools denied_tools;
    common_agent_runtime no_review_runtime(no_review_store,no_review_planner,evidence_executor,denied_reviewer,&denied_tools);
    const auto denied=no_review_runtime.run(review_request);
    require(denied.error.find("no declared evidence-review continuation")!=std::string::npos &&
        denied_reviewer.calls==1 && denied_tools.executed.size()==1 && review_entries==entries_before_denial,
        "missing review authority silently repeated synthesis or executed extra work");

}
