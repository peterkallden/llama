#include "agent/research/research-runner.h"
#include "agent/research/research-workspace.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "runtime/runtime-operation.h"

#include <cstdio>

class deterministic_research_tools final : public common_agent_tool_runtime {
public:
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }

    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        if (call.name != "repository.search" && call.name != "memory_get") {
            error = "only repository.search and memory_get are exposed in this smoke";
            return false;
        }
        error.clear();
        return true;
    }

    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        ++calls;
        if (call.name == "memory_get") {
            return common_tool_execution_result::success(
                "{\"memory\":{\"id\":\"memory-smoke-1\",\"summary\":\"Use the existing runtime contract.\"}}",
                "memory_get returned one deterministic memory record");
        }
        return common_tool_execution_result::success(
            "{\"matches\":[{\"path\":\"common/agent\",\"line\":1}]}",
            "repository.search returned one deterministic repository match");
    }

    mutable int calls = 0;
};

class acquisition_guard_tools final : public common_agent_tool_runtime {
public:
    bool is_read_only(const std::string &) const override { return true; }
    bool is_policy_gated(const std::string &) const override { return false; }

    bool validate(const common_agent_tool_call & call, std::string & error) const override {
        if (call.name != "resource_read") {
            error = "research guard smoke exposes only resource_read";
            return false;
        }
        error.clear();
        return true;
    }

    common_tool_execution_result execute(const common_agent_tool_call & call) const override {
        ++calls;
        common_runtime_resource_ref resource;
        resource.uri = "agent-resource://guarded";
        resource.name = "guarded.txt";
        resource.mime_type = "text/plain";
        resource.size_bytes = 16;
        resource.scope = common_runtime_resource_scope::turn;
        return common_tool_execution_result::success(
            "controller-owned resource evidence",
            "resource_read completed",
            {resource});
    }

    mutable int calls = 0;
};

class acquisition_plan_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "research-acquisition-guard-plan";
        proposal.plan.session_id = request.session_id;
        proposal.plan.goal = request.prompt;
        proposal.plan.success_criteria = "answer";
        proposal.plan.status = common_plan_status::active;
        common_plan_step acquisition{
            "model-acquisition", "Model acquisition", "Read the supplied resource."};
        acquisition.status = common_plan_step_status::active;
        acquisition.mode = common_plan_step_mode::tool;
        acquisition.selected_tool = "resource_read";
        acquisition.tool_call = common_plan_tool_call{
            "resource_read", R"({"uri":"agent-resource://guarded","max_bytes":32768})"};
        common_plan_step answer{"answer", "Answer", "Return the researched answer"};
        answer.mode = common_plan_step_mode::final_response;
        answer.depends_on = {"model-acquisition"};
        proposal.plan.steps = {acquisition, answer};
        proposal.plan.active_step_id = acquisition.id;
        return proposal;
    }
};

class acquisition_plan_executor final : public common_action_executor {
public:
    std::string generate_reasoning(const common_agent_request &, const common_plan_state &,
            const common_plan_step &, std::string & error) override {
        error.clear();
        return "controller-owned acquisition is complete";
    }

    std::string generate_draft(const common_agent_request &, const common_plan_state &,
            const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "guarded-answer";
    }
};

class smoke_planner final : public common_planner {
public:
    std::string received_prompt;

    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        received_prompt = request.prompt;
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "research-runtime-plan";
        proposal.plan.session_id = request.session_id;
        proposal.plan.goal = request.prompt;
        proposal.plan.success_criteria = "answer";
        proposal.plan.status = common_plan_status::active;
        common_plan_step answer{"answer", "Answer", "Return the researched answer"};
        answer.status = common_plan_step_status::active;
        answer.mode = common_plan_step_mode::final_response;
        proposal.plan.steps = {answer};
        proposal.plan.active_step_id = "answer";
        return proposal;
    }
};

class smoke_executor final : public common_action_executor {
public:
    std::string received_prompt;
    int draft_calls = 0;

    std::string generate_draft(const common_agent_request & request, const common_plan_state &, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        ++draft_calls;
        received_prompt = request.prompt;
        return "researched-answer";
    }
};

class reopen_verifier final : public common_agent_research_answer_verifier {
public:
    mutable int calls = 0;

    common_agent_research_verification verify(
            const common_agent_research_result & research,
            const std::string & draft,
            std::string & error) const override {
        common_agent_research_verification_context context;
        return verify(research, draft, context, error);
    }

    common_agent_research_verification verify(
            const common_agent_research_result &,
            const std::string &,
            const common_agent_research_verification_context &,
            std::string & error) const override {
        error.clear();
        ++calls;
        if (calls == 1) {
            common_agent_research_verification result;
            result.decision = common_agent_research_verification_decision::gather_evidence;
            result.more_research_required = true;
            result.summary = "smoke verifier requested one additional evidence pass";
            return result;
        }
        common_agent_research_verification result;
        result.decision = common_agent_research_verification_decision::accept;
        result.answer_sufficient = true;
        result.confidence = 0.9;
        result.summary = "smoke verifier accepted reopened research";
        return result;
    }
};

class smoke_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        return result;
    }
};

class late_research_reflector final : public common_reflection_engine {
public:
    int calls = 0;

    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &,
            const std::string &, std::string & error) override {
        error.clear();
        ++calls;
        common_reflection_result result;
        if (calls == 1) {
            result.decision = common_reflection_decision::revise;
            result.next_action = common_agent_reflection_next_action::escalate_research;
            result.issues.push_back({"missing_evidence", "The draft requires verified evidence.", {}, 0.9f});
            return result;
        }
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        return result;
    }
};

int main() {
    common_agent_research_workspace workspace;
    workspace.workspace_id = "research-smoke";
    workspace.request_id = "request-1";
    workspace.turn_id = "turn-1";
    workspace.session_id = "session-1";
    workspace.scope.namespace_id = "research-smoke";
    workspace.scope.session_id = "session-1";
    workspace.objective.objective_id = "objective-1";
    workspace.objective.question = "Which bounded research gaps remain?";
    workspace.budget.max_iterations = 2;

    std::string error;
    if (!common_agent_research_add_gap(workspace, {"gap-1", "Find the first gap", "smoke", "evidence addresses the first gap", 2}, error) ||
            !common_agent_research_add_gap(workspace, {"gap-2", "Find the second gap", "smoke", "evidence addresses the second gap", 1}, error)) {
        std::fprintf(stderr, "research smoke setup failed: %s\n", error.c_str());
        return 1;
    }

    const auto workspace_descriptor = describe_common_agent_research_workspace(workspace);
    if (workspace_descriptor.state_class != common_agent_state_class::turn_workspace ||
            workspace_descriptor.lifetime != common_agent_state_lifetime::turn ||
            workspace_descriptor.persistence != common_agent_state_persistence::none ||
            workspace_descriptor.source_of_truth != "research workspace") {
        std::fprintf(stderr, "research workspace descriptor is invalid\n");
        return 1;
    }
    common_plan_state plan;
    plan.id = "state-smoke-plan";
    plan.namespace_id = workspace.scope.namespace_id;
    plan.session_id = workspace.scope.session_id;
    plan.scope = common_plan_scope::session;
    const auto plan_descriptor = describe_common_plan(plan);
    agent_resource_descriptor resource;
    resource.resource_id = "state-smoke-resource";
    resource.uri = "agent-resource://state-smoke-resource";
    resource.namespace_id = workspace.scope.namespace_id;
    resource.session_id = workspace.scope.session_id;
    resource.scope = common_runtime_resource_scope::session;
    const auto resource_descriptor = describe_agent_resource(resource);
    common_runtime_operation_status operation;
    operation.operation.operation_id = "state-smoke-operation";
    const auto operation_descriptor = describe_common_runtime_operation(operation);
    if (plan_descriptor.state_class != common_agent_state_class::durable_domain ||
            plan_descriptor.lifetime != common_agent_state_lifetime::session ||
            resource_descriptor.state_class != common_agent_state_class::durable_domain ||
            resource_descriptor.lifetime != common_agent_state_lifetime::session ||
            operation_descriptor.state_class != common_agent_state_class::resident_runtime ||
            operation_descriptor.lifetime != common_agent_state_lifetime::operation) {
        std::fprintf(stderr, "state descriptors are invalid\n");
        return 1;
    }
    if (!common_agent_research_transition_gap(
            workspace, "gap-1", common_agent_research_gap_status::investigating, error) ||
            !common_agent_research_transition_gap(
                workspace, "gap-1", common_agent_research_gap_status::open, error)) {
        std::fprintf(stderr, "research gap transition setup failed: %s\n", error.c_str());
        return 1;
    }
    if (common_agent_research_transition_gap(
            workspace, "gap-1", common_agent_research_gap_status::blocked, error)) {
        std::fprintf(stderr, "research gap transition accepted an invalid direct transition\n");
        return 1;
    }
    if (!common_agent_research_transition_gap(
            workspace, "gap-1", common_agent_research_gap_status::open, error)) {
        std::fprintf(stderr, "research gap reopen transition failed: %s\n", error.c_str());
        return 1;
    }

    common_agent_research_task transition_task;
    transition_task.task_id = "transition-task";
    transition_task.gap_id = "gap-1";
    transition_task.instruction = "validate transitions";
    if (!common_agent_research_add_task(workspace, transition_task, error) ||
            !common_agent_research_transition_task(
                workspace, transition_task.task_id,
                common_agent_research_task_status::active, error) ||
            !common_agent_research_transition_task(
                workspace, transition_task.task_id,
                common_agent_research_task_status::failed, error) ||
            !common_agent_research_transition_task(
                workspace, transition_task.task_id,
                common_agent_research_task_status::pending, error)) {
        std::fprintf(stderr, "research task transition setup failed: %s\n", error.c_str());
        return 1;
    }
    if (common_agent_research_transition_task(
            workspace, transition_task.task_id,
            common_agent_research_task_status::completed, error)) {
        std::fprintf(stderr, "research task transition accepted failed-to-completed\n");
        return 1;
    }

    deterministic_research_tools tools;
    common_agent_research_runtime_adapter adapter(tools);
    common_agent_research_runner runner;
    const auto result = runner.run(workspace, adapter, error);
    if (!error.empty() || !result.complete || result.stop_reason != common_agent_research_stop_reason::sufficient_coverage ||
            result.coverage.answered_gaps != 2 || workspace.sources.size() != 2 || workspace.evidence.size() != 2 ||
            workspace.iterations_completed != 2 || workspace.tool_calls != 2 || tools.calls != 2 ||
            result.coverage.evidence_quality <= 0.0 || result.coverage.source_diversity <= 0.0) {
        std::fprintf(stderr, "research runtime smoke failed: %s\n", error.c_str());
        return 1;
    }
    for (const auto & evidence : workspace.evidence) {
        if (evidence.claim_id.empty() ||
                evidence.origin != common_agent_research_evidence_origin::normalized_tool_result ||
                evidence.directly_observed || evidence.source_location.empty()) {
            std::fprintf(stderr, "research evidence provenance was not normalized correctly\n");
            return 1;
        }
    }
    std::printf("research_runtime=ok gaps=%d sources=%zu evidence=%zu\n",
        result.coverage.answered_gaps, workspace.sources.size(), workspace.evidence.size());

    const auto research_checkpoint = make_common_agent_research_workspace_checkpoint(
        workspace, 3, error);
    if (!error.empty() ||
            !common_agent_research_workspace_checkpoint_valid(research_checkpoint, error) ||
            research_checkpoint.workspace.gaps.size() != workspace.gaps.size() ||
            research_checkpoint.workspace.evidence.size() != workspace.evidence.size()) {
        std::fprintf(stderr, "research workspace checkpoint failed: %s\n", error.c_str());
        return 1;
    }
    auto resumed_workspace = research_checkpoint.workspace;
    resumed_workspace.no_progress_iterations = 0;
    if (resumed_workspace.workspace_id != workspace.workspace_id ||
            resumed_workspace.sources.size() != workspace.sources.size() ||
            !common_agent_research_workspace_validate(resumed_workspace, error)) {
        std::fprintf(stderr, "research workspace checkpoint reload failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("research_workspace_checkpoint=ok sequence=%zu\n", research_checkpoint.sequence);

    common_agent_research_bounded_verifier verifier;
    common_agent_research_verification_context verification_context;
    common_memory_hit verifier_memory;
    verifier_memory.memory.id = "memory-smoke-1";
    std::vector<common_memory_hit> verifier_memories{verifier_memory};
    verification_context.memories = &verifier_memories;
    common_runtime_resource_ref user_resource;
    user_resource.uri = "agent-resource://user-supplied/article";
    user_resource.metadata.purpose = "user supplied research reference";
    verification_context.input_resources.push_back(user_resource);
    verification_context.claims.push_back({
        "claim-supported-by-memory", "The runtime contract is reusable.", {}, {}, {"memory-smoke-1"}, 0.9});
    verification_context.claims.push_back({
        "claim-supported-by-resource", "The supplied article is relevant.", {}, {user_resource.uri}, {}, 0.8});
    common_agent_research_result incomplete_result = result;
    incomplete_result.complete = false;
    incomplete_result.unresolved_claim_ids = {"gap-unresolved"};
    auto verification = verifier.verify(result, "researched-answer", verification_context, error);
    if (!error.empty() || verification.decision != common_agent_research_verification_decision::accept ||
            !verification.answer_sufficient || verification.evidence_ids.empty() || verification.confidence <= 0.0) {
        std::fprintf(stderr, "research answer acceptance verification failed: %s\n", error.c_str());
        return 1;
    }
    verification_context.claims.push_back({
        "claim-without-provenance", "This claim has no verified source.", {}, {}, {}, 0.2});
    verification = verifier.verify(result, "researched-answer", verification_context, error);
    if (!error.empty() || verification.decision != common_agent_research_verification_decision::revise_answer ||
            verification.unsupported_claims.size() != 1) {
        std::fprintf(stderr, "research claim provenance verification failed: %s\n", error.c_str());
        return 1;
    }
    verification_context.claims.pop_back();
    verification_context.claims.clear();
    verification_context.claims.push_back({
        "claim-contradicted", "The evidence supports the opposite conclusion.",
        {result.critical_evidence_ids.front()}, {}, {}, 0.4});
    common_agent_research_result contradicted_result = result;
    contradicted_result.evidence.front().relation = common_agent_research_evidence_relation::contradicts;
    verification = verifier.verify(contradicted_result, "researched-answer", verification_context, error);
    if (!error.empty() || verification.decision != common_agent_research_verification_decision::revise_answer ||
            verification.contradicted_claims.size() != 1) {
        std::fprintf(stderr, "research contradiction verification failed: %s\n", error.c_str());
        return 1;
    }
    verification_context.claims.clear();
    verification = verifier.verify(incomplete_result, "researched-answer", error);
    if (!error.empty() || verification.decision != common_agent_research_verification_decision::gather_evidence ||
            !verification.more_research_required) {
        std::fprintf(stderr, "research missing-evidence verification failed: %s\n", error.c_str());
        return 1;
    }
    verification = verifier.verify(result, "", error);
    if (!error.empty() || verification.decision != common_agent_research_verification_decision::revise_answer ||
            !verification.revision_required) {
        std::fprintf(stderr, "research draft-revision verification failed: %s\n", error.c_str());
        return 1;
    }
    common_agent_research_result no_evidence_result = result;
    no_evidence_result.critical_evidence_ids.clear();
    no_evidence_result.coverage.evidence_quality = 0.0;
    verification = verifier.verify(no_evidence_result, "researched-answer", error);
    if (!error.empty() || verification.decision != common_agent_research_verification_decision::fail_with_uncertainty) {
        std::fprintf(stderr, "research uncertainty verification failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("research_answer_verifier=ok\n");

    common_plan_in_memory_store plan_store;
    if (!plan_store.open("", error)) return 1;
    smoke_planner planner;
    smoke_executor executor;
    smoke_reflector reflector;
    common_agent_runtime runtime(plan_store, planner, executor, reflector, &tools);
    common_agent_request request;
    request.prompt = "Find the bounded research answer";
    request.session_id = "runtime-session";
    request.namespace_id = "research-smoke";
    request.turn_id = "runtime-turn";
    request.max_iterations = 1;
    request.max_reflection_rounds = 1;
    request.max_tool_batches = 2;
    common_memory_hit memory_hit;
    memory_hit.memory.id = "memory-smoke-1";
    memory_hit.memory.kind = common_memory_kind::procedure;
    memory_hit.memory.summary = "Use the existing runtime contract.";
    memory_hit.memory.confidence = 0.9f;
    request.memories.push_back(memory_hit);
    request.deliberation_policy = make_common_agent_deliberation_policy(common_agent_thinking_mode::research);
    const auto runtime_result = runtime.run(request);
    bool research_started = false;
    bool research_completed = false;
    bool research_trace = false;
    bool research_trace_context = false;
    bool research_trace_completed = false;
    bool research_task_started = false;
    bool research_task_completed = false;
    bool research_sources_compared = false;
    bool research_trace_task = false;
    bool research_trace_source = false;
    bool research_trace_evidence = false;
    bool research_trace_comparison = false;
    for (const auto & event : runtime_result.events) {
        research_started = research_started || event.type == common_agent_event_type::research_started;
        research_completed = research_completed || event.type == common_agent_event_type::research_completed;
        research_task_started = research_task_started || event.type == common_agent_event_type::research_task_started;
        research_task_completed = research_task_completed || event.type == common_agent_event_type::research_task_completed;
        research_sources_compared = research_sources_compared || event.type == common_agent_event_type::research_sources_compared;
    }
    for (const auto & trace : runtime_result.trace) {
        research_trace = research_trace || trace.stage == common_runtime_trace_stage::research;
        research_trace_context = research_trace_context ||
            trace.stage == common_runtime_trace_stage::research &&
            trace.kind == common_runtime_trace_kind::started &&
            !trace.plan_id.empty() &&
            trace.plan_id == runtime_result.plan_id.value_or("") &&
            !trace.related_id.empty();
        research_trace_completed = research_trace_completed ||
            trace.stage == common_runtime_trace_stage::research &&
            trace.kind == common_runtime_trace_kind::completed;
        research_trace_task = research_trace_task ||
            trace.stage == common_runtime_trace_stage::research &&
            trace.detail.find("task=") != std::string::npos &&
            trace.detail.find("iteration=") != std::string::npos;
        research_trace_source = research_trace_source ||
            trace.stage == common_runtime_trace_stage::research &&
            trace.detail.find("research source recorded source=") != std::string::npos;
        research_trace_evidence = research_trace_evidence ||
            trace.stage == common_runtime_trace_stage::research &&
            trace.detail.find("research evidence recorded evidence=") != std::string::npos;
        research_trace_comparison = research_trace_comparison ||
            trace.stage == common_runtime_trace_stage::research &&
            trace.detail.find("research sources compared comparison=") != std::string::npos;
    }
    if (!runtime_result.error.empty() || !runtime_result.research_result || !runtime_result.research_result->complete ||
            runtime_result.response != "researched-answer" ||
            executor.received_prompt.find("Host-approved research synthesis context:") == std::string::npos ||
            executor.received_prompt.find("Evidence with provenance:") == std::string::npos ||
            !research_started || !research_completed || !research_trace ||
            !research_trace_context || !research_trace_completed ||
            !research_trace_task || !research_trace_source ||
            !research_trace_evidence || !research_trace_comparison ||
            !research_task_started || !research_task_completed || !research_sources_compared || tools.calls != 4 ||
            runtime_result.research_result->established_claim_ids.empty() ||
            runtime_result.research_result->sources.size() != 2 ||
            runtime_result.research_result->evidence.size() != 2 ||
            !runtime_result.research_verification ||
            runtime_result.research_verification->decision != common_agent_research_verification_decision::accept ||
            !runtime_result.research_verification->answer_sufficient) {
        std::fprintf(stderr, "research agent runtime integration failed: %s\n", runtime_result.error.c_str());
        return 1;
    }
    const auto bridged_plan = plan_store.get(runtime_result.plan_id.value_or(""), error);
    bool research_completion_observation = false;
    if (bridged_plan) {
        for (const auto & observation : bridged_plan->observations) {
            research_completion_observation = research_completion_observation ||
                observation.source == "research_workspace" &&
                observation.id.find("research:completion:") == 0 &&
                observation.evidence_ids.size() == runtime_result.research_result->evidence.size();
        }
    }
    if (!error.empty() || !research_completion_observation) {
        std::fprintf(stderr, "research completion was not bridged to the outer plan: %s\n", error.c_str());
        return 1;
    }
    std::printf("research_agent_runtime=ok response=%s\n", runtime_result.response.c_str());

    common_plan_in_memory_store acquisition_guard_plan_store;
    if (!acquisition_guard_plan_store.open("", error)) return 1;
    acquisition_plan_planner acquisition_guard_planner;
    acquisition_plan_executor acquisition_guard_executor;
    smoke_reflector acquisition_guard_reflector;
    acquisition_guard_tools acquisition_guard_tool_runtime;
    common_agent_runtime acquisition_guard_runtime(
        acquisition_guard_plan_store,
        acquisition_guard_planner,
        acquisition_guard_executor,
        acquisition_guard_reflector,
        &acquisition_guard_tool_runtime);
    auto acquisition_guard_request = request;
    acquisition_guard_request.prompt = "Read the supplied resource and return the answer.";
    acquisition_guard_request.turn_id = "research-acquisition-guard-turn";
    acquisition_guard_request.input_resources.clear();
    common_agent_input_resource guarded_input;
    guarded_input.resource.uri = "agent-resource://guarded";
    guarded_input.resource.name = "guarded.txt";
    guarded_input.resource.mime_type = "text/plain";
    guarded_input.resource.scope = common_runtime_resource_scope::turn;
    guarded_input.role = "reference";
    acquisition_guard_request.input_resources.push_back(guarded_input);
    acquisition_guard_request.max_tool_batches = 1;
    acquisition_guard_request.deliberation_policy.require_source_cross_check = false;
    const auto acquisition_guard_result = acquisition_guard_runtime.run(acquisition_guard_request);
    const auto guarded_plan = acquisition_guard_plan_store.get(
        acquisition_guard_result.plan_id.value_or(""), error);
    bool model_acquisition_was_degraded = false;
    if (guarded_plan) {
        for (const auto & step : guarded_plan->steps) {
            model_acquisition_was_degraded = model_acquisition_was_degraded ||
                step.id == "model-acquisition" &&
                common_plan_step_effective_mode(step) == common_plan_step_mode::reasoning &&
                !step.tool_call;
        }
    }
    if (!error.empty() || !acquisition_guard_result.error.empty() ||
            acquisition_guard_result.response != "guarded-answer" ||
            acquisition_guard_tool_runtime.calls != 1 ||
            !model_acquisition_was_degraded) {
        std::fprintf(stderr, "research acquisition ownership guard failed: %s\n",
            acquisition_guard_result.error.c_str());
        return 1;
    }
    std::printf("research_acquisition_guard=ok controller_calls=%d\n",
        acquisition_guard_tool_runtime.calls);

    common_plan_in_memory_store late_research_plan_store;
    if (!late_research_plan_store.open("", error)) return 1;
    deterministic_research_tools late_research_tools;
    smoke_planner late_research_planner;
    smoke_executor late_research_executor;
    late_research_reflector late_research_reflector_instance;
    common_agent_runtime late_research_runtime(
        late_research_plan_store, late_research_planner, late_research_executor,
        late_research_reflector_instance, &late_research_tools);
    auto late_research_request = request;
    late_research_request.prompt = "Return a bounded answer";
    late_research_request.turn_id = "late-research-turn";
    late_research_request.deliberation_policy = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::reflective);
    late_research_request.max_reflection_rounds = 1;
    late_research_request.max_iterations = 1;
    const auto late_research_result = late_research_runtime.run(late_research_request);
    bool late_research_requested = false;
    bool late_research_allowed = false;
    bool late_research_started = false;
    for (const auto & event : late_research_result.events) {
        late_research_requested = late_research_requested ||
            event.type == common_agent_event_type::thinking_escalation_requested;
        late_research_allowed = late_research_allowed ||
            event.type == common_agent_event_type::thinking_escalation_allowed;
        late_research_started = late_research_started ||
            event.type == common_agent_event_type::research_started;
    }
    if (!late_research_result.error.empty() || late_research_result.response != "researched-answer" ||
            !late_research_result.research_result || !late_research_result.research_result->complete ||
            late_research_reflector_instance.calls != 2 || !late_research_requested ||
            !late_research_allowed || !late_research_started) {
        std::fprintf(stderr, "late reflective-to-research escalation smoke failed: %s\n",
            late_research_result.error.c_str());
        return 1;
    }
    std::printf("late_reflective_to_research=ok\n");

    reopen_verifier verifier_for_reopen;
    common_plan_in_memory_store reopen_plan_store;
    if (!reopen_plan_store.open("", error)) return 1;
    common_agent_runtime reopen_runtime(
        reopen_plan_store, planner, executor, reflector, &tools, nullptr, &verifier_for_reopen);
    auto reopen_request = request;
    reopen_request.turn_id = "runtime-reopen-turn";
    const auto reopened_runtime_result = reopen_runtime.run(reopen_request);
    bool research_reopened = false;
    for (const auto & event : reopened_runtime_result.events) {
        research_reopened = research_reopened || event.type == common_agent_event_type::research_reopened;
    }
    if (!reopened_runtime_result.error.empty() || !reopened_runtime_result.research_result ||
            !reopened_runtime_result.research_result->complete || reopened_runtime_result.response != "researched-answer" ||
            verifier_for_reopen.calls != 2 || executor.draft_calls != 3 || !research_reopened || tools.calls != 7) {
        std::fprintf(stderr, "research reopen smoke failed: %s calls=%d drafts=%d event=%d tools=%d response=%s complete=%d\n",
            reopened_runtime_result.error.c_str(), verifier_for_reopen.calls, executor.draft_calls,
            research_reopened ? 1 : 0, tools.calls, reopened_runtime_result.response.c_str(),
            reopened_runtime_result.research_result && reopened_runtime_result.research_result->complete ? 1 : 0);
        return 1;
    }
    std::printf("research_reopen=ok\n");

    common_agent_research_workspace cancelled_workspace;
    cancelled_workspace.workspace_id = "research-cancelled-smoke";
    cancelled_workspace.request_id = "request-cancelled";
    cancelled_workspace.turn_id = "turn-cancelled";
    cancelled_workspace.session_id = "session-1";
    cancelled_workspace.scope.namespace_id = "research-smoke";
    cancelled_workspace.scope.session_id = "session-1";
    cancelled_workspace.objective.objective_id = "cancelled-objective";
    cancelled_workspace.objective.question = "A cancelled research question";
    cancelled_workspace.budget.max_iterations = 2;
    if (!common_agent_research_add_gap(
            cancelled_workspace,
            {"cancelled-gap", "A cancelled gap", "smoke", "evidence addresses the cancelled gap", 1},
            error)) {
        std::fprintf(stderr, "cancelled research smoke setup failed: %s\n", error.c_str());
        return 1;
    }
    const auto cancelled_result = runner.run(
        cancelled_workspace,
        adapter,
        error,
        []() { return true; },
        []() { return common_agent_research_stop_reason::cancelled; });
    if (!error.empty() || cancelled_result.complete ||
            cancelled_result.stop_reason != common_agent_research_stop_reason::cancelled ||
            cancelled_workspace.tool_calls != 0) {
        std::fprintf(stderr, "research cancellation smoke failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("research_cancellation=ok\n");
    return 0;
}
