#include "agent/agent-runtime.h"
#include "agent/tooling/registry/tool-registry.h"
#include "agent/learning/memory-learning.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "test-tool-runtime-registry-adapter.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <set>

class draft_executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state &, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "draft";
    }
    std::string generate_reasoning(const common_agent_request &, const common_plan_state &, const common_plan_step &, std::string & error) override {
        error.clear();
        return R"({"summary":"reasoned"})";
    }
};

class accepting_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        return result;
    }
};

class resume_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "resume-plan";
        proposal.plan.goal = "Collect two persisted facts";
        proposal.plan.success_criteria = "Both lookups complete before answering.";
        proposal.plan.status = common_plan_status::active;
        common_plan_step first{"first", "First lookup", "Collect the first fact"};
        first.status = common_plan_step_status::active;
        first.selected_tool = "lookup";
        first.tool_call = common_plan_tool_call{"lookup", R"({"id":"first"})"};
        common_plan_step second{"second", "Second lookup", "Collect the second fact"};
        second.selected_tool = "lookup";
        second.tool_call = common_plan_tool_call{"lookup", R"({"id":"second"})"};
        second.depends_on = {"first"};
        proposal.plan.steps = {first, second};
        proposal.plan.active_step_id = "first";
        return proposal;
    }
};

void scenario_persistence_resume() {
    std::string error;
    common_plan_in_memory_store store;
    assert(store.open("", error));
    common_tool_registry tools;
    test_tool_runtime_registry_adapter tool_runtime(tools);
    common_registered_tool tool;
    tool.name = "lookup";
    tool.executor_id = "test.lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string & input) { return common_tool_execution_result::success(input); };
    assert(tools.register_tool(std::move(tool), error));
    resume_planner planner;
    draft_executor executor;
    accepting_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector, &tool_runtime);
    common_agent_request first;
    first.prompt = "collect two facts";
    first.plan_scope = common_plan_scope::session;
    first.session_id = "resume-session";
    first.enable_reflection = false;
    first.max_tool_batches = 1;
    assert(runtime.run(first).error.empty());
    const auto paused = store.get("resume-plan", error);
    assert(paused && paused->status == common_plan_status::active && paused->steps[0].status == common_plan_step_status::completed && paused->steps[1].status == common_plan_step_status::active);
    common_agent_request second = first;
    second.plan_id = "resume-plan";
    const auto resumed = runtime.run(second);
    assert(resumed.error.empty());
    const auto completed = store.get("resume-plan", error);
    assert(completed && completed->status == common_plan_status::completed && completed->observations.size() == 2);
}

class repair_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "repair-plan";
        proposal.plan.goal = "Recover a transient lookup";
        proposal.plan.success_criteria = "A successful retry produces evidence.";
        proposal.plan.status = common_plan_status::active;
        common_plan_step fetch{"fetch", "Fetch", "Retrieve the required fact"};
        fetch.status = common_plan_step_status::active;
        fetch.selected_tool = "retry_lookup";
        fetch.tool_call = common_plan_tool_call{"retry_lookup", R"({"id":"fact"})"};
        common_plan_step answer{"answer", "Answer", "Report the recovered fact"};
        answer.mode = common_plan_step_mode::final_response;
        answer.depends_on = {"fetch"};
        proposal.plan.steps = {fetch, answer};
        proposal.plan.active_step_id = "fetch";
        return proposal;
    }
};

class repairing_reflector final : public common_reflection_engine {
public:
    int calls = 0;
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        if (++calls == 1) {
            result.decision = common_reflection_decision::revise;
            common_plan_operation reset;
            reset.kind = common_plan_operation_kind::reset_step;
            reset.step_id = "fetch";
            reset.reason_summary = "clear the failed attempt before retrying";
            result.proposed_plan_operations.push_back(std::move(reset));
            common_plan_operation retry;
            retry.kind = common_plan_operation_kind::activate_step;
            retry.step_id = "fetch";
            retry.reason_summary = "retry transient lookup";
            result.proposed_plan_operations.push_back(std::move(retry));
            common_plan_operation assumption;
            assumption.kind = common_plan_operation_kind::add_assumption;
            assumption.reason_summary = "record the transient failure assumption";
            assumption.assumption = common_plan_assumption{
                "transient-network",
                "The lookup failure is transient and retryable.",
                0.8f,
                true,
                {}};
            result.proposed_plan_operations.push_back(std::move(assumption));
        } else {
            result.decision = common_reflection_decision::accept;
            result.ready_to_answer = true;
        }
        return result;
    }
};

void scenario_repair() {
    std::string error;
    common_plan_in_memory_store store;
    assert(store.open("", error));
    common_tool_registry tools;
    test_tool_runtime_registry_adapter tool_runtime(tools);
    int calls = 0;
    common_registered_tool tool;
    tool.name = "retry_lookup";
    tool.executor_id = "test.retry_lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [&calls](const std::string &) {
        if (++calls == 1) {
            return common_tool_execution_result::failure("tool.retry_lookup.temporary_network_failure", common_tool_failure_class::network, true,
                "Lookup failed because the network is temporarily unavailable.", "temporary network failure");
        }
        return common_tool_execution_result::success("recovered fact");
    };
    assert(tools.register_tool(std::move(tool), error));
    repair_planner planner;
    draft_executor executor;
    repairing_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector, &tool_runtime);
    common_agent_request request;
    request.prompt = "recover fact";
    request.max_iterations = 2;
    request.max_reflection_rounds = 2;
    request.max_tool_batches = 2;
    request.deliberation_policy = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::deliberate);
    request.deliberation_policy.max_plan_revisions = 1;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "draft");
    assert(result.revised);
    bool step_reviewed = false;
    bool answer_reviewed = false;
    bool plan_revision_requested = false;
    std::string reviewed_step_id;
    size_t reviewed_step_count = 0;
    for (const auto & event : result.events) {
        if (event.type == common_agent_event_type::step_reviewed) {
            step_reviewed = true;
            ++reviewed_step_count;
            reviewed_step_id = event.step_id;
        }
        answer_reviewed = answer_reviewed || event.type == common_agent_event_type::answer_reviewed;
        plan_revision_requested = plan_revision_requested || event.type == common_agent_event_type::plan_revision_requested;
    }
    assert(step_reviewed && reviewed_step_count >= 1 && !reviewed_step_id.empty() && answer_reviewed && plan_revision_requested);
    assert(result.failures.size() == 1 && result.learning_signals.size() == 2);
    const auto plan = store.get("repair-plan", error);
    assert(plan && plan->status == common_plan_status::completed && plan->observations.size() == 2 && plan->steps[0].status == common_plan_step_status::completed);
    assert(plan->assumptions.size() == 1 && plan->assumptions[0].id == "transient-network" && plan->assumptions[0].valid);
    assert(plan->observations[0].id != plan->observations[1].id);
    assert(plan->observations[1].id.find(":attempt:2") != std::string::npos);
}

void scenario_repair_with_iteration_tool_budget() {
    std::string error;
    common_plan_in_memory_store store;
    assert(store.open("", error));
    common_tool_registry tools;
    test_tool_runtime_registry_adapter tool_runtime(tools);
    int calls = 0;
    common_registered_tool tool;
    tool.name = "retry_lookup";
    tool.executor_id = "test.retry_lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [&calls](const std::string &) {
        if (++calls == 1) {
            return common_tool_execution_result::failure("tool.retry_lookup.temporary_network_failure", common_tool_failure_class::network, true,
                "Lookup failed because the network is temporarily unavailable.", "temporary network failure");
        }
        return common_tool_execution_result::success("recovered fact");
    };
    assert(tools.register_tool(std::move(tool), error));
    repair_planner planner;
    draft_executor executor;
    repairing_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector, &tool_runtime);
    common_agent_request request;
    request.prompt = "recover fact";
    request.max_iterations = 2;
    request.max_reflection_rounds = 2;
    request.max_tool_batches = 1;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "draft");
    const auto plan = store.get("repair-plan", error);
    assert(plan && plan->status == common_plan_status::completed && plan->observations.size() == 2);
}

class replacement_repair_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "replacement-repair-plan";
        proposal.plan.goal = "Repair invalid tool arguments";
        proposal.plan.success_criteria = "The corrected lookup produces evidence.";
        proposal.plan.status = common_plan_status::active;
        common_plan_step fetch{"fetch", "Fetch", "Retrieve the fact with corrected arguments"};
        fetch.status = common_plan_step_status::active;
        fetch.selected_tool = "lookup";
        fetch.tool_call = common_plan_tool_call{"lookup", R"({"tool":"unexpected"})"};
        common_plan_step answer{"answer", "Answer", "Report the repaired fact"};
        answer.mode = common_plan_step_mode::final_response;
        answer.depends_on = {"fetch"};
        proposal.plan.steps = {fetch, answer};
        proposal.plan.active_step_id = "fetch";
        return proposal;
    }
};

class replacing_reflector final : public common_reflection_engine {
public:
    int calls = 0;
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        if (++calls == 1) {
            result.decision = common_reflection_decision::revise;
            common_plan_operation replace;
            replace.kind = common_plan_operation_kind::replace_step;
            replace.step_id = "fetch";
            replace.reason_summary = "replace the invalid tool arguments";
            common_plan_step fetch{"fetch", "Fetch", "Retrieve the fact with corrected arguments"};
            fetch.selected_tool = "lookup";
            fetch.tool_call = common_plan_tool_call{"lookup", R"({"id":"fact"})"};
            replace.step = fetch;
            result.proposed_plan_operations.push_back(std::move(replace));
            common_plan_operation activate;
            activate.kind = common_plan_operation_kind::activate_step;
            activate.step_id = "fetch";
            activate.reason_summary = "run the corrected lookup";
            result.proposed_plan_operations.push_back(std::move(activate));
        } else {
            result.decision = common_reflection_decision::accept;
            result.ready_to_answer = true;
        }
        return result;
    }
};

void scenario_replace_repair() {
    std::string error;
    common_plan_in_memory_store store;
    assert(store.open("", error));
    common_tool_registry tools;
    test_tool_runtime_registry_adapter tool_runtime(tools);
    common_registered_tool tool;
    tool.name = "lookup";
    tool.executor_id = "test.lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string &) { return common_tool_execution_result::success("repaired fact"); };
    assert(tools.register_tool(std::move(tool), error));
    replacement_repair_planner planner;
    draft_executor executor;
    replacing_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector, &tool_runtime);
    common_agent_request request;
    request.prompt = "repair invalid lookup";
    request.max_iterations = 2;
    request.max_reflection_rounds = 2;
    request.max_tool_batches = 2;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "draft");
    assert(result.failures.size() == 1);
    const auto plan = store.get("replacement-repair-plan", error);
    assert(plan && plan->status == common_plan_status::completed && plan->steps[0].status == common_plan_step_status::completed && plan->observations.size() == 2);
    assert(plan->observations[0].summary.find("tool.invalid_arguments") != std::string::npos);
    assert(plan->observations[1].summary == "repaired fact");
}

class learning_planner final : public common_planner {
public:
    int calls = 0;
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "learning-plan-" + std::to_string(++calls);
        proposal.plan.goal = "Verify a reusable lookup procedure";
        proposal.plan.success_criteria = "The lookup result is recorded.";
        proposal.plan.status = common_plan_status::active;
        common_plan_step work{"work", "Lookup", "Perform the reusable lookup"};
        work.status = common_plan_step_status::active;
        work.selected_tool = "lookup";
        work.tool_call = common_plan_tool_call{"lookup", R"({"id":"stable"})"};
        common_plan_step answer{"answer", "Answer", "Report the verified lookup"};
        answer.mode = common_plan_step_mode::final_response;
        answer.depends_on = {"work"};
        proposal.plan.steps = {work, answer};
        proposal.plan.active_step_id = "work";
        return proposal;
    }
};

class two_step_reasoning_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "deliberate-review-plan";
        proposal.plan.goal = "Review both reasoning steps";
        proposal.plan.success_criteria = "Both steps are completed before answering.";
        proposal.plan.status = common_plan_status::active;
        common_plan_step first{"frame", "Frame", "Frame the problem"};
        first.status = common_plan_step_status::active;
        first.mode = common_plan_step_mode::reasoning;
        common_plan_step second{"check", "Check", "Check the constraints"};
        second.mode = common_plan_step_mode::reasoning;
        second.depends_on = {"frame"};
        proposal.plan.steps = {first, second};
        proposal.plan.active_step_id = "frame";
        return proposal;
    }
};

class procedure_extractor final : public common_memory_candidate_extractor {
public:
    common_memory_candidate_result extract(const common_agent_request &, const common_plan_state &, const common_agent_result &, std::string & error) override {
        error.clear();
        common_memory_candidate candidate;
        candidate.kind = common_memory_kind::procedure;
        candidate.content = "Use the stable lookup and preserve its result as evidence.";
        candidate.rationale = "Repeated successful lookup procedure.";
        candidate.importance = 0.8f;
        candidate.confidence = 0.9f;
        candidate.expected_reuse = 0.8f;
        candidate.source_plan_step_ids = {"work"};
        return {{candidate}, "reusable verified procedure"};
    }
};

class revise_without_completion_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        common_plan_operation add_followup;
        add_followup.kind = common_plan_operation_kind::add_step;
        add_followup.reason_summary = "follow up after the visible answer";
        common_plan_step followup{"followup", "Followup", "Do more work after answering"};
        followup.mode = common_plan_step_mode::reasoning;
        add_followup.step = std::move(followup);
        result.proposed_plan_operations.push_back(std::move(add_followup));
        return result;
    }
};

void scenario_learning_promotion() {
    std::string error;
    common_plan_in_memory_store plans;
    common_memory_in_memory_store memories;
    assert(plans.open("", error) && memories.open("", error));
    procedure_extractor extractor;
    common_memory_post_turn_learner learner(memories, extractor,
        [](const std::string &, std::vector<float> & embedding, std::string & embed_error) { embedding = {1.0f}; embed_error.clear(); return true; });
    common_tool_registry tools;
    test_tool_runtime_registry_adapter tool_runtime(tools);
    common_registered_tool tool;
    tool.name = "lookup";
    tool.executor_id = "test.lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string &) { return common_tool_execution_result::success("stable result"); };
    assert(tools.register_tool(std::move(tool), error));
    learning_planner planner;
    draft_executor executor;
    accepting_reflector reflector;
    common_agent_runtime runtime(plans, planner, executor, reflector, &tool_runtime, &learner);
    common_agent_request request;
    request.prompt = "run stable lookup";
    request.namespace_id = "local";
    request.session_id = "session";
    request.project_id = "project";
    request.enable_reflection = false;
    std::string memory_id;
    for (int i = 0; i < 3; ++i) {
        const auto result = runtime.run(request);
        assert(result.error.empty() && result.learned_memory_candidate);
        if (i == 0) for (const auto & event : result.events) {
            if (event.type == common_agent_event_type::memory_remembered) memory_id = event.memory_id;
        }
    }
    assert(!memory_id.empty());
    const auto procedure = memories.get(memory_id, error);
    assert(error.empty() && procedure && procedure->metadata.at("procedure_lifecycle") == "promoted");
    const auto blueprint = plans.get("learned-blueprint:" + memory_id, error);
    assert(blueprint && blueprint->kind == common_plan_kind::blueprint && !blueprint->steps.empty());
}

void scenario_learning_skipped_for_incomplete_plan() {
    std::string error;
    common_plan_in_memory_store plans;
    common_memory_in_memory_store memories;
    assert(plans.open("", error) && memories.open("", error));
    procedure_extractor extractor;
    common_memory_post_turn_learner learner(memories, extractor,
        [](const std::string &, std::vector<float> & embedding, std::string & embed_error) { embedding = {1.0f}; embed_error.clear(); return true; });
    common_tool_registry tools;
    test_tool_runtime_registry_adapter tool_runtime(tools);
    common_registered_tool tool;
    tool.name = "lookup";
    tool.executor_id = "test.lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string &) { return common_tool_execution_result::success("stable result"); };
    assert(tools.register_tool(std::move(tool), error));
    learning_planner planner;
    draft_executor executor;
    revise_without_completion_reflector reflector;
    common_agent_runtime runtime(plans, planner, executor, reflector, &tool_runtime, &learner);
    common_agent_request request;
    request.prompt = "run stable lookup";
    request.namespace_id = "local";
    request.session_id = "session";
    request.project_id = "project";
    request.enable_reflection = true;
    request.max_iterations = 2;
    request.max_reflection_rounds = 1;
    request.max_tool_batches = 1;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "draft");
    assert(result.memory_learning_summary == "skipped: plan did not complete");
    assert(!result.learned_memory_candidate);
    std::string list_error;
    const auto stored = memories.list({}, list_error);
    assert(list_error.empty() && stored.empty());
}

class reflection_tool_degrade_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "reflection-tool-degrade-plan";
        proposal.plan.goal = "Verify reflection tool fallback";
        proposal.plan.success_criteria = "Initial lookup and reflected follow-up both produce evidence.";
        proposal.plan.status = common_plan_status::active;
        common_plan_step fetch{"fetch", "Fetch", "Retrieve the first fact"};
        fetch.status = common_plan_step_status::active;
        fetch.selected_tool = "lookup";
        fetch.tool_call = common_plan_tool_call{"lookup", R"({"id":"first"})"};
        common_plan_step answer{"answer", "Answer", "Prepare an answer before checking reflection follow-up"};
        answer.mode = common_plan_step_mode::final_response;
        answer.depends_on = {"fetch"};
        proposal.plan.steps = {fetch, answer};
        proposal.plan.active_step_id = "fetch";
        return proposal;
    }
};

class invalid_tool_add_reflector final : public common_reflection_engine {
public:
    int calls = 0;
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        if (++calls == 1) {
            result.decision = common_reflection_decision::revise;
            common_plan_operation complete_answer;
            complete_answer.kind = common_plan_operation_kind::complete_step;
            complete_answer.step_id = "answer";
            complete_answer.reason_summary = "free the active synthesis step before adding follow-up";
            result.proposed_plan_operations.push_back(std::move(complete_answer));
            common_plan_operation add;
            add.kind = common_plan_operation_kind::add_step;
            add.reason_summary = "add a follow-up, but with incomplete tool arguments";
            common_plan_step followup{"followup", "Followup", "Check whether another lookup is needed"};
            followup.selected_tool = "lookup";
            followup.tool_call = common_plan_tool_call{"lookup", "{}"};
            add.step = followup;
            result.proposed_plan_operations.push_back(std::move(add));
        } else {
            result.decision = common_reflection_decision::accept;
            result.ready_to_answer = true;
        }
        return result;
    }
};

void scenario_invalid_reflection_tool_degrades_to_reasoning() {
    std::string error;
    common_plan_in_memory_store store;
    assert(store.open("", error));
    common_tool_registry tools;
    test_tool_runtime_registry_adapter tool_runtime(tools);
    common_registered_tool tool;
    tool.name = "lookup";
    tool.executor_id = "test.lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string & input) { return common_tool_execution_result::success(input); };
    assert(tools.register_tool(std::move(tool), error));
    reflection_tool_degrade_planner planner;
    draft_executor executor;
    invalid_tool_add_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector, &tool_runtime);
    common_agent_request request;
    request.prompt = "verify reflection fallback";
    request.max_iterations = 3;
    request.max_reflection_rounds = 2;
    request.max_tool_batches = 1;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "draft");
    bool degraded = false;
    for (const auto & event : result.events) {
        if (event.type == common_agent_event_type::tool_rejected && event.detail.find("degraded to reasoning") != std::string::npos) degraded = true;
    }
    assert(degraded);
    const auto plan = store.get("reflection-tool-degrade-plan", error);
    assert(plan && plan->status == common_plan_status::completed && plan->observations.size() == 2);
    assert(plan->steps.size() == 3 && plan->steps[2].status == common_plan_step_status::completed);
    assert(!plan->steps[2].tool_call && !plan->steps[2].selected_tool && common_plan_step_effective_mode(plan->steps[2]) == common_plan_step_mode::reasoning);
}

class initial_tool_guardrail_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "initial-tool-guardrail-plan";
        proposal.plan.goal = "Handle an incomplete initial tool plan";
        proposal.plan.success_criteria = "The incomplete tool step is handled without a tool validation failure.";
        proposal.plan.status = common_plan_status::active;
        common_plan_step lookup{"lookup", "Lookup", "Retrieve the fact if enough arguments are known"};
        lookup.status = common_plan_step_status::active;
        lookup.selected_tool = "lookup";
        lookup.tool_call = common_plan_tool_call{"lookup", "{}"};
        proposal.plan.steps.push_back(std::move(lookup));
        proposal.plan.active_step_id = "lookup";
        return proposal;
    }
};

void scenario_initial_missing_required_tool_degrades_to_reasoning() {
    std::string error;
    common_plan_in_memory_store store;
    assert(store.open("", error));
    common_tool_registry tools;
    test_tool_runtime_registry_adapter tool_runtime(tools);
    bool tool_called = false;
    common_registered_tool tool;
    tool.name = "lookup";
    tool.executor_id = "test.lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [&tool_called](const std::string &) { tool_called = true; return common_tool_execution_result::success("should not run"); };
    assert(tools.register_tool(std::move(tool), error));
    initial_tool_guardrail_planner planner;
    draft_executor executor;
    accepting_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector, &tool_runtime);
    common_agent_request request;
    request.prompt = "lookup something";
    request.enable_reflection = false;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "draft");
    assert(result.failures.empty() && !tool_called);
    bool degraded = false;
    for (const auto & event : result.events) {
        if (event.type == common_agent_event_type::tool_rejected && event.detail.find("initial tool step degraded to reasoning") != std::string::npos) degraded = true;
    }
    assert(degraded);
    const auto plan = store.get("initial-tool-guardrail-plan", error);
    assert(plan && plan->status == common_plan_status::completed && plan->observations.size() == 1);
    assert(!plan->steps.front().tool_call && !plan->steps.front().selected_tool && common_plan_step_effective_mode(plan->steps.front()) == common_plan_step_mode::reasoning);
}

class deliberate_limit_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request &, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "deliberate-limit-plan";
        proposal.plan.goal = "Stop at the deliberate revision limit";
        proposal.plan.success_criteria = "Return a bounded draft";
        proposal.plan.status = common_plan_status::active;
        common_plan_step answer{"answer", "Answer", "Prepare the bounded answer"};
        answer.status = common_plan_step_status::active;
        answer.mode = common_plan_step_mode::final_response;
        proposal.plan.steps.push_back(std::move(answer));
        proposal.plan.active_step_id = "answer";
        return proposal;
    }
};

class always_replan_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::replan;
        result.revision_guidance.push_back("bounded test replan");
        return result;
    }
};

void scenario_deliberate_revision_limit() {
    std::string error;
    common_plan_in_memory_store store;
    assert(store.open("", error));
    deliberate_limit_planner planner;
    draft_executor executor;
    always_replan_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector);
    common_agent_request request;
    request.prompt = "bounded deliberate work";
    request.max_iterations = 1;
    request.max_reflection_rounds = 1;
    request.deliberation_policy = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::deliberate);
    request.deliberation_policy.max_plan_revisions = 0;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "draft");
    assert(result.limit_reached);
    bool saw_limit_event = false;
    for (const auto & event : result.events) {
        saw_limit_event = saw_limit_event ||
            event.type == common_agent_event_type::plan_revision_limit_reached;
    }
    assert(saw_limit_event);

    common_agent_request invalid_request = request;
    invalid_request.enable_reflection = false;
    invalid_request.max_reflection_rounds = 0;
    const auto invalid_result = runtime.run(invalid_request);
    assert(invalid_result.error == "deliberation mode requires bounded step review");
}

void scenario_deliberate_reviews_each_completed_step() {
    std::string error;
    common_plan_in_memory_store store;
    assert(store.open("", error));
    two_step_reasoning_planner planner;
    draft_executor executor;
    accepting_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector);
    common_agent_request request;
    request.prompt = "review the two-step plan";
    request.max_iterations = 1;
    request.max_reflection_rounds = 1;
    request.deliberation_policy = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::deliberate);
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "draft");

    std::set<std::string> reviewed_steps;
    for (const auto & event : result.events) {
        if (event.type == common_agent_event_type::step_reviewed) {
            reviewed_steps.insert(event.step_id);
        }
    }
    const std::set<std::string> expected_reviewed_steps{"frame", "check"};
    assert(reviewed_steps == expected_reviewed_steps);
}

int main() {
    scenario_persistence_resume();
    scenario_repair_with_iteration_tool_budget();
    scenario_repair();
    scenario_replace_repair();
    scenario_learning_promotion();
    scenario_learning_skipped_for_incomplete_plan();
    scenario_invalid_reflection_tool_degrades_to_reasoning();
    scenario_initial_missing_required_tool_degrades_to_reasoning();
    scenario_deliberate_revision_limit();
    scenario_deliberate_reviews_each_completed_step();
    return 0;
}
