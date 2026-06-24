#include "agent/agent-runtime.h"
#include "agent/memory-learning.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

class draft_executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state &, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "draft";
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
    common_registered_tool tool;
    tool.name = "lookup";
    tool.executor_id = "test.lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string & input, std::string & output, std::string & handler_error) { output = input; handler_error.clear(); return true; };
    assert(tools.register_tool(std::move(tool), error));
    resume_planner planner;
    draft_executor executor;
    accepting_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector, &tools);
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
            common_plan_operation retry;
            retry.kind = common_plan_operation_kind::activate_step;
            retry.step_id = "fetch";
            retry.reason_summary = "retry transient lookup";
            result.proposed_plan_operations.push_back(std::move(retry));
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
    int calls = 0;
    common_registered_tool tool;
    tool.name = "retry_lookup";
    tool.executor_id = "test.retry_lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [&calls](const std::string &, std::string & output, std::string & handler_error) {
        if (++calls == 1) { handler_error = "temporary network failure"; return false; }
        output = "recovered fact"; handler_error.clear(); return true;
    };
    assert(tools.register_tool(std::move(tool), error));
    repair_planner planner;
    draft_executor executor;
    repairing_reflector reflector;
    common_agent_runtime runtime(store, planner, executor, reflector, &tools);
    common_agent_request request;
    request.prompt = "recover fact";
    request.max_iterations = 2;
    request.max_reflection_rounds = 2;
    request.max_tool_batches = 2;
    const auto result = runtime.run(request);
    assert(result.error.empty() && result.response == "draft");
    assert(result.failures.size() == 1 && result.learning_signals.size() == 2);
    const auto plan = store.get("repair-plan", error);
    assert(plan && plan->status == common_plan_status::completed && plan->observations.size() == 2 && plan->steps[0].status == common_plan_step_status::completed);
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
        proposal.plan.steps = {work};
        proposal.plan.active_step_id = "work";
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

void scenario_learning_promotion() {
    std::string error;
    common_plan_in_memory_store plans;
    common_memory_in_memory_store memories;
    assert(plans.open("", error) && memories.open("", error));
    procedure_extractor extractor;
    common_memory_post_turn_learner learner(memories, extractor,
        [](const std::string &, std::vector<float> & embedding, std::string & embed_error) { embedding = {1.0f}; embed_error.clear(); return true; });
    common_tool_registry tools;
    common_registered_tool tool;
    tool.name = "lookup";
    tool.executor_id = "test.lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string &, std::string & output, std::string & handler_error) { output = "stable result"; handler_error.clear(); return true; };
    assert(tools.register_tool(std::move(tool), error));
    learning_planner planner;
    draft_executor executor;
    accepting_reflector reflector;
    common_agent_runtime runtime(plans, planner, executor, reflector, &tools, &learner);
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

int main() {
    scenario_persistence_resume();
    scenario_repair();
    scenario_learning_promotion();
    return 0;
}
