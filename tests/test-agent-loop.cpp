#include "agent/agent-runtime.h"
#include "agent/memory-learning.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cassert>

class planner final : public common_planner {
public:
    int calls = 0;

    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        ++calls;
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "turn-1";
        proposal.plan.session_id = request.session_id;
        proposal.plan.goal = request.prompt;
        proposal.plan.success_criteria = "answer";
        common_plan_step step{"lookup", "Lookup", "Get facts"};
        step.status = common_plan_step_status::active;
        step.selected_tool = "lookup";
        step.tool_call = common_plan_tool_call{"lookup", R"({"id":"status"})"};
        proposal.plan.steps.push_back(std::move(step));
        proposal.plan.active_step_id = "lookup";
        proposal.plan.status = common_plan_status::active;
        return proposal;
    }
};

class executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state &, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "draft";
    }
};

class reasoning_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "reasoning-turn";
        proposal.plan.session_id = request.session_id;
        proposal.plan.goal = request.prompt;
        proposal.plan.success_criteria = "answer";
        common_plan_step orient{"orient", "Orient", "Inspect the request"};
        orient.mode = common_plan_step_mode::reasoning;
        orient.status = common_plan_step_status::active;
        common_plan_step answer{"answer", "Answer", "Return the answer"};
        answer.mode = common_plan_step_mode::final_response;
        answer.depends_on = {"orient"};
        proposal.plan.steps = {orient, answer};
        proposal.plan.active_step_id = "orient";
        proposal.plan.status = common_plan_status::active;
        return proposal;
    }
};

class unstructured_reasoning_executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state &, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "draft";
    }
    std::string generate_reasoning(const common_agent_request &, const common_plan_state &, const common_plan_step &, std::string & error) override {
        error.clear();
        return "inspection completed";
    }
};

class reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        return result;
    }
};

class learner_extractor final : public common_memory_candidate_extractor {
public:
    common_memory_candidate_result extract(const common_agent_request &, const common_plan_state &, const common_agent_result &, std::string & error) override {
        error.clear();
        common_memory_candidate candidate;
        candidate.kind = common_memory_kind::procedure;
        candidate.content = "Verify a persisted plan by resuming it in a later process.";
        candidate.rationale = "Explicit reusable verification rule.";
        candidate.importance = 0.8f;
        candidate.confidence = 0.9f;
        candidate.expected_reuse = 0.8f;
        candidate.source_plan_step_ids = {"lookup"};
        return {{candidate}, "explicit user rule"};
    }
};

int main() {
    common_plan_in_memory_store store;
    std::string error;
    assert(store.open("", error));
    common_tool_registry tools;
    common_registered_tool tool;
    tool.name = "lookup";
    tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    tool.handler = [](const std::string &, std::string & value, std::string & err) { value = "current status"; err.clear(); return true; };
    assert(tools.register_tool(std::move(tool), error));

    planner p;
    executor e;
    reflector r;
    common_agent_runtime runtime(store, p, e, r, &tools);
    common_agent_request request;
    request.prompt = "answer";
    request.session_id = "s";
    request.namespace_id = "tenant-a";
    request.project_id = "project-a";
    request.plan_scope = common_plan_scope::project;

    const auto created = runtime.run(request);
    assert(created.error.empty() && created.response == "draft");
    assert(created.plan_id && *created.plan_id == "turn-1" && created.reflected);
    const auto plan = store.get("turn-1", error);
    assert(plan && plan->scope == common_plan_scope::project && plan->namespace_id == "tenant-a" && plan->project_id == "project-a" && plan->observations.size() == 1 && plan->observations[0].summary == "current status");

    request.plan_id = "turn-1";
    const auto resumed = runtime.run(request);
    assert(resumed.error.empty() && resumed.plan_id && *resumed.plan_id == "turn-1");
    assert(p.calls == 1 && !resumed.events.empty() && resumed.events.front().detail == "existing plan resumed");

    common_memory_in_memory_store memories;
    assert(memories.open("", error));
    learner_extractor extractor;
    common_memory_post_turn_learner learner(memories, extractor,
        [](const std::string &, std::vector<float> & embedding, std::string & embed_error) { embedding = {1.0f}; embed_error.clear(); return true; });
    request.project_id = "project-a";
    common_agent_runtime learning_runtime(store, p, e, r, &tools, &learner);
    const auto learned = learning_runtime.run(request);
    assert(learned.error.empty() && learned.learned_memory_candidate);
    assert(learned.memory_learning_summary.rfind("accepted:", 0) == 0);
    assert(!learned.events.empty() && learned.events.back().type == common_agent_event_type::memory_remembered);

    common_plan_in_memory_store reasoning_store;
    assert(reasoning_store.open("", error));
    reasoning_planner reasoning_p;
    unstructured_reasoning_executor reasoning_e;
    common_agent_runtime reasoning_runtime(reasoning_store, reasoning_p, reasoning_e, r);
    common_agent_request reasoning_request;
    reasoning_request.prompt = "inspect";
    reasoning_request.session_id = "s";
    reasoning_request.namespace_id = "tenant-a";
    reasoning_request.plan_scope = common_plan_scope::session;
    reasoning_request.enable_reflection = false;
    const auto reasoning_result = reasoning_runtime.run(reasoning_request);
    assert(reasoning_result.error.empty() && reasoning_result.response == "draft");
    const auto reasoning_plan = reasoning_store.get("reasoning-turn", error);
    assert(reasoning_plan && reasoning_plan->observations.size() == 1);
    assert(reasoning_plan->observations.front().summary.find("\"format\":\"unstructured\"") != std::string::npos);
    return 0;
}
