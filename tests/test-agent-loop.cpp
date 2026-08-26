#include "agent/agent-runtime.h"
#include "agent/tooling/registry/tool-registry.h"
#include "agent/learning/memory-learning.h"
#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"
#include "plan/plan-goal.h"
#include "test-tool-runtime-registry-adapter.h"

#include <algorithm>
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

class metadata_executor final : public common_action_executor {
public:
    std::string generate_draft(const common_agent_request &, const common_plan_state &, const std::vector<std::string> &, std::string & error) override {
        error.clear();
        return "draft";
    }
    common_agent_generated_text_result generate_draft_result(
            const common_agent_request &,
            const common_plan_state &,
            const std::vector<std::string> &,
            std::string & error) override {
        error.clear();
        common_agent_generated_text_result result;
        result.content = "draft";
        result.decoded_tokens = 11;
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::eos;
        return result;
    }
    std::string generate_reasoning(const common_agent_request &, const common_plan_state &, const common_plan_step &, std::string & error) override {
        error.clear();
        return R"({"summary":"reasoned"})";
    }
    common_agent_generated_text_result generate_reasoning_result(
            const common_agent_request &,
            const common_plan_state &,
            const common_plan_step &,
            std::string & error) override {
        error.clear();
        common_agent_generated_text_result result;
        result.content = R"({"summary":"reasoned"})";
        result.decoded_tokens = 7;
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::json_schema;
        return result;
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

class metadata_reasoning_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        return create_plan_result(request, error);
    }
    common_plan_proposal create_plan_result(const common_agent_request & request, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "metadata-turn";
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
        proposal.generation = common_agent_generated_text_result{
            R"({"goal":"inspect"})",
            5,
            common_agent_generation_status::completed,
            common_agent_generation_stop_reason::json_schema,
            {},
        };
        return proposal;
    }
};

class mixed_initial_plan_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "mixed-initial";
        proposal.plan.session_id = request.session_id;
        proposal.plan.goal = request.prompt;
        proposal.plan.success_criteria = "answer";
        proposal.plan.status = common_plan_status::active;

        common_plan_step lookup{"lookup", "Lookup", "Get facts"};
        lookup.status = common_plan_step_status::active;
        lookup.selected_tool = "lookup";
        lookup.tool_call = common_plan_tool_call{"lookup", R"({"id":"status"})"};
        common_plan_operation add_lookup;
        add_lookup.kind = common_plan_operation_kind::add_step;
        lookup.status = common_plan_step_status::pending;
        add_lookup.step = std::move(lookup);
        proposal.operations.push_back(std::move(add_lookup));
        return proposal;
    }
};

class invalid_tool_arguments_planner final : public common_planner {
public:
    common_plan_proposal create_plan(const common_agent_request & request, std::string & error) override {
        error.clear();
        common_plan_proposal proposal;
        proposal.plan.id = "invalid-tool-arguments";
        proposal.plan.session_id = request.session_id;
        proposal.plan.goal = request.prompt;
        proposal.plan.success_criteria = "answer";
        common_plan_step step{"lookup", "Lookup", "Get facts"};
        step.status = common_plan_step_status::active;
        step.selected_tool = "lookup";
        step.tool_call = common_plan_tool_call{"lookup", R"({"id":"status","tool":"lookup"})"};
        proposal.plan.steps.push_back(std::move(step));
        proposal.plan.active_step_id = "lookup";
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

class context_overflow_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error = "model reflection generation failed (status=errored, stop=error): request (3174 tokens) exceeds the available context size (3072 tokens)";
        return {};
    }
};

class hint_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request &, const common_plan_state &, const std::string &, std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        result.confidence = 0.9f;
        result.learning_hint = common_reflection_learning_hint{"tool_precondition", "Verify a path before reading it.", 0.8f};
        return result;
    }
};

class metadata_reflector final : public common_reflection_engine {
public:
    common_reflection_result evaluate(const common_agent_request & request, const common_plan_state & plan, const std::string & draft, std::string & error) override {
        return evaluate_result(request, plan, draft, error);
    }
    common_reflection_result evaluate_result(
            const common_agent_request &,
            const common_plan_state &,
            const std::string &,
            std::string & error) override {
        error.clear();
        common_reflection_result result;
        result.decision = common_reflection_decision::accept;
        result.ready_to_answer = true;
        result.generation = common_agent_generated_text_result{
            R"({"decision":"accept"})",
            3,
            common_agent_generation_status::completed,
            common_agent_generation_stop_reason::json_schema,
            {},
        };
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
    tool.handler = [](const std::string &) { return common_tool_execution_result::success("current status"); };
    assert(tools.register_tool(std::move(tool), error));

    planner p;
    executor e;
    reflector r;
    test_tool_runtime_registry_adapter tool_runtime(tools);
    common_agent_runtime runtime(store, p, e, r, &tool_runtime);
    common_agent_request request;
    request.prompt = "answer";
    request.session_id = "s";
    request.namespace_id = "tenant-a";
    request.project_id = "project-a";
    request.plan_scope = common_plan_scope::project;

    const auto created = runtime.run(request);
    assert(created.error.empty() && created.response == "draft");
    const auto completed_plan = store.get("turn-1", error);
    assert(completed_plan && completed_plan->purpose == request.prompt);
    assert(common_plan_evaluate_goal(*completed_plan).evidence_sufficient);
    assert(created.plan_id && *created.plan_id == "turn-1" && created.reflected);
    const auto plan = store.get("turn-1", error);
    assert(plan && plan->scope == common_plan_scope::project && plan->namespace_id == "tenant-a" && plan->project_id == "project-a" && plan->observations.size() == 1 && plan->observations[0].summary == "current status");

    common_plan_in_memory_store mixed_store;
    assert(mixed_store.open("", error));
    mixed_initial_plan_planner mixed_p;
    common_agent_runtime mixed_runtime(mixed_store, mixed_p, e, r, &tool_runtime);
    common_agent_request mixed_request = request;
    mixed_request.plan_id.reset();
    const auto mixed = mixed_runtime.run(mixed_request);
    assert(mixed.error.empty() && mixed.response == "draft");
    const auto mixed_plan = mixed_store.get("mixed-initial", error);
    assert(mixed_plan && mixed_plan->steps.size() == 1 && mixed_plan->steps.front().id == "lookup");
    assert(mixed_plan->observations.size() == 1 && mixed_plan->observations.front().summary == "current status");

    common_plan_in_memory_store objective_store;
    assert(objective_store.open("", error));
    planner objective_planner;
    common_agent_runtime objective_runtime(objective_store, objective_planner, e, r, &tool_runtime);
    common_agent_request objective_request = request;
    objective_request.objective = common_agent_objective{"Explain the current status", "Return a verified status", {"Use the lookup result"}, {"Do not mutate repository state"}};
    const auto objective_result = objective_runtime.run(objective_request);
    assert(objective_result.error.empty());
    const auto objective_plan = objective_store.get("turn-1", error);
    assert(objective_plan && objective_plan->purpose == "Explain the current status" && objective_plan->success_criteria == "Use the lookup result");
    assert(objective_plan->constraints.size() == 1 && objective_plan->constraints.front().description == "Do not mutate repository state");

    request.plan_id = "turn-1";
    const auto resumed = runtime.run(request);
    assert(resumed.error.empty() && resumed.plan_id && *resumed.plan_id == "turn-1");
    assert(p.calls == 1 && std::any_of(resumed.events.begin(), resumed.events.end(), [](const auto & event) {
        return event.detail == "existing plan resumed";
    }));

    common_memory_in_memory_store memories;
    assert(memories.open("", error));
    learner_extractor extractor;
    common_memory_post_turn_learner learner(memories, extractor,
        [](const std::string &, std::vector<float> & embedding, std::string & embed_error) { embedding = {1.0f}; embed_error.clear(); return true; });
    request.project_id = "project-a";
    common_agent_runtime learning_runtime(store, p, e, r, &tool_runtime, &learner);
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
    const auto reasoning_result = reasoning_runtime.run(reasoning_request);
    assert(reasoning_result.error.empty() && reasoning_result.response == "draft");
    const auto reasoning_plan = reasoning_store.get("reasoning-turn", error);
    assert(reasoning_plan && reasoning_plan->observations.size() == 1);
    assert(reasoning_plan->observations.front().summary.find("\"format\":\"unstructured\"") != std::string::npos);

    common_plan_in_memory_store reflection_overflow_store;
    assert(reflection_overflow_store.open("", error));
    context_overflow_reflector overflow_reflector;
    common_agent_runtime reflection_overflow_runtime(
        reflection_overflow_store, reasoning_p, reasoning_e, overflow_reflector);
    const auto reflection_overflow_result = reflection_overflow_runtime.run(reasoning_request);
    assert(reflection_overflow_result.error.empty());
    assert(reflection_overflow_result.response == "draft");
    assert(std::any_of(reflection_overflow_result.trace.begin(), reflection_overflow_result.trace.end(),
        [](const auto & trace) {
            return trace.detail.find("reflection skipped because its context exceeded the model budget") != std::string::npos;
        }));

    common_plan_in_memory_store hint_store;
    assert(hint_store.open("", error));
    hint_reflector hint_r;
    common_agent_runtime hint_runtime(hint_store, reasoning_p, reasoning_e, hint_r);
    common_agent_request hint_request = reasoning_request;
    hint_request.enable_reflection = true;
    const auto hinted = hint_runtime.run(hint_request);
    assert(hinted.error.empty() && hinted.response == "draft");
    assert(hinted.learning_signals.size() == 1 && hinted.learning_signals.front().type == common_learning_signal_type::reflection_hint);
    const auto hinted_plan = hint_store.get("reasoning-turn", error);
    assert(hinted_plan && hinted_plan->observations.size() == 2 && hinted_plan->observations.back().source == "reflection_hint");

    common_plan_in_memory_store correction_store;
    assert(correction_store.open("", error));
    planner correction_p;
    common_agent_runtime correction_runtime(correction_store, correction_p, e, r, &tool_runtime);
    common_agent_request correction_request = request;
    correction_request.plan_id.reset();
    correction_request.user_correction = common_agent_user_correction{"prior-turn-7", "Use the calculator result rather than mental arithmetic."};
    const auto corrected = correction_runtime.run(correction_request);
    assert(corrected.error.empty() && corrected.learning_signals.size() == 1);
    assert(corrected.learning_signals.front().type == common_learning_signal_type::user_correction);
    const auto correction_plan = correction_store.get("turn-1", error);
    assert(correction_plan && correction_plan->observations.size() == 2);
    assert(correction_plan->observations.front().source == "user_correction");
    assert(correction_plan->observations.front().summary.find("prior-turn-7") != std::string::npos);

    common_plan_in_memory_store failure_store;
    assert(failure_store.open("", error));
    common_tool_registry failing_tools;
    common_registered_tool failing_tool;
    failing_tool.name = "lookup";
    failing_tool.arguments_schema = R"({"type":"object","additionalProperties":false,"required":["id"],"properties":{"id":{"type":"string"}}})";
    failing_tool.handler = [](const std::string &) {
        return common_tool_execution_result::failure("tool.lookup.not_found", common_tool_failure_class::not_found, false,
            "Requested resource was not found.", "not found");
    };
    assert(failing_tools.register_tool(std::move(failing_tool), error));
    planner failing_p;
    test_tool_runtime_registry_adapter failing_tool_runtime(failing_tools);
    common_agent_runtime failure_runtime(failure_store, failing_p, e, r, &failing_tool_runtime);
    common_agent_request failure_request;
    failure_request.prompt = "inspect";
    failure_request.session_id = "s";
    failure_request.namespace_id = "tenant-a";
    failure_request.plan_scope = common_plan_scope::session;
    const auto failed_tool_run = failure_runtime.run(failure_request);
    assert(failed_tool_run.error == "final synthesis is blocked by an unrepaired failed tool step: lookup" && failed_tool_run.response.empty());
    assert(failed_tool_run.learning_signals.size() == 1);
    const auto & failure_signal = failed_tool_run.learning_signals.front();
    assert(failure_signal.type == common_learning_signal_type::tool_failure);
    assert(failure_signal.tool_name == "lookup" && failure_signal.evidence_id == "tool:lookup:lookup");
    assert(failed_tool_run.failures.size() == 1);
    assert(failed_tool_run.failures.front().code == "tool.lookup.not_found");
    assert(failed_tool_run.failures.front().classification == common_agent_failure_class::not_found);
    const auto failed_plan = failure_store.get("turn-1", error);
    assert(failed_plan && failed_plan->observations.size() == 1);
    assert(failed_plan->observations.front().summary.find("tool.lookup.not_found") != std::string::npos);
    assert(failed_plan->observations.front().summary.find("\"failure\"") != std::string::npos);

    common_plan_in_memory_store reflection_failure_store;
    assert(reflection_failure_store.open("", error));
    context_overflow_reflector reflection_failure_reflector;
    common_agent_runtime reflection_failure_runtime(
        reflection_failure_store, failing_p, e, reflection_failure_reflector, &failing_tool_runtime);
    const auto reflection_failure_run = reflection_failure_runtime.run(failure_request);
    assert(reflection_failure_run.response.empty());
    assert(reflection_failure_run.error.find("context budget exceeded") != std::string::npos);

    common_plan_in_memory_store invalid_args_store;
    assert(invalid_args_store.open("", error));
    invalid_tool_arguments_planner invalid_args_p;
    common_agent_runtime invalid_args_runtime(invalid_args_store, invalid_args_p, e, r, &tool_runtime);
    common_agent_request invalid_args_request = failure_request;
    const auto invalid_args_run = invalid_args_runtime.run(invalid_args_request);
    assert(invalid_args_run.error.empty() && invalid_args_run.response == "draft");
    assert(invalid_args_run.learning_signals.empty());
    assert(invalid_args_run.failures.empty());
    const auto invalid_args_plan = invalid_args_store.get("invalid-tool-arguments", error);
    assert(invalid_args_plan && invalid_args_plan->observations.size() == 1);
    assert(invalid_args_plan->observations.front().summary == "current status");

    common_plan_in_memory_store metadata_store;
    assert(metadata_store.open("", error));
    metadata_reasoning_planner metadata_planner;
    metadata_executor metadata_exec;
    metadata_reflector metadata_reflect;
    common_agent_runtime metadata_runtime(metadata_store, metadata_planner, metadata_exec, metadata_reflect);
    common_agent_request metadata_request = reasoning_request;
    metadata_request.enable_reflection = true;
    metadata_request.max_iterations = 2;
    metadata_request.max_reflection_rounds = 1;
    const auto metadata_result = metadata_runtime.run(metadata_request);
    assert(metadata_result.error.empty() && metadata_result.response == "draft");
    assert(metadata_result.reasoning_decoded_tokens == 7);
    assert(metadata_result.response_decoded_tokens == 11);
    assert(metadata_result.total_decoded_tokens == 18);
    assert(metadata_result.response_generation_status == common_agent_generation_status::completed);
    assert(metadata_result.response_stop_reason == common_agent_generation_stop_reason::eos);
    assert(metadata_result.generation_records.size() == 4);
    assert(metadata_result.generation_records[0].stage == common_agent_generation_stage::planner);
    assert(metadata_result.generation_records[0].decoded_tokens == 5);
    assert(metadata_result.generation_records[1].stage == common_agent_generation_stage::reasoning);
    assert(metadata_result.generation_records[1].decoded_tokens == 7);
    assert(metadata_result.generation_records[2].stage == common_agent_generation_stage::draft);
    assert(metadata_result.generation_records[2].decoded_tokens == 11);
    assert(metadata_result.generation_records[3].stage == common_agent_generation_stage::reflection);
    assert(metadata_result.generation_records[3].decoded_tokens == 3);
    return 0;
}
