#include "tools/agent/runtime/agent-runtime-execution.h"
#include "agent/context-pressure.h"
#include "agent/context-compaction.h"
#include "agent/agent-working-state.h"

#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cassert>
#include <chrono>
#include <deque>
#include <functional>
#include <thread>
#include <string>
#include <vector>

namespace {

struct queued_generation {
    std::string content;
    common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::none;
    int decoded_tokens = 0;
};

class fake_inference final : public common_agent_inference {
public:
    std::deque<queued_generation> queued;
    std::vector<common_agent_generation_request> seen;
    std::function<void(size_t)> after_generate;

    bool generate(
            const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
        seen.push_back(request);
        if (queued.empty()) return false;
        const auto next = queued.front();
        queued.pop_front();
        result = {};
        result.content = next.content;
        result.stop_reason = next.stop_reason;
        result.decoded_tokens = next.decoded_tokens;
        result.status = common_agent_generation_status::completed;
        if (after_generate) after_generate(seen.size());
        return true;
    }
};

common_agent_runtime_driver_execution make_execution(
        common_memory_store & memories,
        common_plan_store & plans,
        fake_inference & inference,
        std::string & current_plan_id,
        common_agent_runtime_config runtime_config,
        common_agent_scope scope,
        const std::vector<common_blueprint_candidate> & blueprints,
        const std::vector<common_memory_hit> & hits,
        const common_agent_runtime_tooling & tooling) {
    common_agent_runtime_policy policy;
    policy.enable_reflection = false;
    policy.max_iterations = 1;
    common_agent_orchestration_config orchestration;
    orchestration.prompt = "Continue bounded work";
    orchestration.agent_plan = "off";
    return {
        memories,
        plans,
        inference,
        policy,
        std::move(runtime_config),
        std::move(orchestration),
        current_plan_id,
        std::move(scope),
        blueprints,
        std::nullopt,
        hits,
        common_memory_scope::session,
        true,
        tooling,
    };
}

void test_continuation_is_consumed_in_same_driver_operation() {
    fake_inference inference;
    inference.queued = {
        {"not-json", common_agent_generation_stop_reason::none, 1},
        {"still-not-json", common_agent_generation_stop_reason::none, 1},
        {"first slice", common_agent_generation_stop_reason::limit, 5},
        {"second slice", common_agent_generation_stop_reason::none, 6},
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    common_agent_scope scope;
    scope.namespace_id = "continuation-smoke";
    scope.session_id = "continuation-session";
    scope.turn_id = "continuation-turn";
    scope.memory_scope = common_memory_scope::session;
    scope.plan_scope = common_plan_scope::turn;

    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    const common_agent_runtime_tooling tooling;
    std::string current_plan_id;
    common_agent_runtime_config runtime_config;
    runtime_config.generation_config.n_predict = 64;
    runtime_config.max_continuations = 1;
    auto execution = make_execution(
        memories, plans, inference, current_plan_id, std::move(runtime_config),
        scope, blueprints, hits, tooling);

    common_agent_result result;
    assert(run_agent_runtime_driver(execution, result, error));
    assert(error.empty());
    assert(result.error.empty());
    assert(result.response == "first slice\nsecond slice");
    assert(result.response_stop_reason == common_agent_generation_stop_reason::none);
    assert(!result.continuation_checkpoint);
    assert(result.plan_id);
    assert(inference.seen.size() == 4);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[1].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[1].messages.size() == 2);
    assert(inference.seen[1].messages[1].content.find("[Regeneration]") != std::string::npos);
    assert(inference.seen[2].purpose == common_agent_generation_purpose::draft);
    assert(inference.seen[3].purpose == common_agent_generation_purpose::draft);
}

void test_continuation_budget_emits_checkpoint() {
    fake_inference inference;
    inference.queued = {
        {"not-json", common_agent_generation_stop_reason::none, 1},
        {"still-not-json", common_agent_generation_stop_reason::none, 1},
        {"limited", common_agent_generation_stop_reason::limit, 5},
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));

    common_agent_scope scope;
    scope.namespace_id = "continuation-smoke";
    scope.session_id = "continuation-session";
    scope.turn_id = "checkpoint-turn";
    scope.memory_scope = common_memory_scope::session;
    scope.plan_scope = common_plan_scope::turn;
    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    const common_agent_runtime_tooling tooling;
    std::string current_plan_id;
    common_agent_runtime_config runtime_config;
    runtime_config.generation_config.n_predict = 64;
    runtime_config.max_continuations = 0;
    auto execution = make_execution(
        memories, plans, inference, current_plan_id, std::move(runtime_config),
        scope, blueprints, hits, tooling);

    common_agent_result result;
    assert(run_agent_runtime_driver(execution, result, error));
    assert(error.empty());
    assert(result.response == "limited");
    assert(result.response_stop_reason == common_agent_generation_stop_reason::limit);
    assert(result.continuation_checkpoint);
    assert(result.continuation_checkpoint->turn_id == "checkpoint-turn");
    assert(result.continuation_checkpoint->plan_id == *result.plan_id);
    assert(result.continuation_checkpoint->working_state);
    assert(!result.continuation_checkpoint->working_state->goal.empty());
    assert(!result.continuation_checkpoint->working_state->current_phase.empty());
    assert(inference.seen.size() == 3);
}

void test_working_state_projection_is_bounded_and_preserves_refs() {
    common_plan_state plan;
    plan.goal = "Preserve exact repository identifiers while compacting active context.";
    plan.next_action = "resume verification";
    plan.constraints.push_back({"no-unrelated-changes", "Do not modify unrelated workspace files.", true});
    plan.assumptions.push_back({"build-available", "A reproducible build is available.", 0.2f, false, {}});
    common_plan_step completed;
    completed.id = "inspect";
    completed.title = "Inspect current state";
    completed.status = common_plan_step_status::completed;
    completed.result_summary = "Found exact target and preserved commit identifier.";
    plan.steps.push_back(completed);
    common_plan_step active;
    active.id = "verify";
    active.title = "Verify bounded change";
    active.status = common_plan_step_status::active;
    active.mode = common_plan_step_mode::reasoning;
    plan.steps.push_back(active);
    plan.active_step_id = active.id;
    common_plan_observation observation;
    observation.id = "obs-1";
    observation.source = "resource_chunk";
    observation.summary = "Chunk observation";
    common_runtime_resource_ref resource;
    resource.uri = "agent-resource://authoritative";
    resource.name = "source.txt";
    resource.mime_type = "text/plain";
    resource.size_bytes = 1024;
    resource.lineage.parent_uri = "agent-resource://parent";
    resource.lineage.chunk_index = 0;
    resource.lineage.chunk_count = 2;
    resource.lineage.byte_length = 512;
    resource.lineage.derivation = "smoke";
    observation.resource_refs.push_back(resource);
    plan.observations.push_back(observation);

    const auto state = make_common_agent_working_state(plan, 1024);
    const auto rendered = render_common_agent_working_state(state, 160);
    assert(state.goal.find("repository") != std::string::npos);
    assert(state.current_phase == "reasoning");
    assert(!state.completed_steps.empty());
    assert(!state.constraints.empty());
    assert(!state.open_questions.empty());
    assert(state.resource_refs.size() == 1);
    assert(!state.chunk_status.empty());
    assert(rendered.size() <= 160);

    common_agent_working_state_limits tight_limits;
    tight_limits.max_total_chars = 64;
    tight_limits.max_resource_refs = 0;
    tight_limits.max_chunk_status = 0;
    tight_limits.max_tool_results = 0;
    const auto tight_state = make_common_agent_working_state(plan, tight_limits);
    assert(tight_state.resource_refs.empty());
    assert(tight_state.chunk_status.empty());
    assert(tight_state.tool_results.empty());
    assert(tight_state.goal.size() <= tight_limits.max_value_chars);
}

void test_context_compaction_reuses_existing_state_and_resource_contracts() {
    common_plan_state plan;
    plan.id = "compaction-plan";
    plan.goal = "Keep the active goal and authoritative resource references.";
    plan.next_action = "resume synthesis";

    common_agent_input_resource first;
    first.resource.uri = "agent-resource://authoritative";
    first.resource.name = "source.txt";
    first.required = true;
    auto duplicate = first;
    duplicate.required = false;
    common_agent_input_resource second;
    second.resource.uri = "agent-resource://second";
    second.resource.name = "second.txt";

    common_agent_context_compaction_limits limits;
    limits.max_input_resources = 2;
    limits.working_state.max_total_chars = 256;
    const auto compacted = compact_common_agent_context(
        plan, std::nullopt, {first, duplicate, second}, limits);
    assert(compacted.working_state.goal == plan.goal);
    assert(compacted.working_state.continuation_action == "resume synthesis");
    assert(compacted.input_resources.size() == 2);
    assert(compacted.input_resources.front().required);
    assert(compacted.dropped_input_resources == 1);
}

void test_cancellation_stops_before_next_continuation_slice() {
    fake_inference inference;
    inference.queued = {
        {"not-json", common_agent_generation_stop_reason::none, 1},
        {"still-not-json", common_agent_generation_stop_reason::none, 1},
        {"cancelled slice", common_agent_generation_stop_reason::limit, 5},
        {"must not run", common_agent_generation_stop_reason::none, 6},
    };
    const auto cancellation = std::make_shared<common_agent_runtime_cancellation_state>();
    inference.after_generate = [cancellation](size_t count) {
        if (count == 3) cancellation->request_cancel("continuation cancelled by smoke");
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));
    common_agent_scope scope;
    scope.namespace_id = "continuation-smoke";
    scope.session_id = "continuation-session";
    scope.turn_id = "cancel-turn";
    scope.memory_scope = common_memory_scope::session;
    scope.plan_scope = common_plan_scope::turn;
    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    const common_agent_runtime_tooling tooling;
    std::string current_plan_id;
    common_agent_runtime_config runtime_config;
    runtime_config.generation_config.n_predict = 64;
    runtime_config.max_continuations = 1;
    auto execution = make_execution(
        memories, plans, inference, current_plan_id, std::move(runtime_config),
        scope, blueprints, hits, tooling);
    execution.execution_control.cancellation = cancellation;

    common_agent_result result;
    assert(!run_agent_runtime_driver(execution, result, error));
    assert(result.response == "cancelled slice");
    assert(result.response_generation_status == common_agent_generation_status::cancelled);
    assert(result.response_stop_reason == common_agent_generation_stop_reason::cancelled);
    assert(error == "continuation cancelled by smoke");
    assert(inference.seen.size() == 3);
}

void test_deadline_stops_before_next_continuation_slice() {
    fake_inference inference;
    inference.queued = {
        {"not-json", common_agent_generation_stop_reason::none, 1},
        {"still-not-json", common_agent_generation_stop_reason::none, 1},
        {"deadline slice", common_agent_generation_stop_reason::limit, 5},
        {"must not run", common_agent_generation_stop_reason::none, 6},
    };
    common_agent_runtime_execution_control control;
    control.cancellation = std::make_shared<common_agent_runtime_cancellation_state>();
    control.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    inference.after_generate = [&control](size_t count) {
        if (count == 3) std::this_thread::sleep_for(std::chrono::milliseconds(300));
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));
    common_agent_scope scope;
    scope.namespace_id = "continuation-smoke";
    scope.session_id = "continuation-session";
    scope.turn_id = "deadline-turn";
    scope.memory_scope = common_memory_scope::session;
    scope.plan_scope = common_plan_scope::turn;
    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    const common_agent_runtime_tooling tooling;
    std::string current_plan_id;
    common_agent_runtime_config runtime_config;
    runtime_config.generation_config.n_predict = 64;
    runtime_config.max_continuations = 1;
    auto execution = make_execution(
        memories, plans, inference, current_plan_id, std::move(runtime_config),
        scope, blueprints, hits, tooling);
    execution.execution_control = control;

    common_agent_result result;
    assert(!run_agent_runtime_driver(execution, result, error));
    assert(result.response == "deadline slice");
    assert(result.response_generation_status == common_agent_generation_status::cancelled);
    assert(result.response_stop_reason == common_agent_generation_stop_reason::cancelled);
    assert(error == "turn deadline exceeded");
    assert(inference.seen.size() == 3);
}

void test_context_pressure_reserves_completion_and_tool_space() {
    common_agent_context_budget budget;
    budget.context_limit_tokens = 1000;
    budget.estimated_input_tokens = 300;
    budget.reserved_output_tokens = 200;
    budget.reserved_tool_tokens = 100;
    budget.safety_margin_tokens = 100;
    auto evaluation = evaluate_common_agent_context_pressure(budget);
    assert(evaluation.valid);
    assert(evaluation.pressure == common_agent_context_pressure::normal);
    assert(evaluation.available_input_tokens == 600);

    budget.estimated_input_tokens = 500;
    evaluation = evaluate_common_agent_context_pressure(budget);
    assert(evaluation.pressure == common_agent_context_pressure::compact_recommended);

    budget.estimated_input_tokens = 600;
    evaluation = evaluate_common_agent_context_pressure(budget);
    assert(evaluation.pressure == common_agent_context_pressure::continuation_required);

    budget.context_limit_tokens = 0;
    evaluation = evaluate_common_agent_context_pressure(budget);
    assert(!evaluation.valid);
    assert(evaluation.pressure == common_agent_context_pressure::continuation_required);
}

void test_context_pressure_stops_before_draft_and_checkpoints() {
    fake_inference inference;
    inference.queued = {
        {R"({"goal":"Continue bounded work","steps":[{"id":"context-check","mode":"reasoning","objective":"Check the compact context"}]})",
            common_agent_generation_stop_reason::none, 1},
        {R"({"goal":"Continue bounded work","steps":[{"id":"context-check","mode":"reasoning","objective":"Check the compact context"}]})",
            common_agent_generation_stop_reason::none, 1},
    };

    common_memory_in_memory_store memories;
    common_plan_in_memory_store plans;
    std::string error;
    assert(memories.open("", error));
    assert(plans.open("", error));
    common_agent_scope scope;
    scope.namespace_id = "continuation-smoke";
    scope.session_id = "continuation-session";
    scope.turn_id = "context-pressure-turn";
    scope.memory_scope = common_memory_scope::session;
    scope.plan_scope = common_plan_scope::turn;
    const std::vector<common_blueprint_candidate> blueprints;
    const std::vector<common_memory_hit> hits;
    const common_agent_runtime_tooling tooling;
    std::string current_plan_id;
    common_agent_runtime_config runtime_config;
    runtime_config.generation_config.n_predict = 64;
    runtime_config.generation_config.context_size_tokens = 1024;
    runtime_config.generation_config.context_budgets.working_state.max_total_chars = 96;
    runtime_config.generation_config.context_budgets.working_state.max_value_chars = 24;
    runtime_config.generation_config.context_budgets.working_state.max_completed_steps = 1;
    runtime_config.generation_config.context_budgets.working_state.max_remaining_steps = 1;
    runtime_config.generation_config.context_budgets.working_state.max_constraints = 1;
    runtime_config.generation_config.context_budgets.working_state.max_open_questions = 1;
    runtime_config.generation_config.context_budgets.working_state.max_resource_refs = 1;
    runtime_config.generation_config.context_budgets.working_state.max_chunk_status = 1;
    runtime_config.generation_config.context_budgets.working_state.max_tool_results = 1;
    runtime_config.max_continuations = 1;
    size_t estimator_calls = 0;
    runtime_config.context_token_estimator = [&estimator_calls](
            const common_agent_request &, const common_plan_state &) -> std::optional<size_t> {
        ++estimator_calls;
        return 1000;
    };
    auto execution = make_execution(
        memories, plans, inference, current_plan_id, std::move(runtime_config),
        scope, blueprints, hits, tooling);
    execution.orchestration_config.prompt = "short context-pressure request";

    common_agent_result result;
    assert(run_agent_runtime_driver(execution, result, error));
    assert(error.empty());
    assert(result.error.empty());
    assert(result.response.empty());
    assert(estimator_calls > 0);
    assert(result.limit_reached);
    assert(result.continuation_checkpoint);
    assert(inference.seen.size() == 2);
    assert(result.continuation_checkpoint->working_state);
    assert(!result.continuation_checkpoint->working_state->goal.empty());
    assert(!result.continuation_checkpoint->working_state->current_phase.empty());
    assert(result.continuation_checkpoint->working_state->goal.size() <= 24);
    assert(result.continuation_checkpoint->working_state->completed_steps.size() <= 1);
    assert(result.continuation_checkpoint->working_state->remaining_steps.size() <= 1);
    bool saw_pressure = false;
    for (const auto & trace : result.trace) {
        saw_pressure = saw_pressure ||
            trace.detail.find("context pressure requires") != std::string::npos;
    }
    assert(saw_pressure);
}

} // namespace

int main() {
    test_continuation_is_consumed_in_same_driver_operation();
    test_continuation_budget_emits_checkpoint();
    test_working_state_projection_is_bounded_and_preserves_refs();
    test_context_compaction_reuses_existing_state_and_resource_contracts();
    test_cancellation_stops_before_next_continuation_slice();
    test_deadline_stops_before_next_continuation_slice();
    test_context_pressure_reserves_completion_and_tool_space();
    test_context_pressure_stops_before_draft_and_checkpoints();
    return 0;
}
