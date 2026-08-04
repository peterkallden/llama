#include "tools/agent/runtime/agent-runtime-execution.h"

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
    assert(inference.seen.size() == 3);
    assert(inference.seen[0].purpose == common_agent_generation_purpose::planner);
    assert(inference.seen[1].purpose == common_agent_generation_purpose::draft);
    assert(inference.seen[2].purpose == common_agent_generation_purpose::draft);
}

void test_continuation_budget_emits_checkpoint() {
    fake_inference inference;
    inference.queued = {
        {"not-json", common_agent_generation_stop_reason::none, 1},
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
    assert(inference.seen.size() == 2);
}

void test_cancellation_stops_before_next_continuation_slice() {
    fake_inference inference;
    inference.queued = {
        {"not-json", common_agent_generation_stop_reason::none, 1},
        {"cancelled slice", common_agent_generation_stop_reason::limit, 5},
        {"must not run", common_agent_generation_stop_reason::none, 6},
    };
    const auto cancellation = std::make_shared<common_agent_runtime_cancellation_state>();
    inference.after_generate = [cancellation](size_t count) {
        if (count == 2) cancellation->request_cancel("continuation cancelled by smoke");
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
    assert(inference.seen.size() == 2);
}

void test_deadline_stops_before_next_continuation_slice() {
    fake_inference inference;
    inference.queued = {
        {"not-json", common_agent_generation_stop_reason::none, 1},
        {"deadline slice", common_agent_generation_stop_reason::limit, 5},
        {"must not run", common_agent_generation_stop_reason::none, 6},
    };
    common_agent_runtime_execution_control control;
    control.cancellation = std::make_shared<common_agent_runtime_cancellation_state>();
    control.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    inference.after_generate = [&control](size_t count) {
        if (count == 2) std::this_thread::sleep_for(std::chrono::milliseconds(300));
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
    assert(inference.seen.size() == 2);
}

} // namespace

int main() {
    test_continuation_is_consumed_in_same_driver_operation();
    test_continuation_budget_emits_checkpoint();
    test_cancellation_stops_before_next_continuation_slice();
    test_deadline_stops_before_next_continuation_slice();
    return 0;
}
