#include "tools/agent/runtime/agent-runtime-execution.h"

#include "memory/memory-in-memory.h"
#include "plan/plan-in-memory.h"

#include <cassert>
#include <deque>
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

} // namespace

int main() {
    test_continuation_is_consumed_in_same_driver_operation();
    test_continuation_budget_emits_checkpoint();
    return 0;
}
