#include "agent/agent-continuation.h"
#include "agent/agent-contract.h"
#include "tools/agent/runtime/agent-runtime-turn-execution.h"

#include <cassert>

int main() {
    common_plan_state plan;
    plan.id = "plan-1";
    plan.version = 7;

    common_agent_continuation_checkpoint checkpoint;
    checkpoint.checkpoint_id = "checkpoint-1";
    checkpoint.request_id = "request-1";
    checkpoint.turn_id = "turn-1";
    checkpoint.plan_id = plan.id;
    checkpoint.plan_version = plan.version;
    checkpoint.active_step_id = "step-2";
    checkpoint.sequence = 1;
    checkpoint.reason = common_agent_continuation_reason::completion_limit;
    common_agent_dataset_ref dataset;
    dataset.uri = "dataset://sales";
    dataset.name = "sales";
    dataset.row_count = 2;
    dataset.column_count = 3;
    checkpoint.dataset_refs.push_back(dataset);
    checkpoint.working_state = common_agent_working_state{};
    checkpoint.working_state->dataset_refs.push_back(dataset);

    std::string error;
    assert(common_agent_continuation_checkpoint_valid(checkpoint, error));
    assert(common_agent_continuation_checkpoint_matches(
        checkpoint, "request-1", "turn-1", plan, error));
    assert(common_agent_continuation_reason_name(checkpoint.reason) ==
        std::string("completion_limit"));

    checkpoint.plan_version = 8;
    assert(!common_agent_continuation_checkpoint_matches(
        checkpoint, "request-1", "turn-1", plan, error));
    assert(error == "continuation checkpoint does not match the current plan revision");

    checkpoint.plan_version = plan.version;
    checkpoint.request_id = "request-2";
    assert(!common_agent_continuation_checkpoint_matches(
        checkpoint, "request-1", "turn-1", plan, error));
    assert(error == "continuation checkpoint belongs to a different turn");

    checkpoint.request_id = "request-1";
    checkpoint.active_step_id.clear();
    checkpoint.next_action.clear();
    assert(!common_agent_continuation_checkpoint_valid(checkpoint, error));
    assert(error == "continuation checkpoint requires active_step_id or next_action");

    checkpoint.active_step_id = "step-2";
    common_agent_result agent_result;
    agent_result.continuation_checkpoint = checkpoint;
    common_agent_runtime_turn_execution execution;
    execution.continuation_checkpoint = agent_result.continuation_checkpoint;
    execution.continuation_count = 1;
    assert(execution.continuation_checkpoint->checkpoint_id == "checkpoint-1");
    assert(execution.continuation_count == 1);
    assert(execution.continuation_checkpoint->dataset_refs.front().uri == "dataset://sales");
    assert(execution.continuation_checkpoint->working_state->dataset_refs.front().name == "sales");
    return 0;
}
