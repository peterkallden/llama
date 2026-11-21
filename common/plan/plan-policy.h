#pragma once
#include "plan/plan-types.h"

struct common_plan_policy_config {
    size_t max_steps = 16, max_dependency_depth = 16, max_observations = 32, max_string_length = 4096;
    bool require_evidence_for_completion = true;
    size_t max_resource_chunk_observations = 1024;
    // Host control proofs have a separate bounded budget on graph workflows.
    // They must not silently consume the evidence capacity of a normal plan.
    size_t max_workflow_control_observations = 128;
};
struct common_plan_policy_result { bool allowed = false; std::string reason; };
bool common_plan_workflow_control_observation(const common_plan_state &, const common_plan_observation &);
struct common_plan_storage_capacity {
    size_t steps = 0, observations = 0, workflow_control = 0;
};
common_plan_storage_capacity common_plan_remaining_capacity(const common_plan_state &, const common_plan_policy_config &);
class common_plan_policy { public: explicit common_plan_policy(common_plan_policy_config config = {});
    common_plan_policy_config limits() const { return config; } common_plan_policy_result validate(const common_plan_state & plan, const common_plan_operation & operation) const; private: common_plan_policy_config config; };
