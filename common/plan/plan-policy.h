#pragma once
#include "plan/plan-types.h"

struct common_plan_policy_config { size_t max_steps = 16, max_dependency_depth = 16, max_observations = 32, max_string_length = 4096; bool require_evidence_for_completion = true; };
struct common_plan_policy_result { bool allowed = false; std::string reason; };
class common_plan_policy { public: explicit common_plan_policy(common_plan_policy_config config = {}); common_plan_policy_result validate(const common_plan_state & plan, const common_plan_operation & operation) const; private: common_plan_policy_config config; };
