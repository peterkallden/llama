#pragma once

#include "agent-context-budgets.h"
#include "agent-contract.h"
#include "input-resources.h"
#include "memory/memory-context.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

// Host-owned compaction of the sections that the current driver actually
// carries between inference slices. It reuses plan/working-state, policy-pack
// and resource contracts; it is not a conversation store or a second memory
// database.
struct common_agent_context_compaction_limits {
    common_agent_working_state_limits working_state;
    size_t max_input_resources = 32;
    common_memory_policy_pack_render_config policy_pack;
};

struct common_agent_context_compaction_result {
    common_agent_working_state working_state;
    std::optional<common_memory_policy_pack> policy_pack;
    std::vector<common_agent_input_resource> input_resources;
    size_t dropped_input_resources = 0;
};

inline common_agent_context_compaction_result compact_common_agent_context(
        const common_plan_state & plan,
        const std::optional<common_memory_policy_pack> & policy_pack,
        const std::vector<common_agent_input_resource> & input_resources,
        const common_agent_context_compaction_limits & limits = {}) {
    common_agent_context_compaction_result result;
    result.working_state = make_common_agent_working_state(plan, limits.working_state);
    if (policy_pack) {
        result.policy_pack = common_memory_compact_policy_pack(*policy_pack, limits.policy_pack);
    }

    for (const auto & input : input_resources) {
        const bool duplicate = std::any_of(
            result.input_resources.begin(), result.input_resources.end(),
            [&](const auto & existing) {
                return existing.resource.uri == input.resource.uri;
            });
        if (duplicate || result.input_resources.size() >= limits.max_input_resources) {
            ++result.dropped_input_resources;
            continue;
        }
        result.input_resources.push_back(input);
    }
    return result;
}
