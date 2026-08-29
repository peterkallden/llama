#include "agent/adaptation/learning-domain-policy.h"

#include <algorithm>

namespace {

std::string tool_family(const std::string & name) {
    const auto separator = name.find_first_of("._");
    return separator == std::string::npos ? name : name.substr(0, separator);
}

bool has_signal(
        const common_agent_result & result,
        common_learning_signal_type type) {
    return std::any_of(result.learning_signals.begin(), result.learning_signals.end(),
        [type](const auto & signal) { return signal.type == type; });
}

} // namespace

bool common_learning_domain_policy_allows(
        const common_learning_domain_policy & policy,
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result) {
    if (!policy.configured) return true;

    const bool has_plan = !plan.id.empty() || request.plan_id.has_value();
    if (policy.planning && has_plan) return true;

    for (const auto & signal : result.learning_signals) {
        if (signal.tool_name.empty()) continue;
        const auto family = tool_family(signal.tool_name);
        const auto override = policy.tool_use_families.find(family);
        const bool enabled = override == policy.tool_use_families.end()
            ? policy.tool_use : override->second;
        if (enabled) return true;
    }

    if (policy.research && (request.deliberation_policy.mode == common_agent_thinking_mode::research ||
            result.research_result.has_value() || result.research_workspace_checkpoint.has_value())) {
        return true;
    }
    if (policy.procedure_learning &&
            (has_signal(result, common_learning_signal_type::successful_recovery) ||
             has_signal(result, common_learning_signal_type::user_correction))) {
        return true;
    }
    return false;
}
