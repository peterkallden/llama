#include "agent/adaptation/learning-domain-policy.h"

#include <string>

namespace {

common_agent_result tool_result(const std::string & name) {
    common_agent_result result;
    result.learning_signals.push_back({
        common_learning_signal_type::tool_failure, "plan", "step", name, "evidence", "failure"});
    return result;
}

bool test_transport_neutral_tool_family_filter() {
    common_agent_request request;
    common_plan_state plan;
    common_learning_domain_policy policy;
    policy.configured = true;
    policy.tool_use = false;
    policy.tool_use_families["diagnostics"] = true;
    if (!common_learning_domain_policy_allows(policy, request, plan,
            tool_result("diagnostics.compile"))) return false;
    if (common_learning_domain_policy_allows(policy, request, plan,
            tool_result("dataset.inspect"))) return false;
    if (common_learning_domain_policy_allows(policy, request, plan,
            tool_result("mcp_diagnostics"))) return false;
    auto canonical = tool_result("mcp_diagnostics.compile");
    canonical.learning_signals.front().tool_family = "diagnostics";
    canonical.learning_signals.front().provider_kind = "mcp";
    if (!common_learning_domain_policy_allows(policy, request, plan, canonical)) return false;
    return true;
}

bool test_domains_and_legacy_default() {
    common_agent_request request;
    common_plan_state plan;
    plan.id = "plan-1";
    common_agent_result result;
    result.learning_signals.push_back({
        common_learning_signal_type::reflection_hint, "plan-1", "step", {}, "evidence", "hint"});

    common_learning_domain_policy legacy;
    if (!common_learning_domain_policy_allows(legacy, request, plan, result)) return false;
    common_learning_domain_policy planning;
    planning.configured = true;
    planning.planning = true;
    if (!common_learning_domain_policy_allows(planning, request, plan, result)) return false;
    planning.planning = false;
    if (common_learning_domain_policy_allows(planning, request, plan, result)) return false;

    common_learning_domain_policy research;
    research.configured = true;
    research.research = true;
    request.deliberation_policy.mode = common_agent_thinking_mode::research;
    return common_learning_domain_policy_allows(research, request, plan, result);
}

} // namespace

int main() {
    return test_transport_neutral_tool_family_filter() &&
        test_domains_and_legacy_default() ? 0 : 1;
}
