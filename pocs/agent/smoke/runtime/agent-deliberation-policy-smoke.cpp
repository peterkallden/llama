#include "agent/thinking/deliberation-policy.h"

#include <cstdio>

static int fail(const char * message) {
    std::fprintf(stderr, "deliberation policy smoke failed: %s\n", message);
    return 1;
}

int main() {
    if (common_agent_runtime_mode::chat == common_agent_runtime_mode::agent) {
        return fail("chat and agent runtime modes collapsed");
    }

    const auto reflective = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::reflective);
    if (reflective.mode != common_agent_thinking_mode::reflective ||
            reflective.max_reflection_rounds != 1 ||
            reflective.max_plan_revisions != 0 ||
            reflective.require_plan ||
            reflective.require_evidence) {
        return fail("reflective baseline is incorrect");
    }

    const auto deliberate = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::deliberate);
    if (deliberate.mode != common_agent_thinking_mode::deliberate ||
            deliberate.max_reflection_rounds != 2 ||
            deliberate.max_plan_revisions != 2 ||
            !deliberate.require_plan ||
            !deliberate.require_step_review ||
            deliberate.require_evidence) {
        return fail("deliberate policy is incorrect");
    }

    const auto research = make_common_agent_deliberation_policy(
        common_agent_thinking_mode::research);
    if (research.mode != common_agent_thinking_mode::research ||
            research.max_research_iterations != 4 ||
            !research.require_plan ||
            !research.require_step_review ||
            !research.require_evidence ||
            !research.require_source_cross_check) {
        return fail("research policy is incorrect");
    }

    common_agent_thinking_request request;
    if (!parse_common_agent_thinking_request("auto", request) ||
            request != common_agent_thinking_request::auto_select) {
        return fail("auto request did not parse");
    }

    common_agent_deliberation_policy resolved;
    std::string error;
    if (!resolve_common_agent_deliberation_policy(request, resolved, error) ||
            resolved.mode != common_agent_thinking_mode::reflective ||
            !error.empty()) {
        return fail("auto request did not resolve to reflective");
    }

    if (!resolve_common_agent_deliberation_policy(
            "auto", 2, 2, 4, resolved, error)) {
        return fail("bounded auto policy did not resolve");
    }
    const auto auto_research = make_common_agent_escalated_policy(
        resolved, common_agent_thinking_mode::research);
    if (auto_research.max_reflection_rounds != 2 ||
            auto_research.max_plan_revisions != 2 ||
            auto_research.max_research_iterations != 4) {
        return fail("configured caps were not preserved during escalation");
    }
    const auto auto_deliberate = make_common_agent_escalated_policy(
        resolved, common_agent_thinking_mode::deliberate);
    if (auto_deliberate.max_plan_revisions != 2) {
        return fail("plan revision cap was not preserved for deliberate mode");
    }

    common_agent_escalation_signals signals;
    signals.multiple_constraints = true;
    auto escalation = resolve_common_agent_escalation(reflective, signals);
    if (!escalation.escalation_requested || !escalation.allowed ||
            escalation.to_mode != common_agent_thinking_mode::deliberate ||
            escalation.reason != common_agent_escalation_reason::multiple_constraints) {
        return fail("reflective did not escalate to deliberate");
    }
    signals = {};
    signals.external_uncertainty = true;
    escalation = resolve_common_agent_escalation(deliberate, signals);
    if (!escalation.allowed || escalation.to_mode != common_agent_thinking_mode::research ||
            escalation.reason != common_agent_escalation_reason::external_uncertainty) {
        return fail("deliberate did not escalate to research");
    }
    auto denied_policy = reflective;
    denied_policy.allow_escalation = false;
    escalation = resolve_common_agent_escalation(denied_policy, {true, false, false, false});
    if (escalation.allowed || escalation.reason != common_agent_escalation_reason::policy_denied) {
        return fail("policy-denied escalation was allowed");
    }
    auto capped_policy = reflective;
    capped_policy.maximum_mode = common_agent_thinking_mode::reflective;
    escalation = resolve_common_agent_escalation(capped_policy, {true, false, false, false});
    if (escalation.allowed || escalation.reason != common_agent_escalation_reason::maximum_mode_reached) {
        return fail("maximum-mode escalation was allowed");
    }

    escalation = resolve_common_agent_reflection_escalation(
        reflective, common_agent_reflection_next_action::escalate_deliberate);
    if (!escalation.allowed || escalation.to_mode != common_agent_thinking_mode::deliberate ||
            escalation.reason != common_agent_escalation_reason::reflection_requested) {
        return fail("reflection did not escalate to deliberate");
    }
    capped_policy = reflective;
    capped_policy.maximum_mode = common_agent_thinking_mode::reflective;
    escalation = resolve_common_agent_reflection_escalation(
        capped_policy, common_agent_reflection_next_action::escalate_research);
    if (escalation.allowed || escalation.reason != common_agent_escalation_reason::maximum_mode_reached) {
        return fail("reflection maximum-mode escalation was allowed");
    }

    return 0;
}
