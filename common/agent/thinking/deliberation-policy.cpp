#include "agent/thinking/deliberation-policy.h"

#include <algorithm>
#include <cstddef>

const char * common_agent_thinking_mode_name(common_agent_thinking_mode mode) {
    switch (mode) {
        case common_agent_thinking_mode::reflective: return "reflective";
        case common_agent_thinking_mode::deliberate: return "deliberate";
        case common_agent_thinking_mode::research:   return "research";
    }
    return "reflective";
}

const char * common_agent_thinking_request_name(common_agent_thinking_request request) {
    switch (request) {
        case common_agent_thinking_request::auto_select: return "auto";
        case common_agent_thinking_request::reflective: return "reflective";
        case common_agent_thinking_request::deliberate: return "deliberate";
        case common_agent_thinking_request::research:   return "research";
    }
    return "auto";
}

const char * common_agent_escalation_reason_name(common_agent_escalation_reason reason) {
    switch (reason) {
        case common_agent_escalation_reason::none: return "none";
        case common_agent_escalation_reason::multiple_constraints: return "multiple_constraints";
        case common_agent_escalation_reason::external_uncertainty: return "external_uncertainty";
        case common_agent_escalation_reason::resource_comparison_required: return "resource_comparison_required";
        case common_agent_escalation_reason::user_requested_verification: return "user_requested_verification";
        case common_agent_escalation_reason::reflection_requested: return "reflection_requested";
        case common_agent_escalation_reason::policy_denied: return "policy_denied";
        case common_agent_escalation_reason::maximum_mode_reached: return "maximum_mode_reached";
        case common_agent_escalation_reason::escalation_limit_reached: return "escalation_limit_reached";
    }
    return "none";
}

const char * common_agent_reflection_next_action_name(common_agent_reflection_next_action action) {
    switch (action) {
        case common_agent_reflection_next_action::accept:              return "accept";
        case common_agent_reflection_next_action::revise_response:     return "revise_response";
        case common_agent_reflection_next_action::revise_plan:         return "revise_plan";
        case common_agent_reflection_next_action::escalate_deliberate: return "escalate_deliberate";
        case common_agent_reflection_next_action::escalate_research:   return "escalate_research";
        case common_agent_reflection_next_action::fail_bounded:        return "fail_bounded";
    }
    return "accept";
}

static int thinking_mode_rank(common_agent_thinking_mode mode) {
    switch (mode) {
        case common_agent_thinking_mode::reflective: return 0;
        case common_agent_thinking_mode::deliberate: return 1;
        case common_agent_thinking_mode::research: return 2;
    }
    return 0;
}

common_agent_deliberation_policy make_common_agent_deliberation_policy(
        common_agent_thinking_mode mode) {
    common_agent_deliberation_policy policy;
    policy.mode = mode;
    switch (mode) {
        case common_agent_thinking_mode::reflective:
            policy.requested_mode = common_agent_thinking_request::reflective;
            break;
        case common_agent_thinking_mode::deliberate:
            policy.requested_mode = common_agent_thinking_request::deliberate;
            policy.max_reflection_rounds = 2;
            policy.max_plan_revisions = 2;
            policy.require_plan = true;
            policy.require_step_review = true;
            break;
        case common_agent_thinking_mode::research:
            policy.requested_mode = common_agent_thinking_request::research;
            policy.max_reflection_rounds = 2;
            policy.max_plan_revisions = 3;
            policy.max_research_iterations = 4;
            policy.require_plan = true;
            policy.require_step_review = true;
            policy.require_evidence = true;
            policy.require_source_cross_check = true;
            break;
    }
    return policy;
}

common_agent_deliberation_policy make_common_agent_escalated_policy(
        const common_agent_deliberation_policy & current,
        common_agent_thinking_mode target) {
    auto resolved = make_common_agent_deliberation_policy(target);
    resolved.requested_mode = current.requested_mode;
    resolved.allow_escalation = current.allow_escalation;
    resolved.maximum_mode = current.maximum_mode;
    resolved.max_escalations = std::max(0, current.max_escalations - 1);

    // Reflection is a per-mode budget. A positive plan/research limit is also
    // a caller-provided cap and must survive an escalation.
    resolved.max_reflection_rounds = std::max(
        resolved.max_reflection_rounds, current.max_reflection_rounds);
    if (current.max_plan_revisions > 0) {
        resolved.max_plan_revisions = std::min(
            resolved.max_plan_revisions, current.max_plan_revisions);
    }
    resolved.max_tool_rounds = std::max(
        resolved.max_tool_rounds, current.max_tool_rounds);
    if (current.max_research_iterations > 0) {
        resolved.max_research_iterations = std::min(
            resolved.max_research_iterations, current.max_research_iterations);
    }
    return resolved;
}

common_agent_escalation_decision resolve_common_agent_escalation(
        const common_agent_deliberation_policy & policy,
        const common_agent_escalation_signals & signals) {
    common_agent_escalation_decision decision;
    decision.from_mode = policy.mode;
    decision.to_mode = policy.mode;
    decision.escalation_requested =
        signals.multiple_constraints || signals.external_uncertainty ||
        signals.resource_comparison_required || signals.user_requested_verification;
    if (!decision.escalation_requested || policy.mode == common_agent_thinking_mode::research) {
        decision.allowed = false;
        decision.reason = common_agent_escalation_reason::none;
        decision.summary = "no thinking escalation required";
        return decision;
    }
    if (signals.external_uncertainty) {
        decision.to_mode = common_agent_thinking_mode::research;
        decision.reason = common_agent_escalation_reason::external_uncertainty;
    } else if (signals.resource_comparison_required) {
        decision.to_mode = common_agent_thinking_mode::deliberate;
        decision.reason = common_agent_escalation_reason::resource_comparison_required;
    } else if (signals.user_requested_verification) {
        decision.to_mode = common_agent_thinking_mode::deliberate;
        decision.reason = common_agent_escalation_reason::user_requested_verification;
    } else {
        decision.to_mode = common_agent_thinking_mode::deliberate;
        decision.reason = common_agent_escalation_reason::multiple_constraints;
    }
    if (thinking_mode_rank(decision.to_mode) <= thinking_mode_rank(policy.mode)) {
        decision.allowed = false;
        decision.reason = common_agent_escalation_reason::none;
        decision.summary = "requested signals do not require a higher thinking mode";
        return decision;
    }
    if (!policy.allow_escalation) {
        decision.allowed = false;
        decision.reason = common_agent_escalation_reason::policy_denied;
        decision.summary = "thinking escalation denied by host policy";
        return decision;
    }
    if (policy.max_escalations <= 0) {
        decision.allowed = false;
        decision.reason = common_agent_escalation_reason::escalation_limit_reached;
        decision.summary = "thinking escalation denied by escalation limit";
        return decision;
    }
    if (thinking_mode_rank(decision.to_mode) > thinking_mode_rank(policy.maximum_mode)) {
        decision.allowed = false;
        decision.reason = common_agent_escalation_reason::maximum_mode_reached;
        decision.summary = "thinking escalation denied by maximum mode";
        return decision;
    }
    decision.allowed = true;
    decision.summary = std::string("thinking mode escalated from ") +
        common_agent_thinking_mode_name(decision.from_mode) + " to " +
        common_agent_thinking_mode_name(decision.to_mode);
    return decision;
}

common_agent_escalation_decision resolve_common_agent_reflection_escalation(
        const common_agent_deliberation_policy & policy,
        common_agent_reflection_next_action action) {
    common_agent_escalation_decision decision;
    decision.from_mode = policy.mode;
    decision.to_mode = policy.mode;
    decision.escalation_requested =
        action == common_agent_reflection_next_action::escalate_deliberate ||
        action == common_agent_reflection_next_action::escalate_research;
    if (!decision.escalation_requested) {
        decision.summary = "reflection did not request thinking escalation";
        return decision;
    }
    decision.to_mode = action == common_agent_reflection_next_action::escalate_research
        ? common_agent_thinking_mode::research
        : common_agent_thinking_mode::deliberate;
    decision.reason = common_agent_escalation_reason::reflection_requested;
    if (thinking_mode_rank(decision.to_mode) <= thinking_mode_rank(policy.mode)) {
        decision.reason = common_agent_escalation_reason::maximum_mode_reached;
        decision.summary = "reflection escalation does not move to a higher thinking mode";
        return decision;
    }
    if (!policy.allow_escalation) {
        decision.reason = common_agent_escalation_reason::policy_denied;
        decision.summary = "reflection escalation denied by host policy";
        return decision;
    }
    if (policy.max_escalations <= 0) {
        decision.reason = common_agent_escalation_reason::escalation_limit_reached;
        decision.summary = "reflection escalation denied by escalation limit";
        return decision;
    }
    if (thinking_mode_rank(decision.to_mode) > thinking_mode_rank(policy.maximum_mode)) {
        decision.reason = common_agent_escalation_reason::maximum_mode_reached;
        decision.summary = "reflection escalation denied by maximum mode";
        return decision;
    }
    decision.allowed = true;
    decision.summary = std::string("reflection escalated thinking mode from ") +
        common_agent_thinking_mode_name(decision.from_mode) + " to " +
        common_agent_thinking_mode_name(decision.to_mode);
    return decision;
}

bool parse_common_agent_thinking_request(
        const std::string & value,
        common_agent_thinking_request & request) {
    if (value == "auto" || value == "auto_select") {
        request = common_agent_thinking_request::auto_select;
    } else if (value == "reflective") {
        request = common_agent_thinking_request::reflective;
    } else if (value == "deliberate") {
        request = common_agent_thinking_request::deliberate;
    } else if (value == "research") {
        request = common_agent_thinking_request::research;
    } else {
        return false;
    }
    return true;
}

bool resolve_common_agent_deliberation_policy(
        common_agent_thinking_request request,
        common_agent_deliberation_policy & policy,
        std::string & error) {
    common_agent_thinking_mode mode = common_agent_thinking_mode::reflective;
    switch (request) {
        case common_agent_thinking_request::auto_select:
        case common_agent_thinking_request::reflective:
            mode = common_agent_thinking_mode::reflective;
            break;
        case common_agent_thinking_request::deliberate:
            mode = common_agent_thinking_mode::deliberate;
            break;
        case common_agent_thinking_request::research:
            mode = common_agent_thinking_mode::research;
            break;
    }
    policy = make_common_agent_deliberation_policy(mode);
    policy.requested_mode = request;
    error.clear();
    return true;
}

bool resolve_common_agent_deliberation_policy(
        const std::string & value,
        int max_reflection_rounds,
        int max_plan_revisions,
        size_t max_research_iterations,
        common_agent_deliberation_policy & policy,
        std::string & error) {
    common_agent_thinking_request request;
    if (!parse_common_agent_thinking_request(value, request)) {
        error = "unsupported thinking mode: " + value;
        return false;
    }
    if (max_reflection_rounds < 0 || max_plan_revisions < 0) {
        error = "deliberation limits must not be negative";
        return false;
    }
    if (!resolve_common_agent_deliberation_policy(request, policy, error)) {
        return false;
    }
    policy.max_reflection_rounds = max_reflection_rounds;
    policy.max_plan_revisions = max_plan_revisions;
    if (max_research_iterations > 0 ||
            policy.mode == common_agent_thinking_mode::research) {
        policy.max_research_iterations = max_research_iterations;
    }
    return true;
}
