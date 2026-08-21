#pragma once

#include <string>

enum class common_agent_runtime_mode {
    chat,
    agent,
};

enum class common_agent_thinking_mode {
    reflective,
    deliberate,
    research,
};

enum class common_agent_thinking_request {
    auto_select,
    reflective,
    deliberate,
    research,
};

enum class common_agent_reflection_next_action {
    accept,
    revise_response,
    revise_plan,
    escalate_deliberate,
    escalate_research,
    fail_bounded,
};

struct common_agent_deliberation_policy {
    common_agent_thinking_mode mode = common_agent_thinking_mode::reflective;
    common_agent_thinking_request requested_mode = common_agent_thinking_request::reflective;

    bool allow_escalation = true;
    common_agent_thinking_mode maximum_mode = common_agent_thinking_mode::research;
    int max_escalations = 1;

    int max_reflection_rounds = 1;
    int max_plan_revisions = 0;
    int max_tool_rounds = 16;
    int max_research_iterations = 0;

    bool require_plan = false;
    bool require_step_review = false;
    bool require_evidence = false;
    bool require_source_cross_check = false;
};

enum class common_agent_escalation_reason {
    none,
    multiple_constraints,
    external_uncertainty,
    resource_comparison_required,
    user_requested_verification,
    reflection_requested,
    policy_denied,
    maximum_mode_reached,
    escalation_limit_reached,
};

struct common_agent_escalation_signals {
    bool multiple_constraints = false;
    bool external_uncertainty = false;
    bool resource_comparison_required = false;
    bool user_requested_verification = false;
};

struct common_agent_escalation_decision {
    bool escalation_requested = false;
    bool allowed = false;
    common_agent_thinking_mode from_mode = common_agent_thinking_mode::reflective;
    common_agent_thinking_mode to_mode = common_agent_thinking_mode::reflective;
    common_agent_escalation_reason reason = common_agent_escalation_reason::none;
    std::string summary;
};

const char * common_agent_thinking_mode_name(common_agent_thinking_mode mode);
const char * common_agent_thinking_request_name(common_agent_thinking_request request);
const char * common_agent_escalation_reason_name(common_agent_escalation_reason reason);
const char * common_agent_reflection_next_action_name(common_agent_reflection_next_action action);

common_agent_deliberation_policy make_common_agent_deliberation_policy(
        common_agent_thinking_mode mode);

common_agent_deliberation_policy make_common_agent_escalated_policy(
        const common_agent_deliberation_policy & current,
        common_agent_thinking_mode target);

bool parse_common_agent_thinking_request(
        const std::string & value,
        common_agent_thinking_request & request);

bool resolve_common_agent_deliberation_policy(
        common_agent_thinking_request request,
        common_agent_deliberation_policy & policy,
        std::string & error);

common_agent_escalation_decision resolve_common_agent_escalation(
        const common_agent_deliberation_policy & policy,
        const common_agent_escalation_signals & signals);

common_agent_escalation_decision resolve_common_agent_reflection_escalation(
        const common_agent_deliberation_policy & policy,
        common_agent_reflection_next_action action);

bool resolve_common_agent_deliberation_policy(
        const std::string & value,
        int max_reflection_rounds,
        int max_plan_revisions,
        size_t max_research_iterations,
        common_agent_deliberation_policy & policy,
        std::string & error);
