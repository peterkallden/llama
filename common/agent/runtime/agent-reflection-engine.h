#pragma once

#include "agent/agent-generation.h"
#include "agent/contracts/agent-request.h"
#include "agent/thinking/deliberation-policy.h"
#include "plan/plan-types.h"

#include <optional>
#include <string>
#include <vector>

enum class common_reflection_decision { accept, revise, request_action, replan, abort };
struct common_reflection_issue {
    std::string kind, description, correction;
    float severity = 0.5f;
};
struct common_reflection_learning_hint {
    std::string category, statement;
    float expected_reuse = 0.5f;
};
struct common_reflection_result {
    common_reflection_decision decision = common_reflection_decision::accept;
    std::vector<common_reflection_issue> issues;
    std::vector<common_plan_operation> proposed_plan_operations;
    std::optional<std::string> requested_action;
    std::vector<std::string> revision_guidance;
    std::optional<common_reflection_learning_hint> learning_hint;
    common_agent_reflection_next_action next_action = common_agent_reflection_next_action::accept;
    bool ready_to_answer = false;
    float confidence = 0.5f;
    std::optional<common_agent_generated_text_result> generation;
};

class common_reflection_engine {
public:
    virtual ~common_reflection_engine() = default;
    virtual common_reflection_result evaluate(
            const common_agent_request & request,
            const common_plan_state & plan,
            const std::string & draft,
            std::string & error) = 0;
    virtual common_reflection_result evaluate_result(
            const common_agent_request & request,
            const common_plan_state & plan,
            const std::string & draft,
            std::string & error) {
        return evaluate(request, plan, draft, error);
    }
};
