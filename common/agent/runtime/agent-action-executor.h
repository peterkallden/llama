#pragma once

#include "agent/agent-generation.h"
#include "agent/contracts/agent-request.h"
#include "plan/plan-types.h"

#include <string>
#include <vector>

class common_action_executor {
public:
    virtual ~common_action_executor() = default;
    virtual std::string generate_draft(
            const common_agent_request & request,
            const common_plan_state & plan,
            const std::vector<std::string> & guidance,
            std::string & error) = 0;
    virtual common_agent_generated_text_result generate_draft_result(
            const common_agent_request & request,
            const common_plan_state & plan,
            const std::vector<std::string> & guidance,
            std::string & error) {
        common_agent_generated_text_result result;
        result.content = generate_draft(request, plan, guidance, error);
        if (!error.empty()) {
            result.status = common_agent_generation_status::errored;
            result.stop_reason = common_agent_generation_stop_reason::error;
            result.error_message = error;
            return result;
        }
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::none;
        return result;
    }
    virtual std::string generate_reasoning(
            const common_agent_request &, const common_plan_state &,
            const common_plan_step &, std::string & error) {
        error = "reasoning step generation is unavailable";
        return {};
    }
    virtual common_agent_generated_text_result generate_reasoning_result(
            const common_agent_request & request,
            const common_plan_state & plan,
            const common_plan_step & step,
            std::string & error) {
        common_agent_generated_text_result result;
        result.content = generate_reasoning(request, plan, step, error);
        if (!error.empty()) {
            result.status = common_agent_generation_status::errored;
            result.stop_reason = common_agent_generation_stop_reason::error;
            result.error_message = error;
            return result;
        }
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::none;
        return result;
    }
};
