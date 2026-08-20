#pragma once

#include "agent-runtime.h"
#include "agent/thinking/research/research-assessor.h"
#include "agent/thinking/research/research-runner.h"

#include <functional>
#include <memory>
#include <optional>
#include <set>

struct common_agent_runtime_turn_context {
    common_agent_request & request;
    common_agent_result & result;
    std::string & error;
    const common_agent_tool_runtime * tools = nullptr;
    common_reflection_engine & reflector;

    std::string research_synthesis_context;
    std::optional<common_agent_research_workspace> research_workspace;
    std::unique_ptr<common_agent_research_bounded_assessor> research_assessor;
    std::unique_ptr<common_agent_research_runtime_adapter> research_adapter;
    std::unique_ptr<common_agent_research_runner> research_runner;

    bool research_reopened = false;
    bool late_escalation_used = false;

    std::function<void(common_agent_event_type, std::string)> emit_event;
    std::function<void(common_agent_event)> emit_full_event;
    std::function<void(
            common_runtime_trace_stage,
            common_runtime_trace_kind,
            std::string,
            std::string,
            std::string)> emit_trace;
    std::function<common_agent_deliberation_policy(
            const common_agent_deliberation_policy &,
            common_agent_thinking_mode)> policy_after_escalation;

    // The outer plan remains the durable source of truth. Research writes a
    // bounded completion observation into it; the workspace remains turn
    // scoped and owns the acquisition details.
    common_plan_store * plan_store = nullptr;
    common_plan_state * outer_plan = nullptr;
};

bool run_common_agent_research_phase(
        common_agent_runtime_turn_context & context);

common_agent_research_lifecycle_sink make_common_agent_research_lifecycle_sink(
        common_agent_runtime_turn_context & context);

bool evaluate_common_agent_reflection_phase(
        common_agent_runtime_turn_context & context,
        common_plan_state & plan,
        const std::string & draft,
        const std::set<std::string> & executed_step_ids,
        common_reflection_result & reflection);

struct common_agent_reflection_escalation_result {
    bool requested = false;
    bool denied = false;
    bool continue_turn = false;
    bool stop_turn = false;
};

common_agent_reflection_escalation_result handle_common_agent_reflection_escalation(
        common_agent_runtime_turn_context & context,
        common_plan_state & plan,
        const common_reflection_result & reflection,
        const std::string & draft,
        std::vector<std::string> & guidance,
        size_t iteration,
        size_t & runtime_iteration_limit);
