#pragma once

#include "agent/agent-context-budgets.h"
#include "agent/contracts/agent-result.h"
#include "agent/runtime/agent-action-executor.h"
#include "agent/runtime/agent-planner.h"
#include "agent/runtime/agent-reflection-engine.h"
#include "agent/runtime/agent-tool-runtime.h"
#include "agent/thinking/research/research-verifier.h"
#include "plan/plan-store.h"

#include <cstddef>
#include <functional>
#include <optional>

class common_memory_post_turn_learner;

using common_agent_context_token_estimator = std::function<std::optional<size_t>(
    const common_agent_request &, const common_plan_state &)>;

class common_agent_runtime {
public:
    common_agent_runtime(
            common_plan_store & store,
            common_planner & planner,
            common_action_executor & executor,
            common_reflection_engine & reflector,
            const common_agent_tool_runtime * tools = nullptr,
            common_memory_post_turn_learner * memory_learner = nullptr,
            const common_agent_research_answer_verifier * research_verifier = nullptr,
            common_agent_context_budget_config context_budgets = {},
            size_t context_size_tokens = 0,
            size_t reserved_output_tokens = 0,
            common_agent_context_token_estimator context_token_estimator = {});

    common_agent_result run(const common_agent_request & request);

private:
    common_plan_store & store;
    common_planner & planner;
    common_action_executor & executor;
    common_reflection_engine & reflector;
    const common_agent_tool_runtime * tools;
    common_memory_post_turn_learner * memory_learner;
    const common_agent_research_answer_verifier * research_verifier;
    common_agent_context_budget_config context_budgets;
    size_t context_size_tokens = 0;
    size_t reserved_output_tokens = 0;
    common_agent_context_token_estimator context_token_estimator;
    common_agent_research_bounded_verifier default_research_verifier;
};
