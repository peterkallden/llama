#pragma once

#include "agent/agent-continuation.h"
#include "agent/agent-generation.h"
#include "agent/agent-scope.h"
#include "agent/agent-working-state.h"
#include "agent/contracts/agent-events.h"
#include "agent/contracts/agent-learning.h"
#include "agent/thinking/deliberation-policy.h"
#include "agent/thinking/research/research-contract.h"
#include "agent/tooling/contracts/tool-runtime-contract.h"
#include "memory/memory-candidate.h"
#include "memory/memory-policy-pack.h"
#include "memory/memory-types.h"
#include "plan/plan-types.h"
#include "chat.h"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

struct common_agent_objective {
    std::string purpose;
    std::string desired_outcome;
    std::vector<std::string> success_criteria;
    std::vector<std::string> constraints;
};

struct common_agent_input_resource {
    common_runtime_resource_ref resource;
    std::string role = "reference";
    bool required = false;
};

struct common_agent_request {
    std::vector<common_chat_msg> messages;
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;
    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;
    std::optional<std::string> plan_id;
    bool enable_memory = false;
    bool enable_planning = true;
    bool enable_reflection = true;
    common_agent_deliberation_policy deliberation_policy;
    std::string prompt;
    std::vector<common_agent_input_resource> input_resources;
    std::optional<common_agent_working_state> working_state;
    std::optional<common_agent_objective> objective;
    std::optional<common_memory_policy_pack> policy_pack;
    std::vector<common_memory_hit> memories;
    std::optional<common_agent_user_correction> user_correction;
    std::optional<common_memory_candidate> explicit_memory_candidate;
    bool explicit_memory_confirmed = false;
    std::optional<common_agent_tool_call> tool_call;
    common_agent_event_sink event_sink;
    size_t max_iterations = 2;
    size_t max_reflection_rounds = 1;
    size_t max_tool_batches = 1;
    bool allow_policy_gated_tool_proposals = false;
    std::function<bool()> research_should_stop;
    std::function<common_agent_research_stop_reason()> research_stop_reason;
};

inline common_agent_scope common_agent_scope_from_request(const common_agent_request & request) {
    common_agent_scope scope;
    scope.memory_scope = request.memory_scope;
    scope.plan_scope = request.plan_scope;
    scope.namespace_id = request.namespace_id;
    scope.session_id = request.session_id;
    scope.project_id = request.project_id;
    scope.turn_id = request.turn_id;
    return scope;
}

inline void common_agent_scope_apply(const common_agent_scope & scope, common_agent_request & request) {
    request.memory_scope = scope.memory_scope;
    request.plan_scope = scope.plan_scope;
    request.namespace_id = scope.namespace_id;
    request.session_id = scope.session_id;
    request.project_id = scope.project_id;
    request.turn_id = scope.turn_id;
}
