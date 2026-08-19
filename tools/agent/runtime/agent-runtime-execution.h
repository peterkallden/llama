#pragma once

#include "../runtime/agent-runtime-tooling.h"
#include "../tooling/agent-tool-provider.h"
#include "agent/agent-contract.h"
#include "agent/agent-inference.h"
#include "agent/learning/blueprint-selector.h"
#include "agent/tooling/registry/tool-registry.h"
#include "../runtime/agent-plan-orchestration.h"
#include "../runtime/agent-runtime-session.h"
#include "memory/memory-store.h"
#include "plan/plan-store.h"
#include "tools/agent/resource/agent-resource-chunker.h"

#include <string>
#include <vector>

struct common_agent_runtime_policy {
    std::string agent_inference_backend = "cli";
    std::string tool_profile;
    std::string memory_learn = "off";
    bool memory_learn_show_candidate = false;
    bool plan_show_summary = false;
    bool agent_trace = false;
    bool enable_reflection = false;
    size_t max_iterations = 1;
    size_t max_reflection_rounds = 0;
    size_t max_tool_rounds = 0;
    bool allow_policy_gated_tool_proposals = false;
    common_agent_deliberation_policy deliberation_policy =
        make_common_agent_deliberation_policy(common_agent_thinking_mode::reflective);
};

struct common_agent_runtime_policy_build_config {
    std::string agent_inference_backend = "cli";
    std::string tool_profile;
    std::string memory_learn = "off";
    bool memory_learn_show_candidate = false;
    bool plan_show_summary = false;
    bool agent_trace = false;
    size_t max_tool_rounds = 0;
    std::map<std::string, std::vector<std::string>> tool_capabilities;
    std::map<std::string, common_tool_profile> tool_profiles;
};

common_agent_runtime_policy make_agent_runtime_policy(
    common_agent_runtime_policy_build_config config);

// Driver inputs are runtime-owned contracts, stores, and prebuilt scope/tool state.
struct common_agent_runtime_driver_inputs {
    common_memory_store & memory_store;
    common_plan_store & plan_store;
    common_agent_inference_options inference_options;
    common_agent_runtime_policy policy;
    common_agent_runtime_config runtime_config;
    common_agent_orchestration_config orchestration_config;
    std::string & current_plan_id;
    const common_agent_scope & scope;
    const std::vector<common_blueprint_candidate> & installed_blueprint_candidates;
    std::optional<common_memory_policy_pack> policy_pack;
    const std::vector<common_memory_hit> & memories;
    common_memory_scope memory_scope = common_memory_scope::session;
    bool memory_enabled = false;
    const std::string & fallback_reason;
    const common_agent_runtime_tooling & tooling;
    std::vector<common_agent_input_resource> input_resources;
    std::function<bool()> research_should_stop;
    std::function<common_agent_research_stop_reason()> research_stop_reason;
    std::optional<common_memory_candidate> explicit_memory_candidate;
    bool explicit_memory_confirmed = false;
    common_agent_runtime_execution_control execution_control;
};

struct common_agent_runtime_driver_execution {
    common_memory_store & memory_store;
    common_plan_store & plan_store;
    common_agent_inference & inference;
    common_agent_runtime_policy policy;
    common_agent_runtime_config runtime_config;
    common_agent_orchestration_config orchestration_config;
    std::string & current_plan_id;
    common_agent_scope scope;
    const std::vector<common_blueprint_candidate> & installed_blueprint_candidates;
    std::optional<common_memory_policy_pack> policy_pack;
    const std::vector<common_memory_hit> & memories;
    common_memory_scope memory_scope = common_memory_scope::session;
    bool memory_enabled = false;
    const common_agent_runtime_tooling & tooling;
    std::vector<common_agent_input_resource> input_resources;
    std::optional<common_agent_working_state> compact_working_state;
    std::function<bool()> research_should_stop;
    std::function<common_agent_research_stop_reason()> research_stop_reason;
    std::optional<common_memory_candidate> explicit_memory_candidate;
    bool explicit_memory_confirmed = false;
    std::vector<common_agent_event> pre_turn_events;
    std::vector<common_runtime_trace_entry> pre_turn_trace;
    common_agent_runtime_execution_control execution_control;
    std::vector<agent_resource_chunk_plan> resource_chunk_plans;
    // Keep the authoritative input view so the lane can restore it for the
    // synthesis slice after the last bounded chunk has been processed.
    std::vector<common_agent_input_resource> resource_chunk_original_inputs;
    bool resource_chunk_observations_prepared = false;
    // Only one resource chunk chain is active at a time.  The original
    // resource remains authoritative in the store; this is the transient
    // session-lane cursor for the bounded input view.
    size_t active_resource_chunk_plan = static_cast<size_t>(-1);
    size_t active_resource_chunk_input = static_cast<size_t>(-1);
    size_t active_resource_chunk_index = 0;
};

common_agent_runtime_driver_execution make_agent_runtime_driver_execution(
    common_agent_runtime_driver_inputs & inputs,
    common_agent_inference & inference);

common_agent_request make_agent_runtime_driver_request(
    const common_agent_runtime_driver_execution & execution);

bool run_agent_runtime_driver_session(
    common_agent_runtime_driver_inputs & inputs,
    common_agent_runtime_session & session,
    common_agent_result & result,
    std::string & error);

bool run_agent_runtime_driver(
    common_agent_runtime_driver_execution & execution,
    common_agent_result & result,
    std::string & error);
