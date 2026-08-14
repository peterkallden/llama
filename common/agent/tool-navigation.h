#pragma once

#include "agent/deliberation-policy.h"

#include <string>
#include <vector>

// A bounded, runtime-local view of where a plan step is inside the host-owned
// tool tree. This is not a scheduler, queue, or second plan representation.
enum class common_agent_tool_navigation_state {
    idle,
    family_active,
    awaiting_result,
    return_to_plan,
    blocked,
};

enum class common_agent_tool_navigation_disposition {
    continue_family,
    await_result,
    return_to_plan,
    blocked,
};

struct common_agent_tool_navigation_context {
    std::string operation_id;
    std::string plan_id;
    std::string step_id;
    common_agent_thinking_mode mode = common_agent_thinking_mode::reflective;
    std::string current_family;
    std::string current_tool;
    std::string pending_operation_id;
    std::string return_step_id;
    std::vector<std::string> required_evidence;
    std::vector<std::string> completed_evidence;
    common_agent_tool_navigation_state state = common_agent_tool_navigation_state::idle;
};

// Uses the host naming convention (the prefix before the first dot). Tools
// without a domain are placed in the bounded "utility" family.
std::string common_agent_tool_family_name(const std::string & tool_name);

bool common_agent_tool_navigation_begin(
        common_agent_tool_navigation_context & context,
        common_agent_thinking_mode mode,
        const std::string & operation_id,
        const std::string & plan_id,
        const std::string & step_id,
        const std::string & tool_name,
        const std::vector<std::string> & required_evidence,
        std::string & error);

bool common_agent_tool_navigation_select_tool(
        common_agent_tool_navigation_context & context,
        const std::string & tool_name,
        std::string & error);

common_agent_tool_navigation_disposition common_agent_tool_navigation_begin_async(
        common_agent_tool_navigation_context & context,
        const std::string & operation_id,
        std::string & error);

common_agent_tool_navigation_disposition common_agent_tool_navigation_complete_tool(
        common_agent_tool_navigation_context & context,
        bool success,
        const std::vector<std::string> & evidence_ids,
        std::string & error);

bool common_agent_tool_navigation_return_to_plan(
        common_agent_tool_navigation_context & context,
        std::string & error);
