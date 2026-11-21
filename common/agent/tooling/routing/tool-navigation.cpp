#include "agent/tooling/routing/tool-navigation.h"

#include <algorithm>

namespace {

void append_unique(std::vector<std::string> & values, const std::vector<std::string> & additions) {
    for (const auto & value : additions) {
        if (value.empty() || std::find(values.begin(), values.end(), value) != values.end()) continue;
        values.push_back(value);
    }
}

bool has_all_required_evidence(const common_agent_tool_navigation_context & context) {
    return std::all_of(context.required_evidence.begin(), context.required_evidence.end(),
        [&](const auto & required) {
            return std::find(context.completed_evidence.begin(), context.completed_evidence.end(), required) !=
                context.completed_evidence.end();
        });
}

} // namespace

std::string common_agent_tool_family_name(const std::string & tool_name) {
    if (tool_name.empty()) return {};
    const auto separator = tool_name.find('.');
    return separator == std::string::npos ? "utility" : tool_name.substr(0, separator);
}

bool common_agent_tool_navigation_begin(
        common_agent_tool_navigation_context & context,
        common_agent_thinking_mode mode,
        const std::string & operation_id,
        const std::string & plan_id,
        const std::string & step_id,
        const std::string & tool_name,
        const std::vector<std::string> & required_evidence,
        std::string & error) {
    if (operation_id.empty() || plan_id.empty() || step_id.empty() || tool_name.empty()) {
        error = "tool navigation requires operation, plan, step and tool identities";
        return false;
    }
    context = {};
    context.operation_id = operation_id;
    context.plan_id = plan_id;
    context.step_id = step_id;
    context.return_step_id = step_id;
    context.mode = mode;
    context.required_evidence = required_evidence;
    context.state = common_agent_tool_navigation_state::family_active;
    return common_agent_tool_navigation_select_tool(context, tool_name, error);
}

bool common_agent_tool_navigation_select_tool(
        common_agent_tool_navigation_context & context,
        const std::string & tool_name,
        std::string & error) {
    if (tool_name.empty()) { error = "tool navigation requires a non-empty tool"; return false; }
    if (context.state == common_agent_tool_navigation_state::blocked ||
            context.state == common_agent_tool_navigation_state::return_to_plan) {
        error = "tool navigation cannot select a tool after returning or blocking";
        return false;
    }
    const auto family = common_agent_tool_family_name(tool_name);
    if (family.empty()) { error = "tool navigation could not resolve a tool family"; return false; }
    context.current_family = family;
    context.current_tool = tool_name;
    context.pending_operation_id.clear();
    context.state = common_agent_tool_navigation_state::family_active;
    error.clear();
    return true;
}

common_agent_tool_navigation_disposition common_agent_tool_navigation_begin_async(
        common_agent_tool_navigation_context & context,
        const std::string & operation_id,
        std::string & error) {
    if (context.state != common_agent_tool_navigation_state::family_active || operation_id.empty()) {
        error = "tool navigation async wait requires an active tool and operation";
        context.state = common_agent_tool_navigation_state::blocked;
        return common_agent_tool_navigation_disposition::blocked;
    }
    context.pending_operation_id = operation_id;
    context.state = common_agent_tool_navigation_state::awaiting_result;
    error.clear();
    return common_agent_tool_navigation_disposition::await_result;
}

common_agent_tool_navigation_disposition common_agent_tool_navigation_complete_tool(
        common_agent_tool_navigation_context & context,
        bool success,
        const std::vector<std::string> & evidence_ids,
        std::string & error) {
    if (context.state != common_agent_tool_navigation_state::family_active &&
            context.state != common_agent_tool_navigation_state::awaiting_result) {
        error = "tool navigation completion requires an active or awaiting tool";
        context.state = common_agent_tool_navigation_state::blocked;
        return common_agent_tool_navigation_disposition::blocked;
    }
    if (!context.pending_operation_id.empty()) context.pending_operation_id.clear();
    if (!success) {
        context.state = common_agent_tool_navigation_state::blocked;
        error.clear();
        return common_agent_tool_navigation_disposition::blocked;
    }
    append_unique(context.completed_evidence, evidence_ids);
    if (has_all_required_evidence(context)) {
        context.state = common_agent_tool_navigation_state::return_to_plan;
        error.clear();
        return common_agent_tool_navigation_disposition::return_to_plan;
    }
    context.state = common_agent_tool_navigation_state::family_active;
    error.clear();
    return common_agent_tool_navigation_disposition::continue_family;
}

bool common_agent_tool_navigation_return_to_plan(
        common_agent_tool_navigation_context & context,
        std::string & error) {
    if (context.state != common_agent_tool_navigation_state::return_to_plan) {
        error = "tool navigation can return to plan only after required evidence is complete";
        return false;
    }
    context.current_tool.clear();
    context.current_family.clear();
    context.state = common_agent_tool_navigation_state::idle;
    error.clear();
    return true;
}
