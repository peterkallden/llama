#include "agent/tool-navigation.h"

#include <cassert>

int main() {
    common_agent_tool_navigation_context context;
    std::string error;
    assert(common_agent_tool_family_name("statistics.value_counts") == "statistics");
    assert(common_agent_tool_family_name("resource_read") == "utility");
    assert(common_agent_tool_navigation_begin(context, common_agent_thinking_mode::research,
        "operation-1", "plan-1", "step-1", "resource_read", {"r1", "r2"}, error));
    assert(context.current_family == "utility");
    assert(common_agent_tool_navigation_begin_async(context, "runtime-op-1", error) ==
        common_agent_tool_navigation_disposition::await_result);
    assert(common_agent_tool_navigation_complete_tool(context, true, {"r1"}, error) ==
        common_agent_tool_navigation_disposition::continue_family);
    assert(common_agent_tool_navigation_select_tool(context, "resource_inspect", error));
    assert(common_agent_tool_navigation_complete_tool(context, true, {"r2", "r2"}, error) ==
        common_agent_tool_navigation_disposition::return_to_plan);
    assert(common_agent_tool_navigation_return_to_plan(context, error));
    assert(context.state == common_agent_tool_navigation_state::idle);

    assert(common_agent_tool_navigation_begin(context, common_agent_thinking_mode::deliberate,
        "operation-2", "plan-2", "step-2", "statistics.describe", {}, error));
    assert(common_agent_tool_navigation_complete_tool(context, false, {}, error) ==
        common_agent_tool_navigation_disposition::blocked);
    assert(!common_agent_tool_navigation_select_tool(context, "statistics.value_counts", error));
    return 0;
}
