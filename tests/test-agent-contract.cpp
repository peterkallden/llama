#include "agent/agent-contract.h"

#include <cassert>
#include <type_traits>

int main() {
    static_assert(std::is_same_v<decltype(common_agent_request::messages), std::vector<common_chat_msg>>);
    static_assert(std::is_same_v<decltype(common_agent_result::plan_version), uint64_t>);

    common_agent_request request;
    assert(request.memory_scope == common_memory_scope::session);
    assert(request.plan_scope == common_plan_scope::session);
    assert(request.namespace_id == "local");
    assert(request.session_id == "default");
    assert(!request.enable_memory);
    assert(!request.enable_planning);
    assert(!request.enable_reflection);

    common_agent_result result;
    result.response = "done";
    result.memory_ids.push_back("memory-1");
    result.plan_id = "plan-1";
    result.plan_version = 2;
    result.reflected = true;
    result.revised = true;
    result.events.push_back({common_agent_event_type::reflection_completed, "response checked", {}, result.plan_id});

    assert(result.events.size() == 1);
    assert(result.events.front().type == common_agent_event_type::reflection_completed);
    assert(result.events.front().plan_id == result.plan_id);
    return 0;
}
