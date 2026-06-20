#include "agent/agent-contract.h"

#include <cassert>
int main() {
    static_assert(std::is_same_v<decltype(common_agent_request::messages), std::vector<common_chat_msg>>);

    common_agent_request request;
    assert(request.memory_scope == common_memory_scope::session);
    assert(request.namespace_id == "local");
    assert(request.session_id == "default");
    assert(!request.enable_memory);

    common_agent_result result;
    result.response = "done";
    result.memory_ids.push_back("memory-1");
    result.events.push_back({common_agent_event_type::memory_retrieved, "memory used", "memory-1"});

    assert(result.events.size() == 1);
    assert(result.events.front().type == common_agent_event_type::memory_retrieved);
    assert(result.events.front().memory_id == "memory-1");
    return 0;
}
