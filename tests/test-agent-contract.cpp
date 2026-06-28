#include "agent/agent-contract.h"

#include <cassert>
int main() {
    static_assert(std::is_same_v<decltype(common_agent_request::messages), std::vector<common_chat_msg>>);

    common_agent_request request;
    assert(request.memory_scope == common_memory_scope::session);
    assert(request.plan_scope == common_plan_scope::turn);
    assert(request.namespace_id == "local");
    assert(request.session_id == "default");
    assert(!request.enable_memory);
    assert(request.enable_planning);
    assert(request.enable_reflection);

    common_agent_result result;
    result.response = "done";
    result.memory_ids.push_back("memory-1");
    result.events.push_back({common_agent_event_type::memory_retrieved, "memory used", "memory-1", std::nullopt});

    assert(result.events.size() == 1);
    assert(result.events.front().type == common_agent_event_type::memory_retrieved);
    assert(result.events.front().memory_id == "memory-1");

    request.memory_scope = common_memory_scope::project;
    request.plan_scope = common_plan_scope::project;
    request.namespace_id = "tenant-a";
    request.session_id = "session-a";
    request.project_id = "project-a";
    request.turn_id = "turn-a";

    const auto scope = common_agent_scope_from_request(request);
    assert(scope.memory_scope == common_memory_scope::project);
    assert(scope.plan_scope == common_plan_scope::project);
    assert(scope.namespace_id == "tenant-a");
    assert(scope.session_id == "session-a");
    assert(scope.project_id == "project-a");
    assert(scope.turn_id == "turn-a");

    common_memory_query query;
    common_agent_scope_apply(scope, query);
    assert(query.scope == common_memory_scope::project);
    assert(query.namespace_id == "tenant-a");
    assert(query.session_id == "session-a");
    assert(query.project_id == "project-a");
    assert(query.turn_id == "turn-a");

    common_memory_record record;
    common_agent_scope_apply(scope, record);
    assert(record.scope == common_memory_scope::project);
    assert(record.namespace_id == "tenant-a");
    assert(record.session_id == "session-a");
    assert(record.project_id == "project-a");
    assert(record.turn_id == "turn-a");

    common_agent_request copied;
    common_agent_scope_apply(scope, copied);
    assert(copied.memory_scope == common_memory_scope::project);
    assert(copied.plan_scope == common_plan_scope::project);
    assert(copied.namespace_id == "tenant-a");
    assert(copied.session_id == "session-a");
    assert(copied.project_id == "project-a");
    assert(copied.turn_id == "turn-a");
    return 0;
}
