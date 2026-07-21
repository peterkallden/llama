#pragma once

#include "memory/memory-types.h"
#include "plan/plan-types.h"

#include <string>

struct common_agent_scope {
    common_memory_scope memory_scope = common_memory_scope::session;
    common_plan_scope plan_scope = common_plan_scope::turn;
    std::string namespace_id = "local";
    std::string session_id = "default";
    std::string project_id;
    std::string turn_id;
    bool memory_global_opt_in = false;
};

inline void common_agent_scope_apply(const common_agent_scope & scope, common_memory_record & record) {
    record.scope = scope.memory_scope;
    record.namespace_id = scope.namespace_id;
    record.session_id = scope.session_id;
    record.project_id = scope.project_id;
    record.turn_id = scope.turn_id;
}

inline void common_agent_scope_apply(const common_agent_scope & scope, common_memory_query & query) {
    query.scope = scope.memory_scope;
    query.namespace_id = scope.namespace_id;
    query.session_id = scope.session_id;
    query.project_id = scope.project_id;
    query.turn_id = scope.turn_id;
    query.global_opt_in = scope.memory_global_opt_in;
}

