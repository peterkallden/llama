#pragma once

#include "agent/agent-scope.h"
#include "tools/agent/cli/agent-cli-options.h"

inline common_plan_scope common_cli_matching_plan_scope(common_memory_scope scope) {
    switch (scope) {
        case common_memory_scope::turn:    return common_plan_scope::turn;
        case common_memory_scope::session: return common_plan_scope::session;
        case common_memory_scope::project: return common_plan_scope::project;
        case common_memory_scope::global:  return common_plan_scope::global;
    }
    return common_plan_scope::turn;
}

inline common_memory_scope common_cli_memory_scope(const args & a) {
    common_memory_scope scope = common_memory_scope::session;
    common_memory_scope_parse(a.memory_scope, scope);
    return scope;
}

inline common_agent_scope common_cli_make_agent_scope(const args & a, common_plan_scope plan_scope) {
    common_agent_scope scope;
    scope.memory_scope = common_cli_memory_scope(a);
    scope.plan_scope = plan_scope;
    scope.namespace_id = a.memory_namespace;
    scope.session_id = a.memory_session;
    scope.project_id = a.memory_project;
    scope.turn_id = a.memory_turn;
    scope.memory_global_opt_in = a.memory_global_opt_in;
    return scope;
}

inline common_agent_scope common_cli_make_agent_scope_with_matching_plan_scope(const args & a) {
    const auto memory_scope = common_cli_memory_scope(a);
    return common_cli_make_agent_scope(a, common_cli_matching_plan_scope(memory_scope));
}

inline bool common_cli_supports_bootstrap_package_scope(const common_agent_scope & scope) {
    return scope.memory_scope == common_memory_scope::session || scope.memory_scope == common_memory_scope::project;
}
