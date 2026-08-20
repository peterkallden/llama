#include "tools/agent/cli/agent-cli-scope.h"

#include <cassert>

static void test_matching_scope_mapping() {
    args turn_args;
    turn_args.memory_scope = "turn";
    const auto turn_scope = common_cli_make_agent_scope_with_matching_plan_scope(turn_args);
    assert(turn_scope.memory_scope == common_memory_scope::turn);
    assert(turn_scope.plan_scope == common_plan_scope::turn);

    args project_args;
    project_args.memory_scope = "project";
    project_args.memory_project = "repo-1";
    const auto project_scope = common_cli_make_agent_scope_with_matching_plan_scope(project_args);
    assert(project_scope.memory_scope == common_memory_scope::project);
    assert(project_scope.plan_scope == common_plan_scope::project);

    args global_args;
    global_args.memory_scope = "global";
    global_args.memory_global_opt_in = true;
    const auto global_scope = common_cli_make_agent_scope_with_matching_plan_scope(global_args);
    assert(global_scope.memory_scope == common_memory_scope::global);
    assert(global_scope.plan_scope == common_plan_scope::global);
}

static void test_explicit_plan_scope_override() {
    args options;
    options.memory_scope = "project";
    options.memory_namespace = "tenant-a";
    options.memory_session = "session-42";
    options.memory_project = "repo-1";
    options.memory_turn = "turn-7";
    options.memory_global_opt_in = true;

    const auto scope = common_cli_make_agent_scope(options, common_plan_scope::session);
    assert(scope.memory_scope == common_memory_scope::project);
    assert(scope.plan_scope == common_plan_scope::session);
    assert(scope.namespace_id == "tenant-a");
    assert(scope.session_id == "session-42");
    assert(scope.project_id == "repo-1");
    assert(scope.turn_id == "turn-7");
    assert(scope.memory_global_opt_in);
}

static void test_bootstrap_scope_guard() {
    args session_args;
    const auto session_scope = common_cli_make_agent_scope_with_matching_plan_scope(session_args);
    assert(common_cli_supports_bootstrap_package_scope(session_scope));

    args global_args;
    global_args.memory_scope = "global";
    global_args.memory_global_opt_in = true;
    const auto global_scope = common_cli_make_agent_scope_with_matching_plan_scope(global_args);
    assert(!common_cli_supports_bootstrap_package_scope(global_scope));
}

int main() {
    test_matching_scope_mapping();
    test_explicit_plan_scope_override();
    test_bootstrap_scope_guard();
    return 0;
}
