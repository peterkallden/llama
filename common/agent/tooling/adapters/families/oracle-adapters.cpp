#include "agent/tooling/adapters/families/oracle-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"

bool common_try_register_oracle_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed=false;
    if(definition.executor_id!="builtin.oracle_ruleset_define"&&
            definition.executor_id!="builtin.oracle_action_verify"&&
            definition.executor_id!="builtin.oracle_goal_search"&&
            definition.executor_id!="builtin.oracle_problem_define")return false;
    if(bindings.semi_oracle) {
        installed=common_adapter_register_definition(definition,registry,
            [bindings, name=definition.name](const std::string & input) {
                return bindings.semi_oracle(name,input);
            },error, definition.executor_id=="builtin.oracle_goal_search");
    }
    return true;
}
