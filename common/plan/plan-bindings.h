#pragma once

#include "plan/plan-json.h"
#include "plan/plan-types.h"

#include <string>

bool common_plan_materialize_tool_arguments_contract(
    const common_plan_state & plan,
    const common_plan_step & step,
    const common_plan_tool_arguments_contract & contract,
    common_plan_tool_arguments_contract & materialized_contract,
    std::string & error);

// Resolves strict {$from_step, $json_pointer} values in a JSON-encoded tool
// argument object from completed tool/reasoning observations. The resulting
// object must still pass the registered tool's normal schema validation.
bool common_plan_materialize_tool_arguments(
    const common_plan_state & plan,
    const common_plan_step & step,
    const std::string & arguments_json,
    std::string & materialized_arguments_json,
    std::string & error);
