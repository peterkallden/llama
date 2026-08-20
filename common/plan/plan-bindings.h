#pragma once

#include "plan/plan-json.h"
#include "plan/plan-types.h"

#include <functional>
#include <string>
#include <vector>

struct common_plan_tool_field_contract {
    std::string name;
    std::string semantic_type;
    bool required = false;
    bool collection = false;
};

struct common_plan_tool_dataflow_contract {
    std::string tool_name;
    std::vector<common_plan_tool_field_contract> inputs;
    std::vector<common_plan_tool_field_contract> outputs;
};

using common_plan_tool_dataflow_contract_resolver = std::function<bool(
    const std::string &, common_plan_tool_dataflow_contract &, std::string &)>;

bool common_plan_semantic_types_compatible(
    const std::string & source_type,
    const std::string & target_type);

bool common_plan_dataflow_contract_from_schemas(
    const std::string & tool_name,
    const std::string & input_schema_json,
    const std::string & result_schema_json,
    common_plan_tool_dataflow_contract & contract,
    std::string & error);

bool common_plan_materialize_tool_arguments_contract(
    const common_plan_state & plan,
        const common_plan_step & step,
        const common_plan_tool_arguments_contract & contract,
        common_plan_tool_arguments_contract & materialized_contract,
        std::string & error,
        const common_plan_tool_dataflow_contract_resolver & dataflow_resolver = {});

// Resolves strict {$from_step, $json_pointer} values in a JSON-encoded tool
// argument object from completed tool/reasoning observations. The resulting
// object must still pass the registered tool's normal schema validation.
bool common_plan_materialize_tool_arguments(
    const common_plan_state & plan,
        const common_plan_step & step,
        const std::string & arguments_json,
        std::string & materialized_arguments_json,
        std::string & error,
        const common_plan_tool_dataflow_contract_resolver & dataflow_resolver = {});
