#pragma once

#include "plan/plan-types.h"

#include <nlohmann/json.hpp>

#include <string>

struct common_plan_tool_arguments_contract {
    nlohmann::ordered_json value;
};

// Schema is intended for existing llama.cpp JSON-schema-to-grammar helpers.
std::string common_plan_proposal_json_schema();
// Bounded model-facing projection of the strict planner schema. The model
// still returns JSON; this only removes schema/protocol noise from the prompt.
std::string common_render_compact_plan_schema(
    const std::string & schema_json,
    std::string & error);
bool common_plan_parse_tool_arguments_contract_json(
    const std::string & tool_name,
    const std::string & arguments_json,
    common_plan_tool_arguments_contract & contract,
    std::string & error);
bool common_plan_parse_tool_arguments_contract_value(
    const std::string & tool_name,
    const nlohmann::ordered_json & arguments,
    common_plan_tool_arguments_contract & contract,
    std::string & error);
bool common_plan_serialize_tool_arguments_contract_json(
    const std::string & tool_name,
    const common_plan_tool_arguments_contract & contract,
    std::string & arguments_json,
    std::string & error);
bool common_plan_normalize_tool_arguments_json(
    const std::string & tool_name,
    const std::string & arguments_json,
    std::string & normalized_json,
    std::string & error);
// Merge a bounded repair patch into the previous tool arguments. The patch
// must be an object; existing keys are preserved unless explicitly replaced.
bool common_plan_merge_tool_arguments_json(
    const std::string & base_json,
    const std::string & patch_json,
    std::string & merged_json,
    std::string & error);
bool common_plan_parse_proposal_json(
    const std::string & json_text,
    common_plan_state & plan,
    std::vector<common_plan_operation> & operations,
    std::string & error,
    size_t max_operations = 8);
