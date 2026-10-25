#pragma once
#include "plan/plan-types.h"
#include <nlohmann/json.hpp>

// Shared package/store extension codec, so transports cannot omit grounding
// or thinking policy during persistence and re-export.
void common_plan_workflow_grounding_to_json(
    const common_plan_workflow_definition &, nlohmann::ordered_json &);
bool common_plan_workflow_grounding_from_json(
    const nlohmann::ordered_json &, common_plan_workflow_definition &, std::string &);

// Store codecs preserve host-owned step requirements; model plan JSON does not
// accept this metadata. Older stored calls have no requirement.
void common_plan_tool_result_requirement_to_json(const common_plan_tool_call &, nlohmann::ordered_json &);
void common_plan_tool_result_requirement_from_json(const nlohmann::ordered_json &, common_plan_tool_call &);
