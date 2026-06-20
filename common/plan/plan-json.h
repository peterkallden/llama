#pragma once

#include "plan/plan-types.h"

#include <string>

// Schema is intended for existing llama.cpp JSON-schema-to-grammar helpers.
std::string common_plan_proposal_json_schema();
bool common_plan_parse_proposal_json(
    const std::string & json_text,
    common_plan_state & plan,
    std::vector<common_plan_operation> & operations,
    std::string & error,
    size_t max_operations = 8);
