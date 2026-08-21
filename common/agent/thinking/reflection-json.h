#pragma once
#include "agent/agent-runtime.h"
bool common_reflection_parse_json(const std::string & json_text, common_reflection_result & result, std::string & error, size_t max_operations = 8);
