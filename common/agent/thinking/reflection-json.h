#pragma once
#include "agent/agent-runtime.h"
bool common_reflection_parse_json(
        const std::string & json_text,
        common_reflection_result & result,
        std::string & error,
        size_t max_operations = 8,
        const std::string & inferred_replace_step_id = {});
bool common_reflection_parse_compact_dsl(
        const std::string & dsl_text,
        common_reflection_result & result,
        std::string & error,
        const std::string & inferred_replace_step_id = {},
        size_t max_operations = 8,
        const std::string & inferred_replace_tool_name = {},
        const std::string & inferred_retry_step_id = {},
        const std::vector<common_chat_tool> & tools = {},
        const std::vector<std::pair<std::string, std::string>> & retry_step_choices = {});
