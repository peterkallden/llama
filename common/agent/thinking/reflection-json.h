#pragma once
#include "agent/agent-runtime.h"
// Shared by the native schema and the JSON/DSL parser. The plan store's
// independent string limit remains authoritative when an operation is saved.
constexpr size_t common_reflection_next_action_max_chars = 1024;
bool common_reflection_parse_json(
        const std::string & json_text,
        common_reflection_result & result,
        std::string & error,
        size_t max_operations = 8,
        const std::string & inferred_replace_step_id = {},
        const std::vector<std::string> & authorized_step_ids = {});
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

// Compact model review uses the same assurance actions as the JSON contract.
std::string common_reflection_review_grammar();

// Validate stop bookkeeping against the host plan and remove redundant completions.
bool common_reflection_validate_bounded_stop(common_reflection_result &,
        const common_plan_state &, std::string & error);
