#pragma once

#include "agent/tooling/contracts/tool-runtime-contract.h"
#include "chat.h"

#include <string>
#include <vector>

// Public profile choices are json and dsl. The json choice maps to native
// structured output; jsonl remains a legacy textual dialect.
enum class common_agent_tool_output_format {
    native,
    jsonl,
    compact_dsl,
};

const char * common_agent_tool_output_format_name(
        common_agent_tool_output_format format);

bool common_parse_agent_tool_output_format(
        const std::string & value,
        common_agent_tool_output_format & format,
        std::string & error);

// Compact DSL supports bounded scalar, array and object arguments. Schemas
// outside that value grammar remain available through native/JSONL profiles.
bool common_compact_dsl_schema_supported(
        const std::string & schema_json,
        std::string & reason);

bool common_parse_model_tool_call(
        common_agent_tool_output_format format,
        const std::string & text,
        common_agent_tool_call & call,
        std::string & error);

// Parse the compact multi-step planner form into the canonical JSON proposal
// consumed by the ordinary plan parser and host validation pipeline.
// Grammar: `plan goal="..."`, then one `step [as=alias] | open! TOOL ...`
// line per operation. Steps are sequential unless the host adds dependencies.
bool common_parse_compact_dsl_plan(
        const std::string & text,
        std::string & proposal_json,
        std::string & error);

// Normalize a string that the caller's schema explicitly defines as a
// comma-separated list. Whitespace inside ordinary scalar strings is never
// changed implicitly.
bool common_normalize_comma_separated_string(
        const std::string & value,
        std::string & normalized,
        std::string & error);

// Renders a model-facing instruction block from the same common_chat_tool
// definitions used by the host. Unsupported compact-DSL tools are omitted;
// callers can inspect the returned supported count through the optional
// unsupported_names output.
std::string common_render_model_tool_output_instructions(
        common_agent_tool_output_format format,
        const std::vector<common_chat_tool> & tools,
        std::vector<std::string> * unsupported_names,
        std::string & error);
