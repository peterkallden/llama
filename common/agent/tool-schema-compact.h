#pragma once

#include <string>

std::string common_render_compact_tool_schema(
    const std::string & schema_json,
    std::string & error);

std::string common_render_compact_tool_description(
    const std::string & name,
    const std::string & description,
    const std::string & input_schema_json,
    const std::string & result_schema_json,
    std::string & error);
