#pragma once

#include "chat.h"

#include <cstddef>
#include <string>
#include <vector>

// A generated model-facing index over an already resolved tool view. It does
// not grant access to tools; the input vector must already be filtered by the
// active host profile and policy.
struct common_tool_family_index {
    std::string id;
    std::string description;
    std::vector<std::string> tool_names;
};

struct common_tool_family_selection {
    bool needs_tools = false;
    std::vector<std::string> family_ids;
};

std::vector<common_tool_family_index> common_generate_tool_family_index(
    const std::vector<common_chat_tool> & tools);

std::string common_render_tool_family_index(
    const std::vector<common_tool_family_index> & families,
    size_t max_chars = 2048);

std::vector<common_chat_tool> common_filter_tools_by_families(
    const std::vector<common_chat_tool> & tools,
    const std::vector<std::string> & family_ids);

std::string common_tool_family_selection_schema();

bool common_parse_tool_family_selection(
    const std::string & json_text,
    common_tool_family_selection & selection,
    std::string & error);
