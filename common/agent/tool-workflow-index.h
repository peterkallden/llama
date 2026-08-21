#pragma once

#include <cstddef>
#include <string>
#include <vector>

// A compact model-facing composition hint. It describes how registered tools
// are normally combined without exposing host-owned plan IR or dependencies.
struct common_tool_workflow {
    std::string id;
    std::string description;
    std::vector<std::string> family_ids;
    std::vector<std::string> tool_names;
    std::vector<std::string> steps;
    std::vector<std::string> rules;
};

struct common_tool_workflow_selection {
    std::vector<std::string> workflow_ids;
};

std::vector<common_tool_workflow> common_generate_tool_workflow_index();

std::string common_render_tool_workflow_index(
        const std::vector<common_tool_workflow> & workflows,
        const std::vector<std::string> & family_ids,
        size_t max_chars = 4096);

std::string common_render_selected_tool_workflows(
        const std::vector<common_tool_workflow> & workflows,
        const std::vector<std::string> & workflow_ids,
        size_t max_chars = 4096);

std::string common_tool_workflow_selection_schema();

bool common_parse_tool_workflow_selection(
        const std::string & json_text,
        common_tool_workflow_selection & selection,
        std::string & error);
