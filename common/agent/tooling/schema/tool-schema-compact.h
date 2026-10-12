#pragma once

#include <string>
#include <vector>

struct common_model_tool_field {
    std::string name;
    std::string description;
    std::string semantic_type;
    std::string display_type;
    std::string role;
    bool required = false;
    bool may_be_inferred = false;
};

struct common_model_tool_contract {
    std::string name;
    std::string purpose;
    std::vector<common_model_tool_field> inputs;
    std::vector<common_model_tool_field> outputs;
};

common_model_tool_contract common_project_model_tool_contract(
    const std::string & name,
    const std::string & description,
    const std::string & input_schema_json,
    const std::string & result_schema_json,
    std::string & error);

// Add host-known mandatory inputs to the model-facing projection without
// changing the execution schema used by host validation.  Every promoted
// field must already exist in the schema properties; a mismatch is an error
// rather than an implicit or silently invented model input.
std::string common_project_model_input_schema_required_parameters(
    const std::string & schema_json,
    const std::vector<std::string> & required_parameters,
    std::string & error);

std::string common_render_compact_tool_schema(
    const std::string & schema_json,
    std::string & error);

std::string common_render_compact_tool_description(
    const common_model_tool_contract & contract,
    std::string & error);

std::string common_render_compact_tool_description(
    const std::string & name,
    const std::string & description,
    const std::string & input_schema_json,
    const std::string & result_schema_json,
    std::string & error);
