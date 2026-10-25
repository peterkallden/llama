#pragma once

#include <map>
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

// Render a concrete call with the same flat fields/directives as tool examples.
std::string common_render_flat_tool_call(
    const std::string & name,
    const std::string & input_schema_json,
    const std::string & arguments_json,
    std::string & error);

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

// Candidate-selection view: purpose only, including when a provider has
// already embedded a compact argument contract in its description.
std::string common_render_tool_choice_description(
    const std::string & name,
    const std::string & description,
    const std::string & input_schema_json,
    const std::string & result_schema_json,
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

// Narrow existing string fields or string-array items to host-authorized values.
// Existing enums/consts are intersected, never broadened. This projection does
// not replace the execution schema or authorize a reference.
std::string common_project_model_input_schema_enums(
    const std::string & schema_json,
    const std::map<std::string, std::vector<std::string>> & field_values,
    std::string & error);

// Remove inputs supplied by an explicit host binding from the model contract.
// The execution schema is deliberately unchanged. Unknown names are harmless
// because some model schemas already omit host-owned inputs.
std::string common_project_model_input_schema_without_bound_fields(
    const std::string & schema_json,
    const std::vector<std::string> & fields,
    std::string & error);
