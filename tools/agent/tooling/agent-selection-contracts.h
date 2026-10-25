#pragma once

#include "plan/plan-json.h"
#include "agent/tooling/schema/tool-output-codec.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

struct agent_blueprint_selection_contract {
    std::string decision;
    std::string blueprint_id;
    float confidence = 0.0f;
};

struct agent_plan_selection_contract {
    std::string decision;
    std::string plan_id;
    float confidence = 0.0f;
};

struct agent_blueprint_binding_contract_entry {
    std::string step_id;
    std::string tool_name;
    common_plan_tool_arguments_contract arguments;
};

nlohmann::ordered_json make_agent_blueprint_selection_schema_json(
    const std::vector<std::string> & blueprint_ids);

std::string make_agent_blueprint_selection_schema_json_string(
    const std::vector<std::string> & blueprint_ids);

nlohmann::ordered_json make_agent_plan_selection_schema_json(
    const std::vector<std::string> & plan_ids);

std::string make_agent_plan_selection_schema_json_string(
    const std::vector<std::string> & plan_ids);

bool parse_agent_blueprint_selection_contract_json(
    const std::string & text,
    agent_blueprint_selection_contract & contract,
    std::string & error);

bool parse_agent_plan_selection_contract_json(
    const std::string & text,
    agent_plan_selection_contract & contract,
    std::string & error);

bool parse_agent_blueprint_binding_contract_json(
    const std::string & text,
    std::vector<agent_blueprint_binding_contract_entry> & bindings,
    std::string & error);

// Local control contracts share the flat tool codec without registering tools.
struct agent_selection_presentation {
    std::string instructions;
    std::string grammar;
    std::string json_schema;
};
bool make_agent_selection_presentation(common_agent_tool_output_format format,
    const std::string & name, const nlohmann::ordered_json & schema,
    agent_selection_presentation & out, std::string & error);
bool parse_agent_selection_response(common_agent_tool_output_format format,
    const std::string & name, const nlohmann::ordered_json & schema,
    const std::string & text, nlohmann::ordered_json & value, std::string & error);
