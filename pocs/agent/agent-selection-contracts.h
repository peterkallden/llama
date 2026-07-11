#pragma once

#include "plan/plan-json.h"

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
