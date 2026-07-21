// Bounded JSON object contract validation shared by native invokers.
#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using common_json_contract_value = nlohmann::ordered_json;

bool common_json_contract_parse_object(
    const std::string & input_json,
    common_json_contract_value & object,
    std::string & error);

bool common_json_contract_required_string(
    const common_json_contract_value & object,
    const char * key,
    size_t max_length,
    std::string & value,
    std::string & error);

bool common_json_contract_optional_string_array(
    const common_json_contract_value & object,
    const char * key,
    size_t max_items,
    size_t per_item_max_length,
    std::vector<std::string> & values,
    std::string & error);

bool common_json_contract_optional_unit_number(
    const common_json_contract_value & object,
    const char * key,
    float default_value,
    float & value,
    std::string & error);

// Parses an object-shaped external payload, emits its canonical JSON form and
// validates the supported JSON-schema subset (required/properties/types,
// enum, scalar bounds and array bounds). It never supplies semantic defaults.
bool common_schema_normalize_and_validate_object(
    const std::string & input_json,
    const std::string & schema_json,
    std::string & normalized_json,
    std::string & error);
