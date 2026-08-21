#include "agent/tooling/contracts/schema-contract.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

bool validate_value(const json & value, const json & schema, const std::string & field, std::string & error) {
    const auto type = schema.value("type", std::string());
    const bool type_ok = type.empty() ||
        (type == "string" && value.is_string()) || (type == "integer" && value.is_number_integer()) ||
        (type == "number" && value.is_number()) || (type == "boolean" && value.is_boolean()) ||
        (type == "array" && value.is_array()) || (type == "object" && value.is_object());
    if (!type_ok) { error = "contract field '" + field + "' has invalid type"; return false; }
    if (schema.contains("enum")) {
        bool found = false;
        for (const auto & allowed : schema["enum"]) if (allowed == value) { found = true; break; }
        if (!found) { error = "contract field '" + field + "' is not an allowed value"; return false; }
    }
    if (value.is_string()) {
        const auto size = value.get_ref<const std::string &>().size();
        if (schema.contains("minLength") && size < schema["minLength"].get<size_t>()) { error = "contract field '" + field + "' is too short"; return false; }
        if (schema.contains("maxLength") && size > schema["maxLength"].get<size_t>()) { error = "contract field '" + field + "' is too long"; return false; }
    }
    if (value.is_number()) {
        const auto number = value.get<double>();
        if (schema.contains("minimum") && number < schema["minimum"].get<double>()) { error = "contract field '" + field + "' is below its minimum"; return false; }
        if (schema.contains("maximum") && number > schema["maximum"].get<double>()) { error = "contract field '" + field + "' exceeds its maximum"; return false; }
    }
    if (value.is_array()) {
        if (schema.contains("minItems") && value.size() < schema["minItems"].get<size_t>()) { error = "contract field '" + field + "' has too few items"; return false; }
        if (schema.contains("maxItems") && value.size() > schema["maxItems"].get<size_t>()) { error = "contract field '" + field + "' has too many items"; return false; }
        if (schema.contains("items")) for (size_t i = 0; i < value.size(); ++i) if (!validate_value(value[i], schema["items"], field + "[" + std::to_string(i) + "]", error)) return false;
    }
    return true;
}

} // namespace

bool common_json_contract_parse_object(const std::string & input_json, common_json_contract_value & object, std::string & error) {
    object = json::parse(input_json, nullptr, false);
    if (!object.is_object()) { error = "contract input must be a JSON object"; return false; }
    error.clear();
    return true;
}

bool common_json_contract_required_string(const common_json_contract_value & object, const char * key, size_t max_length,
        std::string & value, std::string & error) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_string()) { error = std::string("contract field '") + key + "' must be a string"; return false; }
    value = it->get<std::string>();
    if (value.empty() || value.size() > max_length) { error = std::string("contract field '") + key + "' is out of bounds"; return false; }
    return true;
}

bool common_json_contract_optional_string_array(const common_json_contract_value & object, const char * key,
        size_t max_items, size_t per_item_max_length, std::vector<std::string> & values, std::string & error) {
    values.clear();
    const auto it = object.find(key);
    if (it == object.end()) return true;
    if (!it->is_array() || it->size() > max_items) { error = std::string("contract field '") + key + "' must be a bounded string array"; return false; }
    for (const auto & item : *it) {
        if (!item.is_string() || item.get_ref<const std::string &>().size() > per_item_max_length) { error = std::string("contract field '") + key + "' contains an invalid string"; return false; }
        values.push_back(item.get<std::string>());
    }
    return true;
}

bool common_json_contract_optional_unit_number(const common_json_contract_value & object, const char * key,
        float default_value, float & value, std::string & error) {
    value = default_value;
    const auto it = object.find(key);
    if (it == object.end()) return true;
    if (!it->is_number()) { error = std::string("contract field '") + key + "' must be a number"; return false; }
    value = it->get<float>();
    if (value < 0.0f || value > 1.0f) { error = std::string("contract field '") + key + "' must be between zero and one"; return false; }
    return true;
}

bool common_schema_normalize_and_validate_object(const std::string & input_json, const std::string & schema_json,
        std::string & normalized_json, std::string & error) {
    const auto input = json::parse(input_json, nullptr, false);
    const auto schema = json::parse(schema_json, nullptr, false);
    if (!input.is_object()) { error = "contract input must be a JSON object"; return false; }
    if (!schema.is_object() || schema.value("type", std::string()) != "object") { error = "contract schema must describe an object"; return false; }
    const auto required = schema.value("required", json::array());
    const auto properties = schema.value("properties", json::object());
    for (const auto & key : required) if (!key.is_string() || !input.contains(key.get<std::string>())) { error = "required contract field is missing"; return false; }
    if (schema.value("additionalProperties", true) == false) for (auto field = input.begin(); field != input.end(); ++field) if (!properties.contains(field.key())) { error = "unexpected contract field: " + field.key(); return false; }
    for (auto field = input.begin(); field != input.end(); ++field) if (properties.contains(field.key()) && !validate_value(field.value(), properties[field.key()], field.key(), error)) return false;
    normalized_json = input.dump();
    error.clear();
    return true;
}
