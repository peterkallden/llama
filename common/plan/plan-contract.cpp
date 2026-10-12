#include "plan/plan-contract.h"

#include <nlohmann/json.hpp>

#include <set>

using json = nlohmann::ordered_json;

bool common_plan_extract_schema_fields(
        const std::string & schema_json,
        std::vector<common_plan_schema_field> & fields,
        std::string & error) {
    const auto schema = json::parse(schema_json, nullptr, false);
    if (!schema.is_object() || schema.value("type", std::string()) != "object") {
        error = "schema contract requires an object schema";
        fields.clear();
        return false;
    }
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) {
        error = "schema contract properties must be an object";
        fields.clear();
        return false;
    }
    std::set<std::string> required;
    for (const auto & value : schema.value("required", json::array())) {
        if (value.is_string()) required.insert(value.get<std::string>());
    }
    fields.clear();
    for (const auto & item : properties.items()) {
        const auto & property = item.value();
        if (!property.is_object()) continue;
        common_plan_schema_field field;
        field.name = item.key();
        field.required = required.count(field.name) != 0;
        field.semantic_type = property.value("x-agent-type", std::string());
        field.role = property.value("x-agent-role", std::string());
        field.inferable = property.value("x-agent-inferable", false);
        fields.push_back(std::move(field));
    }
    error.clear();
    return true;
}
