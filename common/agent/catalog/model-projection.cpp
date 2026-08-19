#include "agent/catalog/model-projection.h"

#include <nlohmann/json.hpp>

#include <set>

using json = nlohmann::ordered_json;

std::string common_tool_default_model_result_projection(const common_tool_definition & definition) {
    const auto result = json::parse(definition.result_schema_json, nullptr, false);
    if (!result.is_object() || result.value("type", std::string()) != "object") return {};
    const auto properties = result.value("properties", json::object());
    if (!properties.is_object()) return {};
    static const std::set<std::string> evidence_fields = {
        "rows", "columns", "values", "value", "count", "distinct_count",
        "null_count", "valid", "violations", "name", "path", "format"
    };
    json projection = json::object();
    projection["type"] = "object";
    projection["properties"] = json::object();
    std::set<std::string> retained;
    for (const auto & item : properties.items()) {
        const auto & property = item.value();
        if (!property.is_object()) continue;
        const auto role = property.value("x-agent-role", std::string());
        if (property.contains("x-agent-type") || role == "dataflow" ||
                role == "evidence" || evidence_fields.count(item.key()) != 0) {
            projection["properties"][item.key()] = property;
            retained.insert(item.key());
        }
    }
    if (retained.empty()) return {};
    projection["additionalProperties"] = false;
    json required = json::array();
    for (const auto & value : result.value("required", json::array())) {
        if (value.is_string() && retained.count(value.get<std::string>()) != 0) required.push_back(value);
    }
    if (!required.empty()) projection["required"] = std::move(required);
    return projection.dump();
}

std::string common_tool_default_model_input_projection(const common_tool_definition & definition) {
    const auto input = json::parse(definition.input_schema_json, nullptr, false);
    if (!input.is_object() || input.value("type", std::string()) != "object" ||
            input.contains("anyOf") || input.contains("oneOf")) return {};
    const auto properties = input.value("properties", json::object());
    if (!properties.is_object()) return {};
    static const std::set<std::string> host_control_fields = {
        "max_scan_rows", "max_result_rows", "materialize", "result_dataset",
        "backend", "execution_class", "timeout_ms"
    };
    json projection = json::object();
    projection["type"] = "object";
    projection["properties"] = json::object();
    std::set<std::string> retained;
    for (const auto & item : properties.items()) {
        const auto & property = item.value();
        if (!property.is_object() || host_control_fields.count(item.key()) != 0) continue;
        projection["properties"][item.key()] = property;
        retained.insert(item.key());
    }
    if (retained.empty()) return {};
    projection["additionalProperties"] = false;
    json required = json::array();
    for (const auto & value : input.value("required", json::array())) {
        if (value.is_string() && retained.count(value.get<std::string>()) != 0) required.push_back(value);
    }
    if (!required.empty()) projection["required"] = std::move(required);
    return projection.dump();
}
