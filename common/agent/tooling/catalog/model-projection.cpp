#include "agent/tooling/catalog/model-projection.h"
#include "agent/tooling/schema/tool-schema-compact.h"

#include <algorithm>
#include <map>

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

bool common_agent_project_dataset_model_schema(
        nlohmann::ordered_json & schema,
        const std::vector<common_agent_dataset_descriptor> & datasets,
        const nlohmann::ordered_json & bound_arguments,
        std::string & error) {
    using json = nlohmann::ordered_json;
    error.clear();
    if (!schema.is_object() || !bound_arguments.is_object() ||
            (schema.contains("properties") && !schema["properties"].is_object())) {
        error = "dataset model projection requires an object schema and object bindings";
        return false;
    }
    if (!schema.contains("properties")) return true;
    auto & properties = schema["properties"];
    std::map<std::string, std::vector<std::string>> choices;
    for (auto & property : properties.items()) {
        auto & field = property.value();
        if (!field.is_object()) continue;
        if (field.contains("x-agent-type") && !field["x-agent-type"].is_string()) {
            error = "x-agent-type must be a string";
            return false;
        }
        if (field.value("x-agent-type", std::string()) != "dataset_ref" ||
                bound_arguments.contains(property.key()) || datasets.empty() ||
                field.value("type", json()) != "string" || field.contains("enum") || field.contains("const")) continue;
        std::vector<std::string> handles;
        for (size_t index = 0; index < datasets.size(); ++index) {
            if (!datasets[index].ref.uri.empty()) handles.push_back("d" + std::to_string(index + 1));
        }
        if (handles.empty()) continue;
        choices[property.key()] = std::move(handles);
        if (field.contains("description") && !field["description"].is_string()) {
            error = "dataset-reference description must be a string";
            return false;
        }
        field["description"] = field.value("description", std::string()) +
            " Select an exact dN handle from the current host dataset inventory.";
    }
    for (auto & property : properties.items()) {
        auto & field = property.value();
        if (!field.is_object() || !field.contains("x-agent-column-of")) continue;
        if (!field["x-agent-column-of"].is_string()) {
            error = "x-agent-column-of must name a dataset-reference field";
            return false;
        }
        const auto source = field["x-agent-column-of"].get<std::string>();
        if (!properties.contains(source) || !properties[source].is_object() ||
                properties[source].value("x-agent-type", std::string()) != "dataset_ref") {
            error = "x-agent-column-of references an undeclared dataset field";
            return false;
        }
        json allowed = json::array();
        bool complete = true;
        size_t candidates = 0;
        const auto & source_schema = properties[source];
        for (size_t index = 0; index < datasets.size(); ++index) {
            const auto & dataset = datasets[index];
            if (dataset.ref.uri.empty()) continue;
            const std::string handle = "d" + std::to_string(index + 1);
            const auto matches = [&](const json & reference) {
                return reference.is_string() && (reference == dataset.ref.uri || reference == handle);
            };
            if (bound_arguments.contains(source)) {
                if (!matches(bound_arguments[source])) continue;
            } else if (source_schema.contains("const")) {
                if (!matches(source_schema["const"])) continue;
            } else if (source_schema.contains("enum")) {
                if (!source_schema["enum"].is_array()) { error = "dataset enum must be an array"; return false; }
                if (!std::any_of(source_schema["enum"].begin(), source_schema["enum"].end(), matches)) continue;
            }
            ++candidates;
            if (dataset.columns.empty() || dataset.ref.column_count != dataset.columns.size()) {
                complete = false;
                break;
            }
            for (const auto & column : dataset.columns) {
                if (column.name.empty()) { complete = false; break; }
                if (std::find(allowed.begin(), allowed.end(), json(column.name)) == allowed.end()) allowed.push_back(column.name);
            }
            if (!complete) break;
        }
        if (!complete || candidates == 0 || allowed.empty()) continue;
        json * values = &field;
        if (field.value("type", json()) == "array") {
            if (!field.contains("items") || !field["items"].is_object()) {
                error = "column list requires an item schema";
                return false;
            }
            values = &field["items"];
        }
        if (values->value("type", json()) != "string") { error = "column field must contain strings"; return false; }
        auto & columns = choices[property.key()];
        for (const auto & column : allowed) columns.push_back(column.get<std::string>());
    }
    if (!choices.empty()) {
        const auto projected = common_project_model_input_schema_enums(schema.dump(), choices, error);
        if (!error.empty()) return false;
        schema = json::parse(projected);
    }
    return true;
}

bool common_agent_validate_dataset_column_arguments(
        const nlohmann::ordered_json & schema,
        const nlohmann::ordered_json & arguments,
        const std::vector<common_agent_dataset_descriptor> & datasets,
        std::string & error) {
    using json = nlohmann::ordered_json;
    error.clear();
    if (!schema.is_object() || !arguments.is_object()) return true;
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) return true;
    for (const auto & property : properties.items()) {
        if (!property.value().is_object() || !property.value().contains("x-agent-column-of") ||
                !arguments.contains(property.key())) continue;
        const auto & source = property.value()["x-agent-column-of"];
        if (!source.is_string()) { error = "x-agent-column-of must be a string"; return false; }
        const auto reference = arguments.find(source.get<std::string>());
        if (reference == arguments.end() || !reference->is_string()) continue;
        const auto dataset = std::find_if(datasets.begin(), datasets.end(), [&](const auto & candidate) {
            return candidate.ref.uri == reference->get<std::string>();
        });
        if (dataset == datasets.end() || dataset->columns.empty() || dataset->ref.column_count != dataset->columns.size()) continue;
        const auto valid = [&](const json & value) {
            return value.is_string() && std::any_of(dataset->columns.begin(), dataset->columns.end(), [&](const auto & column) {
                return column.name == value.get<std::string>();
            });
        };
        const auto & value = arguments[property.key()];
        if (value.is_array() ? std::all_of(value.begin(), value.end(), valid) : valid(value)) continue;
        json allowed = json::array();
        for (const auto & column : dataset->columns) allowed.push_back(column.name);
        error = "dataset column argument " + json(property.key()).dump() +
            " must use a source column from dataset " + json(dataset->ref.name).dump() +
            "; allowed: " + allowed.dump();
        return false;
    }
    return true;
}
