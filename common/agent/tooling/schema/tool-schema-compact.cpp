#include "agent/tooling/schema/tool-schema-compact.h"
#include "plan/plan-contract.h"

#include <nlohmann/json.hpp>

#include <map>
#include <set>
#include <sstream>

using json = nlohmann::ordered_json;

namespace {

std::string scalar_type(const json & schema, size_t depth = 0) {
    if (schema.contains("x-agent-type") && schema["x-agent-type"].is_string()) {
        return schema["x-agent-type"].get<std::string>();
    }
    if (schema.contains("enum") && schema["enum"].is_array()) {
        std::ostringstream out;
        bool first = true;
        for (const auto & value : schema["enum"]) {
            if (!first) out << '|';
            first = false;
            out << (value.is_string() ? value.get<std::string>() : value.dump());
        }
        return out.str();
    }
    const auto type = schema.value("type", std::string("value"));
    if (type == "array") {
        return scalar_type(schema.value("items", json::object()), depth + 1) + "[]";
    }
    if (type == "integer" || type == "number") {
        std::string result = type;
        if (schema.contains("minimum") || schema.contains("maximum")) {
            result += '[';
            result += schema.contains("minimum") ? schema["minimum"].dump() : "";
            result += "..";
            result += schema.contains("maximum") ? schema["maximum"].dump() : "";
            result += ']';
        }
        return result;
    }
    if (type == "object") {
        const auto properties = schema.value("properties", json::object());
        if (depth >= 2 || !properties.is_object() || properties.empty()) return "object";
        std::set<std::string> required;
        for (const auto & value : schema.value("required", json::array())) {
            if (value.is_string()) required.insert(value.get<std::string>());
        }
        std::ostringstream nested;
        nested << '{';
        size_t count = 0;
        for (auto it = properties.begin(); it != properties.end() && count < 8; ++it, ++count) {
            if (count) nested << "; ";
            nested << it.key() << (required.count(it.key()) == 0 ? "?" : "")
                << ':' << scalar_type(it.value(), depth + 1);
        }
        if (properties.size() > count) nested << "; ...";
        nested << '}';
        return nested.str();
    }
    return type;
}

std::string render_object(const json & schema, std::string & error) {
    if (!schema.is_object() || schema.value("type", std::string()) != "object") {
        error = "compact tool schema requires a JSON object schema";
        return {};
    }

    std::set<std::string> required;
    if (schema.contains("required")) {
        if (!schema["required"].is_array()) {
            error = "tool schema required must be an array";
            return {};
        }
        for (const auto & value : schema["required"]) {
            if (!value.is_string()) {
                error = "tool schema required entries must be strings";
                return {};
            }
            required.insert(value.get<std::string>());
        }
    }

    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) {
        error = "tool schema properties must be an object";
        return {};
    }

    std::set<std::string> autowire_fields;
    for (const auto & value : schema.value("x-agent-autowire-fields", json::array())) {
        if (value.is_string()) autowire_fields.insert(value.get<std::string>());
    }

    std::ostringstream out;
    bool first = true;
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (!first) out << "; ";
        first = false;
        out << it.key() << (required.count(it.key()) == 0 ? "?" : "")
            << ':' << scalar_type(it.value());
        if (it.value().contains("default")) out << '=' << it.value()["default"].dump();
        if (autowire_fields.count(it.key())) out << " [may be inferred]";
    }
    return out.str();
}

std::vector<common_model_tool_field> project_fields(
        const json & schema, bool outputs, std::string & error) {
    std::vector<common_model_tool_field> fields;
    if (!schema.is_object() || schema.value("type", std::string()) != "object") {
        error = "model tool contract requires an object schema";
        return fields;
    }
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) {
        error = "tool schema properties must be an object";
        return fields;
    }
    std::vector<common_plan_schema_field> extracted;
    std::string extraction_error;
    if (!common_plan_extract_schema_fields(schema.dump(), extracted, extraction_error)) {
        error = extraction_error;
        return fields;
    }
    std::map<std::string, common_plan_schema_field> semantic_fields;
    for (auto & field : extracted) semantic_fields.emplace(field.name, std::move(field));
    std::set<std::string> inferable;
    for (const auto & value : schema.value("x-agent-autowire-fields", json::array())) {
        if (value.is_string()) inferable.insert(value.get<std::string>());
    }
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        common_model_tool_field field;
        field.name = it.key();
        const auto semantic = semantic_fields.find(field.name);
        if (semantic != semantic_fields.end()) {
            field.required = semantic->second.required;
            field.semantic_type = semantic->second.semantic_type;
            field.role = semantic->second.role;
            field.may_be_inferred = !outputs &&
                (inferable.count(field.name) != 0 || semantic->second.inferable);
        }
        field.display_type = scalar_type(it.value());
        fields.push_back(std::move(field));
    }
    return fields;
}

std::string render_fields(const std::vector<common_model_tool_field> & fields, bool inputs) {
    std::ostringstream out;
    bool first = true;
    for (const auto & field : fields) {
        if (!first) out << (inputs ? "; " : ", ");
        first = false;
        out << field.name << (inputs && !field.required ? "?" : "") << ':' << field.display_type;
        if (field.may_be_inferred) out << " [may be inferred]";
    }
    return out.str();
}

} // namespace

std::string common_render_compact_tool_schema(
        const std::string & schema_json,
        std::string & error) {
    error.clear();
    const auto schema = json::parse(schema_json, nullptr, false);
    if (schema.is_discarded()) {
        error = "tool schema is not valid JSON";
        return {};
    }
    return render_object(schema, error);
}

common_model_tool_contract common_project_model_tool_contract(
        const std::string & name,
        const std::string & description,
        const std::string & input_schema_json,
        const std::string & result_schema_json,
        std::string & error) {
    error.clear();
    const auto input = json::parse(input_schema_json, nullptr, false);
    const auto result = json::parse(result_schema_json, nullptr, false);
    common_model_tool_contract contract;
    contract.name = name;
    contract.purpose = description;
    if (input.is_discarded() || result.is_discarded()) {
        error = "tool schema is not valid JSON";
        return contract;
    }
    contract.inputs = project_fields(input, false, error);
    if (!error.empty()) return contract;
    contract.outputs = project_fields(result, true, error);
    return contract;
}

std::string common_render_compact_tool_description(
        const common_model_tool_contract & contract,
        std::string & error) {
    error.clear();
    std::ostringstream out;
    out << contract.name << "\n" << contract.purpose << "\n";
    const auto args = render_fields(contract.inputs, true);
    out << "args: " << (args.empty() ? "object" : args) << "\n";
    const auto returns = render_fields(contract.outputs, false);
    out << "returns: " << (returns.empty() ? "value" : returns);
    return out.str();
}

std::string common_render_compact_tool_description(
        const std::string & name,
        const std::string & description,
        const std::string & input_schema_json,
        const std::string & result_schema_json,
        std::string & error) {
    const auto contract = common_project_model_tool_contract(
        name, description, input_schema_json, result_schema_json, error);
    if (!error.empty()) return {};
    std::string rendered = common_render_compact_tool_description(contract, error);
    if (!error.empty()) return {};
    if (name == "data.join") {
        rendered += "\nexample: args:{left:$orders.dataset; right:$customers.dataset; on:[{left:customer_id; right:customer_id}]}";
    } else if (name == "data.aggregate") {
        rendered += "\nexample: args:{dataset:$joined.dataset; measures:[{function:sum; column:amount}]}";
    } else if (name == "dataset.select") {
        rendered += "\nexample: args:{name:orders} -> dataset:$orders.dataset";
    } else if (name == "dataset.list") {
        rendered += "\nexample: args:{} -> datasets:$datasets.datasets[]";
    } else if (name == "statistics.describe") {
        rendered += "\nexample: args:{dataset:$joined.dataset; columns:[amount]}";
    }
    return rendered;
}
