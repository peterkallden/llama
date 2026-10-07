#include "agent/tooling/schema/tool-schema-compact.h"
#include "plan/plan-contract.h"

#include <nlohmann/json.hpp>

#include <map>
#include <set>
#include <sstream>

using json = nlohmann::ordered_json;

namespace {

std::string flat_scalar(const json & value) {
    if (!value.is_string()) return value.dump();
    const auto text = value.get<std::string>();
    if (text.rfind("$", 0) == 0 && text.find_first_of(" \t\r\n") == std::string::npos) return text;
    if (!text.empty() && text.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-") == std::string::npos &&
            text != "true" && text != "false" && text != "null") {
        const auto number = json::parse(text, nullptr, false);
        if (number.is_discarded() || !number.is_number()) return text;
    }
    return value.dump();
}

std::string scalar_type(const json & schema, size_t depth = 0);

std::string flat_directive_contract(const json & schema) {
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) return {};
    std::ostringstream out;
    for (const auto & property : properties.items()) {
        if (!property.value().is_object() || !property.value().contains("x-agent-flat") ||
                !property.value()["x-agent-flat"].is_object()) continue;
        const auto & annotation = property.value()["x-agent-flat"];
        const auto directive = annotation.value("directive", std::string{});
        const auto slots = annotation.value("positional", json::array());
        if (directive.empty() || !slots.is_array() || slots.empty()) continue;
        out << "\n  " << directive << ":";
        for (const auto & slot : slots) {
            if (slot.is_string()) out << " <" << slot.get<std::string>() << ">";
        }
        out << " (repeat for each item; named slots also accepted)";
        out << "\n  named form: " << directive << ":";
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].is_string()) out << " " << slots[i].get<std::string>() << "=<value>";
        }
    }
    return out.str();
}

void append_flat_schema_paths(
        const json & schema,
        const std::string & prefix,
        std::vector<std::string> & paths,
        size_t depth = 0) {
    if (depth >= 8 || !schema.is_object()) return;
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) return;
    for (const auto & property : properties.items()) {
        if (paths.size() >= 32) return;
        const std::string path = prefix.empty() ? property.key() : prefix + "." + property.key();
        const auto & field = property.value();
        if (field.is_object() && field.value("type", std::string{}) == "array") {
            if (field.contains("x-agent-flat")) continue;
            const auto item = field.value("items", json::object());
            if (item.value("type", std::string{}) == "object") {
                append_flat_schema_paths(item, path + "[0]", paths, depth + 1);
            } else {
                paths.push_back(path + "[0]: " + scalar_type(item));
            }
        } else if (field.is_object() && field.value("type", std::string{}) == "object") {
            append_flat_schema_paths(field, path, paths, depth + 1);
        } else if (!prefix.empty()) {
            std::string description = path + ": " + scalar_type(field);
            if (field.is_object() && field.contains("description") && field["description"].is_string()) {
                description += " (" + field["description"].get<std::string>() + ")";
            }
            paths.push_back(std::move(description));
        }
    }
}

void flatten_example_fields(
        const json & schema,
        const json & value,
        const std::string & prefix,
        std::vector<std::string> & lines) {
    if (!schema.is_object() || !value.is_object()) return;
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) return;
    for (const auto & entry : value.items()) {
        const auto & field = entry.value();
        if (!properties.contains(entry.key())) continue;
        const auto & field_schema = properties[entry.key()];
        const std::string path = prefix.empty() ? entry.key() : prefix + "." + entry.key();
        if (field_schema.is_object() && field_schema.contains("x-agent-flat") &&
                field_schema["x-agent-flat"].is_object() && field.is_array()) {
            const auto slots = field_schema["x-agent-flat"].value("positional", json::array());
            const auto directive = field_schema["x-agent-flat"].value("directive", entry.key());
            for (const auto & item : field) {
                if (!item.is_object()) continue;
                bool positional = slots.is_array() && !slots.empty();
                std::vector<std::string> values;
                for (const auto & slot : slots) {
                    if (!slot.is_string() || !item.contains(slot.get<std::string>())) {
                        positional = false;
                        break;
                    }
                    values.push_back(flat_scalar(item[slot.get<std::string>()]));
                }
                std::string line = directive + ":";
                if (positional) {
                    for (const auto & slot_value : values) line += " " + slot_value;
                } else {
                    for (const auto & entry : item.items()) {
                        line += " " + entry.key() + "=" + flat_scalar(entry.value());
                    }
                }
                lines.push_back(std::move(line));
            }
        } else if (field.is_object()) {
            flatten_example_fields(field_schema, field, path, lines);
        } else if (field.is_array() && field_schema.value("type", std::string{}) == "array") {
            for (size_t index = 0; index < field.size(); ++index) {
                if (field[index].is_object()) {
                    flatten_example_fields(field_schema.value("items", json::object()), field[index],
                        path + "[" + std::to_string(index) + "]", lines);
                } else {
                    lines.push_back(path + ": " + flat_scalar(field[index]));
                }
            }
        } else {
            lines.push_back(path + ": " + flat_scalar(field));
        }
    }
}

std::string flat_example(const std::string & tool, const json & schema, const json & value) {
    std::vector<std::string> lines;
    flatten_example_fields(schema, value, {}, lines);
    if (lines.empty()) return {};
    std::string result = "\nexample:\nopen! " + tool;
    for (const auto & line : lines) result += "\n" + line;
    return result;
}

std::string scalar_type(const json & schema, size_t depth) {
    for (const auto * key : {"oneOf", "anyOf"}) {
        if (!schema.contains(key) || !schema[key].is_array()) continue;
        std::ostringstream alternatives;
        alternatives << key << '(';
        bool first = true;
        for (const auto & alternative : schema[key]) {
            if (!first) alternatives << "|";
            first = false;
            alternatives << scalar_type(alternative, depth + 1);
        }
        alternatives << ')';
        return alternatives.str();
    }
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
        if (schema.contains("x-agent-flat") && schema["x-agent-flat"].is_object()) {
            return "directive(" + schema["x-agent-flat"].value("directive", std::string("item")) + ")[]";
        }
        if (schema.contains("items") && schema["items"].is_object() &&
                schema["items"].value("type", std::string{}) == "object") {
            return "object[]";
        }
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
        if (depth >= 4 || !properties.is_object() || properties.empty()) return "object";
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
            if (it.value().is_object() && it.value().contains("description") &&
                    it.value()["description"].is_string()) {
                nested << " (" << it.value()["description"].get<std::string>() << ')';
            }
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
    if (!schema.is_object()) {
        error = "model tool contract schema must be a JSON schema object";
        return fields;
    }
    // Tool inputs remain object-shaped, but OpenAPI responses may legitimately
    // be top-level arrays or scalars (for example a collection operation).  A
    // model-facing compact contract needs one stable field for those values;
    // the host-facing result schema and actual payload stay unchanged.
    if (schema.value("type", std::string()) != "object") {
        common_model_tool_field field;
        field.name = "value";
        field.description = schema.value("description", std::string());
        field.display_type = scalar_type(schema);
        fields.push_back(std::move(field));
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
        if (it.value().is_object()) {
            field.description = it.value().value("description", std::string());
        }
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
        if (!field.description.empty()) out << " (" << field.description << ')';
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

std::string common_project_model_input_schema_required_parameters(
        const std::string & schema_json,
        const std::vector<std::string> & required_parameters,
        std::string & error) {
    error.clear();
    auto schema = json::parse(schema_json, nullptr, false);
    if (schema.is_discarded() || !schema.is_object() ||
            schema.value("type", std::string()) != "object") {
        error = "model input schema must be a JSON object schema";
        return {};
    }
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) {
        error = "model input schema properties must be an object";
        return {};
    }

    std::set<std::string> required;
    if (schema.contains("required")) {
        if (!schema["required"].is_array()) {
            error = "model input schema required must be an array";
            return {};
        }
        for (const auto & value : schema["required"]) {
            if (!value.is_string()) {
                error = "model input schema required entries must be strings";
                return {};
            }
            required.insert(value.get<std::string>());
        }
    }
    for (const auto & parameter : required_parameters) {
        if (parameter.empty()) {
            error = "model input schema required parameter must not be empty";
            return {};
        }
        if (!properties.contains(parameter)) {
            error = "host-required parameter is absent from model input schema: " + parameter;
            return {};
        }
        required.insert(parameter);
    }

    schema["required"] = json::array();
    for (const auto & parameter : required) {
        schema["required"].push_back(parameter);
    }
    return schema.dump();
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
    const auto schema = json::parse(input_schema_json, nullptr, false);
    const auto directives = flat_directive_contract(schema);
    if (!directives.empty()) rendered += "\nflat DSL directives:" + directives;
    std::vector<std::string> paths;
    append_flat_schema_paths(schema, {}, paths);
    if (!paths.empty()) {
        rendered += "\nflat field paths:";
        for (const auto & path : paths) rendered += "\n  " + path;
    }
    const auto rules = schema.is_object()
        ? schema.value("x-agent-rules", json::array())
        : json::array();
    if (rules.is_array()) {
        for (const auto & rule : rules) {
            if (rule.is_string()) rendered += "\nrule: " + rule.get<std::string>();
        }
    }
    const auto examples = schema.is_object()
        ? schema.value("x-agent-examples", json::array())
        : json::array();
    if (examples.is_array() && !examples.empty()) {
        size_t count = 0;
        for (const auto & example : examples) {
            if (!example.is_object() || ++count > 3) continue;
            const std::string formatted = flat_example(name, schema, example);
            if (formatted.size() <= 1024) rendered += formatted;
        }
    } else if (name == "data.join") {
        rendered += flat_example(name, schema, json{
            {"left", "$orders.dataset"}, {"right", "$customers.dataset"},
            {"on", json::array({json{{"left", "customer_id"}, {"right", "customer_id"}}})}});
    } else if (name == "data.aggregate") {
        rendered += flat_example(name, schema, json{
            {"dataset", "$joined.dataset"}, {"measures", json::array({json{{"function", "sum"}, {"column", "amount"}}})}});
    } else if (name == "dataset.select") {
        rendered += flat_example(name, schema, json{{"name", "orders"}});
    } else if (name == "dataset.list") {
        rendered += flat_example(name, schema, json::object());
    } else if (name == "statistics.describe") {
        rendered += flat_example(name, schema, json{{"dataset", "$joined.dataset"}, {"columns", json::array({"amount"})}});
    }
    return rendered;
}
