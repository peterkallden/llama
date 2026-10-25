#include "agent/tooling/schema/tool-schema-compact.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "plan/plan-contract.h"

#include <nlohmann/json.hpp>

#include <algorithm>
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
        const auto item_schema = property.value()["items"];
        const auto item_properties = item_schema.value("properties", json::object());
        for (const auto & slot : slots) {
            if (!slot.is_string() || !item_properties.contains(slot.get<std::string>())) continue;
            const auto & slot_schema = item_properties[slot.get<std::string>()];
            out << "\n    " << slot.get<std::string>() << ": " << scalar_type(slot_schema, 1);
            if (slot_schema.is_object() && slot_schema.contains("description") &&
                    slot_schema["description"].is_string()) {
                out << " (" << slot_schema["description"].get<std::string>() << ')';
            }
        }
        out << "\n  named form: " << directive << ":";
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].is_string()) out << " " << slots[i].get<std::string>() << "=<value>";
        }
    }
    return out.str();
}

std::string flat_field_type(const json & schema) {
    // Nested structures are described once in the indexed-field definitions.
    if (schema.is_object() && !schema.contains("x-agent-flat")) {
        const auto type = schema.value("type", std::string{});
        if (type == "object" || schema.contains("properties")) return "object";
        if (type == "array" && schema.contains("items")) {
            const auto & item = schema["items"];
            if (item.is_object() && (item.value("type", std::string{}) == "object" ||
                    item.contains("properties") || item.contains("oneOf") || item.contains("anyOf") || item.contains("allOf"))) {
                return "object[]";
            }
        }
    }
    return scalar_type(schema);
}

bool append_flat_schema_paths(
        const json & schema,
        const std::string & prefix,
        std::vector<std::string> & paths,
        std::string & error,
        size_t depth = 0) {
    if (!schema.is_object() || depth >= 32) {
        error = "invalid or excessively nested flat model contract";
        return false;
    }
    for (const char * key : {"oneOf", "anyOf", "allOf"}) {
        const auto branches = schema.find(key);
        if (branches == schema.end()) continue;
        if (!branches->is_array()) {
            error = "flat model contract alternatives must be arrays";
            return false;
        }
        size_t index = 0;
        for (const auto & branch : *branches) {
            paths.push_back(std::string(key) + " branch " + std::to_string(++index) +
                (prefix.empty() ? std::string{} : " at " + prefix) + ":");
            if (branch.is_object()) {
                const auto branch_properties = branch.value("properties", json::object());
                if (branch_properties.is_object()) for (const auto & field : branch_properties.items()) {
                    if (field.value().is_object() && (field.value().contains("const") || field.value().contains("enum"))) {
                        paths.push_back("condition " + (prefix.empty() ? field.key() : prefix + "." + field.key()) + ": " + flat_field_type(field.value()));
                    }
                }
                const auto branch_required = branch.value("required", json::array());
                if (!branch_required.is_array()) { error = "flat model contract required fields must be an array"; return false; }
                for (const auto & field : branch_required) {
                    if (!field.is_string()) { error = "flat model contract required field must be a string"; return false; }
                    paths.push_back("requires " + (prefix.empty() ? std::string{} : prefix + ".") + field.get<std::string>());
                }
            }
            if (!append_flat_schema_paths(branch, prefix, paths, error, depth + 1)) return false;
        }
    }
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) {
        error = "flat model contract properties must be an object";
        return false;
    }
    const auto required = schema.value("required", json::array());
    if (!required.is_array()) {
        error = "flat model contract required fields must be an array";
        return false;
    }
    // Declare a shared prefix once. Calls still use the existing indexed flat
    // paths; these are descriptions of their fields, not a new DSL syntax.
    std::string fields;
    for (const auto & property : properties.items()) {
        const auto & field = property.value();
        if (!field.is_object()) {
            error = "flat model contract field must be an object";
            return false;
        }
        if (field.contains("x-agent-flat")) continue;
        const bool mandatory = std::find(required.begin(), required.end(), json(property.key())) != required.end();
        const auto label = property.key() + (mandatory ? "!" : "?");
        if (!prefix.empty()) {
            if (!fields.empty()) fields += "; ";
            fields += label + ": " + flat_field_type(field);
            if (field.contains("description") && field["description"].is_string()) {
                fields += " (" + field["description"].get<std::string>() + ")";
            }
        }
    }
    if (!fields.empty()) paths.push_back(prefix + ".{ " + fields + " }");
    for (const auto & property : properties.items()) {
        const auto & field = property.value();
        if (field.contains("x-agent-flat")) continue;
        const std::string path = prefix.empty() ? property.key() : prefix + "." + property.key();
        const auto type = field.value("type", std::string{});
        if (type == "array") {
            const auto item = field.value("items", json::object());
            if (item.value("type", std::string{}) == "object" || item.contains("properties") ||
                    item.contains("oneOf") || item.contains("anyOf") || item.contains("allOf")) {
                if (!append_flat_schema_paths(item, path + "[0]", paths, error, depth + 1)) return false;
            }
        } else if (type == "object" || field.contains("properties") ||
                field.contains("oneOf") || field.contains("anyOf") || field.contains("allOf")) {
            if (!append_flat_schema_paths(field, path, paths, error, depth + 1)) return false;
        }
    }
    return true;
}

void flatten_example_fields(
        const json & schema,
        const json & value,
        const std::string & prefix,
        std::vector<std::string> & lines) {
    if (!schema.is_object() || !value.is_object()) return;
    // Pick example branches with the existing validator, rather than dropping
    // object-array fields whose properties live inside oneOf/anyOf.
    for (const char * key : {"oneOf", "anyOf", "allOf"}) {
        const auto branches = schema.find(key);
        if (branches == schema.end() || !branches->is_array()) continue;
        for (const auto & branch : *branches) {
            std::string normalized, validation_error;
            if (std::string(key) == "allOf" || common_schema_normalize_and_validate_object(
                    value.dump(), branch.dump(), normalized, validation_error)) {
                flatten_example_fields(branch, value, prefix, lines);
            }
        }
    }
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
                    lines.push_back(path + (prefix.empty() ? std::string{} : "[" + std::to_string(index) + "]") +
                        ": " + flat_scalar(field[index]));
                }
            }
        } else if (field.is_array()) {
            // Untyped JSON value slots can hold scalar lists, for example
            // membership predicate values. The schema-aware parser resolves
            // the entire path when admitting this bracketed representation.
            lines.push_back(path + ": " + flat_scalar(field));
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
    std::set<std::string> emitted;
    for (const auto & line : lines) if (emitted.insert(line).second) result += "\n" + line;
    return result;
}

std::string scalar_type(const json & schema, size_t depth) {
    if (!schema.is_object() || depth > 12) return "value";
    if (schema.contains("const")) return schema["const"].dump();
    if (schema.contains("required") && !schema.contains("properties") && !schema.contains("type")) {
        return "requires" + schema["required"].dump();
    }
    for (const auto * key : {"oneOf", "anyOf"}) {
        if (schema.contains("properties") || !schema.contains(key) || !schema[key].is_array()) continue;
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
    if (schema.contains("x-agent-type") && schema["x-agent-type"].is_string()) {
        return schema["x-agent-type"].get<std::string>();
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

std::string model_facing_type(std::string type) {
    constexpr const char * suffix = "[]";
    if (type.size() >= 2 && type.compare(type.size() - 2, 2, suffix) == 0) {
        type.resize(type.size() - 2);
        return type + " list";
    }
    return type;
}

std::string render_return_names(const std::vector<common_model_tool_field> & fields) {
    if (fields.empty()) return "result";
    std::ostringstream out;
    for (size_t index = 0; index < fields.size(); ++index) {
        if (index != 0) out << ", ";
        out << fields[index].name;
        if (!fields[index].description.empty()) {
            out << " (" << fields[index].description << ')';
        }
    }
    return out.str();
}

std::string render_input_fields(const std::vector<common_model_tool_field> & fields) {
    std::ostringstream out;
    for (const auto & field : fields) {
        out << "\n  " << field.name << " ("
            << (field.required && !field.may_be_inferred ? "required" : "optional")
            << ", " << model_facing_type(field.display_type) << ')';
        if (field.may_be_inferred) out << " [host may infer]";
        if (!field.description.empty()) out << ": " << field.description;
    }
    return out.str();
}

std::string compact_tool_purpose(
        const std::string & name,
        const std::string & description,
        const std::vector<common_model_tool_field> & outputs) {
    const std::string prefix = name + ": ";
    if (description.compare(0, prefix.size(), prefix) != 0) return description;

    const std::string marker = " Returns: " + render_return_names(outputs) + ".";
    const auto marker_pos = description.find(marker, prefix.size());
    if (marker_pos == std::string::npos) return description;
    const auto after_marker = marker_pos + marker.size();
    if (after_marker < description.size() && description[after_marker] != '\n') {
        return description;
    }

    // Tool providers already store a compact model contract in
    // common_chat_tool::description. Planner/output prompt builders also
    // render from the typed schemas; strip that known prefix before rendering
    // again, or the model receives duplicated purpose, returns and fields.
    return description.substr(prefix.size(), marker_pos - prefix.size());
}

} // namespace

std::string common_render_flat_tool_call(
        const std::string & name, const std::string & input_schema_json,
        const std::string & arguments_json, std::string & error) {
    error.clear();
    const auto schema = json::parse(input_schema_json, nullptr, false);
    const auto arguments = json::parse(arguments_json, nullptr, false);
    if (name.empty() || !schema.is_object() || !arguments.is_object()) {
        error = "flat tool call requires a tool name, object schema and object arguments";
        return {};
    }
    std::vector<std::string> lines;
    flatten_example_fields(schema, arguments, {}, lines);
    std::string rendered = "open! " + name;
    std::set<std::string> emitted;
    for (const auto & line : lines) if (emitted.insert(line).second) rendered += "\n" + line;
    return rendered;
}

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

std::string common_render_tool_choice_description(
        const std::string & name,
        const std::string & description,
        const std::string & input_schema_json,
        const std::string & result_schema_json,
        std::string & error) {
    const auto contract = common_project_model_tool_contract(
        name, description, input_schema_json, result_schema_json, error);
    if (!error.empty()) return {};
    std::string summary = name + ": " + compact_tool_purpose(name, description, contract.outputs);
    std::string required;
    for (const auto & field : contract.inputs) {
        if (!field.required || field.may_be_inferred) continue;
        if (!required.empty()) required += ", ";
        required += field.name;
    }
    if (!required.empty()) summary += " Required inputs: " + required + ".";
    const auto schema = json::parse(input_schema_json);
    const auto kinds = schema.value("x-agent-investigation-source-kinds", json::array());
    if (!kinds.is_array()) {
        error = "tool source requirements must be an array";
        return {};
    }
    if (!kinds.empty()) summary += " Requires selected source kind: " + kinds.dump() + ".";
    return summary;
}

std::string common_render_compact_tool_description(
        const common_model_tool_contract & contract,
        std::string & error) {
    error.clear();
    std::ostringstream out;
    out << contract.name << ": " << compact_tool_purpose(
            contract.name, contract.purpose, contract.outputs)
        << " Returns: " << render_return_names(contract.outputs) << '.';
    out << render_input_fields(contract.inputs);
    return out.str();
}

std::string common_render_compact_tool_description(
        const std::string & name,
        const std::string & description,
        const std::string & input_schema_json,
        const std::string & result_schema_json,
        std::string & error) {
    auto contract = common_project_model_tool_contract(
        name, description, input_schema_json, result_schema_json, error);
    if (!error.empty()) return {};
    const auto schema = json::parse(input_schema_json, nullptr, false);
    const auto properties = schema.value("properties", json::object());
    for (auto & field : contract.inputs) {
        if (properties.contains(field.name)) field.display_type = flat_field_type(properties[field.name]);
    }
    std::string rendered = common_render_compact_tool_description(contract, error);
    if (!error.empty()) return {};
    const auto directives = flat_directive_contract(schema);
    if (!directives.empty()) rendered += "\nflat DSL directives:" + directives;
    std::vector<std::string> paths;
    if (!append_flat_schema_paths(schema, {}, paths, error)) return {};
    if (!paths.empty()) {
        rendered += "\nFlat indexed fields (! required per object, ? optional; prefix.field: value; [0] is the first item):";
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

std::string common_project_model_input_schema_enums(
        const std::string & schema_json,
        const std::map<std::string, std::vector<std::string>> & field_values,
        std::string & error) {
    error.clear();
    auto schema = json::parse(schema_json, nullptr, false);
    if (!schema.is_object() || schema.value("type", json()) != "object" ||
            !schema.contains("properties") || !schema["properties"].is_object()) {
        error = "host enum projection requires an object schema with properties";
        return {};
    }
    for (const auto & field : field_values) {
        if (!schema["properties"].contains(field.first) || field.second.empty() || field.second.size() > 512) {
            error = "host enum projection has an unknown field or invalid choice count";
            return {};
        }
        auto & field_schema = schema["properties"][field.first];
        if (!field_schema.is_object()) {
            error = "host enum projection requires an object field schema";
            return {};
        }
        auto & property = field_schema.value("type", json()) == "array" && field_schema.contains("items")
            ? field_schema["items"] : field_schema;
        if (!property.is_object() || property.value("type", json()) != "string" ||
                (property.contains("enum") && !property["enum"].is_array())) {
            error = "host enum projection requires a string field with a valid enum";
            return {};
        }
        json allowed = json::array();
        std::set<std::string> seen;
        for (const auto & value : field.second) {
            if (!seen.insert(value).second) continue;
            if (property.contains("const") && property["const"] != value) continue;
            if (property.contains("enum") && std::find(property["enum"].begin(), property["enum"].end(), json(value)) == property["enum"].end()) continue;
            allowed.push_back(value);
        }
        if (allowed.empty()) {
            error = "host enum projection has no value allowed by the registered contract";
            return {};
        }
        property["enum"] = std::move(allowed);
    }
    return schema.dump();
}

std::string common_project_model_input_schema_without_bound_fields(
        const std::string & schema_json, const std::vector<std::string> & fields, std::string & error) {
    error.clear();
    auto schema = json::parse(schema_json, nullptr, false);
    if (!schema.is_object() || schema.value("type", json()) != "object") {
        error = "host-bound input projection requires an object schema";
        return {};
    }
    // Visit alternatives as well: a removed host input cannot remain required
    // in a branch of the projected model contract.
    const auto remove = [&](const auto & self, json & node, size_t depth) -> bool {
        if (!node.is_object() || depth > 32) {
            error = "invalid or excessively nested host-bound input schema";
            return false;
        }
        auto properties = node.find("properties");
        if (properties != node.end() && !properties->is_object()) {
            error = "model input properties must be an object";
            return false;
        }
        auto required = node.find("required");
        if (required != node.end() && !required->is_array()) {
            error = "model input required must be an array";
            return false;
        }
        for (const auto & field : fields) {
            if (properties != node.end()) properties->erase(field);
            if (required != node.end()) {
                required->erase(std::remove(required->begin(), required->end(), json(field)), required->end());
            }
        }
        for (const char * key : {"allOf", "anyOf", "oneOf"}) {
            auto branches = node.find(key);
            if (branches == node.end()) continue;
            if (!branches->is_array() || branches->empty()) {
                error = "model input alternatives must be nonempty arrays";
                return false;
            }
            for (auto & branch : *branches) if (!self(self, branch, depth + 1)) return false;
        }
        return true;
    };
    if (!remove(remove, schema, 0)) return {};
    return schema.dump();
}
