#include "agent/tooling/schema/tool-output-codec.h"

#include "agent/tooling/schema/tool-schema-compact.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <variant>

using json = nlohmann::ordered_json;

namespace {

std::string trim(const std::string & value) {
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) ++first;
    size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) --last;
    return value.substr(first, last - first);
}

bool identifier_char(char value) {
    return std::isalnum(static_cast<unsigned char>(value)) ||
        value == '_' || value == '.' || value == '-';
}

bool parse_identifier(const std::string & text, size_t & position, std::string & value) {
    const size_t start = position;
    while (position < text.size() && identifier_char(text[position])) ++position;
    if (position == start) return false;
    value = text.substr(start, position - start);
    return true;
}

bool parse_quoted(const std::string & text, size_t & position, json & value, std::string & error) {
    if (position >= text.size() || text[position] != '"') {
        error = "compact DSL string must start with a double quote";
        return false;
    }
    const size_t start = position++;
    bool escaped = false;
    while (position < text.size()) {
        const char character = text[position++];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (character == '\\') {
            escaped = true;
        } else if (character == '"') {
            const auto parsed = json::parse(text.substr(start, position - start), nullptr, false);
            if (parsed.is_discarded() || !parsed.is_string()) {
                error = "compact DSL string is not valid JSON string syntax";
                return false;
            }
            value = parsed;
            return true;
        }
    }
    error = "compact DSL string is unterminated";
    return false;
}

bool parse_value(const std::string & text, size_t & position, json & value, std::string & error, size_t depth = 0);

bool parse_value_object(const std::string & text, size_t & position, json & value, std::string & error, size_t depth) {
    if (depth >= 12) {
        error = "compact DSL value nesting exceeds the supported depth";
        return false;
    }
    ++position; // {
    value = json::object();
    while (position < text.size()) {
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        if (position < text.size() && text[position] == '}') {
            ++position;
            return true;
        }
        std::string key;
        if (!parse_identifier(text, position, key)) {
            error = "compact DSL object key is invalid";
            return false;
        }
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        if (position >= text.size() || text[position] != '=') {
            error = "compact DSL object field must be key=value";
            return false;
        }
        ++position;
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        json field;
        if (!parse_value(text, position, field, error, depth + 1)) return false;
        if (value.contains(key)) {
            error = "compact DSL object repeats field: " + key;
            return false;
        }
        value[key] = std::move(field);
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        if (position < text.size() && text[position] == ';') {
            ++position;
            continue;
        }
        if (position < text.size() && text[position] == '}') {
            ++position;
            return true;
        }
        error = "compact DSL object fields must be separated by semicolons";
        return false;
    }
    error = "compact DSL object is unterminated";
    return false;
}

bool parse_value_array(const std::string & text, size_t & position, json & value, std::string & error, size_t depth) {
    if (depth >= 12) {
        error = "compact DSL value nesting exceeds the supported depth";
        return false;
    }
    ++position; // [
    value = json::array();
    while (position < text.size()) {
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        if (position < text.size() && text[position] == ']') {
            ++position;
            return true;
        }
        json item;
        if (!parse_value(text, position, item, error, depth + 1)) return false;
        value.push_back(std::move(item));
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        if (position < text.size() && text[position] == ',') {
            ++position;
            continue;
        }
        if (position < text.size() && text[position] == ']') {
            ++position;
            return true;
        }
        error = "compact DSL array values must be comma-separated";
        return false;
    }
    error = "compact DSL array is unterminated";
    return false;
}

bool parse_value(const std::string & text, size_t & position, json & value, std::string & error, size_t depth) {
    if (position < text.size() && text[position] == '"') {
        return parse_quoted(text, position, value, error);
    }
    if (position < text.size() && text[position] == '{') {
        return parse_value_object(text, position, value, error, depth);
    }
    if (position < text.size() && text[position] == '[') {
        return parse_value_array(text, position, value, error, depth);
    }
    const size_t start = position;
    while (position < text.size() && !std::isspace(static_cast<unsigned char>(text[position])) &&
            text[position] != ',' && text[position] != ')' && text[position] != ']' &&
            text[position] != '}' && text[position] != ';') {
        ++position;
    }
    if (position == start) {
        error = "compact DSL argument value is missing";
        return false;
    }
    const std::string token = text.substr(start, position - start);
    if (token == "true" || token == "false") {
        value = token == "true";
        return true;
    }
    if (token == "null") {
        value = nullptr;
        return true;
    }
    const auto number = json::parse(token, nullptr, false);
    if (!number.is_discarded() && number.is_number()) {
        value = number;
        return true;
    }
    if (token.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-$") != std::string::npos) {
        error = "compact DSL value must be quoted, boolean, number or identifier";
        return false;
    }
    value = token;
    return true;
}

bool parse_arguments(
        const std::string & text,
        size_t & position,
        bool call_style,
        json & arguments,
        std::string & error) {
    arguments = json::object();
    while (position < text.size()) {
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        if (call_style && position < text.size() && text[position] == ',') {
            ++position;
            while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        }
        if (position >= text.size() || (call_style && text[position] == ')')) break;
        std::string key;
        if (!parse_identifier(text, position, key)) {
            error = "compact DSL argument name is invalid";
            return false;
        }
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        if (position >= text.size() || text[position] != '=') {
            error = "compact DSL argument must be key=value";
            return false;
        }
        ++position;
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        json value;
        if (!parse_value(text, position, value, error)) return false;
        if (arguments.contains(key)) {
            error = "compact DSL repeats argument: " + key;
            return false;
        }
        arguments[key] = std::move(value);
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
        if (!call_style) continue;
        if (position < text.size() && text[position] != ',' && text[position] != ')') {
            error = "compact DSL call arguments must be comma-separated";
            return false;
        }
    }
    return true;
}

bool parse_compact(const std::string & input, common_agent_tool_call & call, std::string & error) {
    const std::string text = trim(input);
    if (text.empty() || text.find('\n') != std::string::npos || text.find('\r') != std::string::npos) {
        error = "compact DSL must be exactly one non-empty line";
        return false;
    }

    size_t position = 0;
    bool call_style = false;
    if (text.rfind("open!", 0) == 0) {
        position = 5;
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
    } else {
        call_style = true;
    }
    if (!parse_identifier(text, position, call.name)) {
        error = "compact DSL has no valid tool name";
        return false;
    }
    while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
    if (call_style) {
        if (position >= text.size() || text[position] != '(') {
            error = "compact DSL call syntax requires parentheses";
            return false;
        }
        ++position;
    }
    json arguments;
    if (!parse_arguments(text, position, call_style, arguments, error)) return false;
    if (call_style) {
        if (position >= text.size() || text[position] != ')') {
            error = "compact DSL call syntax is missing closing parenthesis";
            return false;
        }
        ++position;
    }
    while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
    if (position != text.size()) {
        error = "compact DSL has trailing content after the tool call";
        return false;
    }
    call.arguments_json = arguments.dump();
    return true;
}

bool split_flat_path(const std::string & path, std::vector<std::variant<std::string, size_t>> & parts) {
    parts.clear();
    size_t position = 0;
    while (position < path.size()) {
        const size_t key_start = position;
        while (position < path.size() &&
                (std::isalnum(static_cast<unsigned char>(path[position])) ||
                 path[position] == '_' || path[position] == '-')) ++position;
        if (position == key_start) return false;
        std::string key = path.substr(key_start, position - key_start);
        parts.emplace_back(std::move(key));
        while (position < path.size() && path[position] == '[') {
            ++position;
            const size_t start = position;
            while (position < path.size() && std::isdigit(static_cast<unsigned char>(path[position]))) ++position;
            if (start == position || position >= path.size() || path[position++] != ']') return false;
            size_t index = 0;
            try {
                index = static_cast<size_t>(std::stoull(path.substr(start, position - start - 1)));
            } catch (...) {
                return false;
            }
            parts.emplace_back(index);
        }
        if (position == path.size()) break;
        if (path[position++] != '.' || position == path.size()) return false;
    }
    return !parts.empty();
}

bool assign_flat_path(
        json & target,
        const std::vector<std::variant<std::string, size_t>> & parts,
        size_t depth,
        const json & value,
        std::string & error) {
    if (depth >= parts.size()) {
        error = "flat DSL path has no destination";
        return false;
    }
    if (const auto * key = std::get_if<std::string>(&parts[depth])) {
        if (!target.is_object()) {
            error = "flat DSL path expected an object before field " + *key;
            return false;
        }
        if (depth + 1 == parts.size()) {
            if (target.contains(*key)) {
                error = "flat DSL repeats field path: " + *key;
                return false;
            }
            target[*key] = value;
            return true;
        }
        const bool next_is_index = std::holds_alternative<size_t>(parts[depth + 1]);
        if (!target.contains(*key)) target[*key] = next_is_index ? json::array() : json::object();
        return assign_flat_path(target[*key], parts, depth + 1, value, error);
    }
    const size_t index = std::get<size_t>(parts[depth]);
    if (!target.is_array() || index > target.size()) {
        error = "flat DSL array indexes must be contiguous and match an array field";
        return false;
    }
    if (index == target.size()) {
        const bool next_is_index = depth + 1 < parts.size() &&
            std::holds_alternative<size_t>(parts[depth + 1]);
        target.push_back(depth + 1 == parts.size()
            ? value
            : (next_is_index ? json::array() : json::object()));
        if (depth + 1 == parts.size()) return true;
    }
    return assign_flat_path(target[index], parts, depth + 1, value, error);
}

bool parse_flat_value(const std::string & source, json & value, std::string & error) {
    const std::string text = trim(source);
    if (text.empty()) {
        error = "flat DSL field value is empty; omit an unset field instead";
        return false;
    }
    if (text.front() == '{') {
        error = "flat DSL does not accept nested objects; use field paths or declared directives";
        return false;
    }
    size_t position = 0;
    if (!parse_value(text, position, value, error)) return false;
    while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
    if (position != text.size()) {
        error = "flat DSL scalar values containing whitespace must be quoted";
        return false;
    }
    return true;
}

bool parse_flat_scalar_array(const std::string & source, json & value, std::string & error) {
    value = json::array();
    size_t start = 0;
    bool quoted = false;
    bool escaped = false;
    for (size_t i = 0; i <= source.size(); ++i) {
        const char ch = i < source.size() ? source[i] : ',';
        if (i < source.size() && quoted) {
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == '"') quoted = false;
            continue;
        }
        if (i < source.size() && ch == '"') {
            quoted = true;
            continue;
        }
        if (ch != ',') continue;
        const std::string item_text = trim(source.substr(start, i - start));
        if (item_text.empty()) {
            error = "flat DSL array values must be non-empty";
            return false;
        }
        json item;
        if (!parse_flat_value(item_text, item, error)) return false;
        if (item.is_array() || item.is_object()) {
            error = "flat DSL comma-separated lists accept scalar values only";
            return false;
        }
        value.push_back(std::move(item));
        start = i + 1;
    }
    return true;
}

const json * find_flat_directive(const json & schema, const std::string & name) {
    if (!schema.is_object() || !schema.contains("properties")) return nullptr;
    const auto & properties = schema["properties"];
    if (!properties.is_object()) return nullptr;
    for (const auto & property : properties.items()) {
        if (!property.value().is_object() || property.value().value("type", std::string{}) != "array" ||
                !property.value().contains("x-agent-flat") || !property.value()["x-agent-flat"].is_object()) continue;
        if (property.value()["x-agent-flat"].value("directive", std::string{}) == name) return &property.value();
    }
    return nullptr;
}

std::string flat_directive_property_name(const json & schema, const std::string & name) {
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) return {};
    for (const auto & property : properties.items()) {
        if (property.value().is_object() && property.value().contains("x-agent-flat") &&
                property.value()["x-agent-flat"].is_object() &&
                property.value()["x-agent-flat"].value("directive", std::string{}) == name) {
            return property.key();
        }
    }
    return {};
}

bool parse_flat_directive_value(
        const std::string & source,
        const json & property_schema,
        json & object,
        std::string & error) {
    const auto & annotation = property_schema["x-agent-flat"];
    const auto & item_schema = property_schema["items"];
    const auto item_properties = item_schema.value("properties", json::object());
    if (!item_schema.is_object() || item_schema.value("type", std::string{}) != "object" ||
            !item_properties.is_object()) {
        error = "flat DSL directive annotation requires an object array item schema";
        return false;
    }

    object = json::object();
    const std::string value = trim(source);
    const size_t first_equals = value.find('=');
    const size_t first_space = value.find_first_of(" \t\r\n");
    const bool named = first_equals != std::string::npos &&
        (first_space == std::string::npos || first_equals < first_space) &&
        first_equals > 0 && value.substr(0, first_equals).find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") == std::string::npos;
    if (named) {
        size_t position = 0;
        if (!parse_arguments(value, position, false, object, error)) return false;
        while (position < value.size() && std::isspace(static_cast<unsigned char>(value[position]))) ++position;
        if (position != value.size()) {
            error = "flat DSL named directive has trailing content";
            return false;
        }
    } else {
        const auto slots = annotation.value("positional", json::array());
        if (!slots.is_array() || slots.empty()) {
            error = "flat DSL directive has no positional slot declaration; use named slots";
            return false;
        }
        size_t position = 0;
        size_t slot_index = 0;
        while (position < value.size()) {
            while (position < value.size() && std::isspace(static_cast<unsigned char>(value[position]))) ++position;
            if (position >= value.size()) break;
            if (slot_index >= slots.size() || !slots[slot_index].is_string()) {
                error = "flat DSL directive contains more values than declared positional slots";
                return false;
            }
            const auto field = slots[slot_index++].get<std::string>();
            json slot_value;
            if (!parse_value(value, position, slot_value, error)) return false;
            if (slot_value.is_object() || slot_value.is_array()) {
                error = "flat DSL directive slots must be scalar values";
                return false;
            }
            object[field] = std::move(slot_value);
        }
    }
    for (const auto & item : object.items()) {
        if (!item_properties.contains(item.key())) {
            error = "flat DSL directive contains an unknown slot: " + item.key();
            return false;
        }
        if (item.value().is_object() || item.value().is_array()) {
            error = "flat DSL directive slots must be scalar values";
            return false;
        }
    }
    if (!object.is_object() || object.empty()) {
        error = "flat DSL directive must provide at least one slot";
        return false;
    }
    return true;
}

bool parse_flat_compact(
        const std::string & input,
        const std::vector<common_chat_tool> & tools,
        common_agent_tool_call & call,
        std::string & error) {
    std::vector<std::string> lines;
    std::istringstream input_stream(input);
    std::string line;
    while (std::getline(input_stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        line = trim(line);
        if (!line.empty()) lines.push_back(std::move(line));
    }
    if (lines.empty()) {
        error = "flat DSL tool call is empty";
        return false;
    }
    if (lines.size() == 1) return parse_compact(lines.front(), call, error);
    if (lines.front().rfind("open! ", 0) != 0) {
        error = "flat DSL must start with `open! TOOL_NAME`";
        return false;
    }
    const std::string tool_name = trim(lines.front().substr(6));
    if (tool_name.empty() || tool_name.find_first_of(" \t") != std::string::npos) {
        error = "flat DSL tool header must contain one exact tool name";
        return false;
    }
    const auto selected = std::find_if(tools.begin(), tools.end(), [&](const auto & tool) {
        return tool.name == tool_name;
    });
    if (selected == tools.end()) {
        error = "flat DSL selected a tool outside the available exact tool catalog";
        return false;
    }
    const auto schema = json::parse(selected->parameters, nullptr, false);
    if (!schema.is_object() || schema.value("type", std::string{}) != "object") {
        error = "flat DSL selected tool has no object input schema";
        return false;
    }

    json arguments = json::object();
    std::set<std::string> bracketed_array_fields;
    for (size_t index = 1; index < lines.size(); ++index) {
        const size_t separator = lines[index].find(':');
        if (separator == std::string::npos) {
            error = "flat DSL argument line must use `field: value`";
            return false;
        }
        const std::string field = trim(lines[index].substr(0, separator));
        const std::string source = trim(lines[index].substr(separator + 1));
        if (field.empty()) {
            error = "flat DSL argument field name is empty";
            return false;
        }
        if (const auto * directive = find_flat_directive(schema, field)) {
            json item;
            if (!parse_flat_directive_value(source, *directive, item, error)) return false;
            const auto property_name = flat_directive_property_name(schema, field);
            if (property_name.empty()) {
                error = "flat DSL directive has no schema destination: " + field;
                return false;
            }
            if (!arguments.contains(property_name)) arguments[property_name] = json::array();
            if (!arguments[property_name].is_array()) {
                error = "flat DSL directive conflicts with an existing field path: " + field;
                return false;
            }
            arguments[property_name].push_back(std::move(item));
            continue;
        }

        const auto properties = schema.value("properties", json::object());
        const bool scalar_array_field = properties.is_object() && properties.contains(field) &&
            properties[field].is_object() && properties[field].value("type", std::string{}) == "array" &&
            properties[field].value("items", json::object()).is_object() &&
            properties[field]["items"].value("type", std::string{}) != "object" &&
            properties[field]["items"].value("type", std::string{}) != "array";
        if (scalar_array_field && source != "null" && (source.empty() || source.front() != '[')) {
            if (bracketed_array_fields.count(field)) {
                error = "flat DSL cannot mix a bracketed list with repeated lines for field: " + field;
                return false;
            }
            json values;
            if (!parse_flat_scalar_array(source, values, error)) return false;
            if (!arguments.contains(field)) arguments[field] = json::array();
            if (!arguments[field].is_array()) {
                error = "flat DSL repeated field conflicts with a non-array value: " + field;
                return false;
            }
            for (auto & item : values) arguments[field].push_back(std::move(item));
            continue;
        }

        json value;
        if (!parse_flat_value(source, value, error)) return false;
        if (value.is_array()) {
            const auto properties = schema.value("properties", json::object());
            if (!properties.is_object() || !properties.contains(field) ||
                    !properties[field].is_object() ||
                    properties[field].value("type", std::string{}) != "array") {
                error = "flat DSL bracketed lists are valid only for array fields";
                return false;
            }
            for (const auto & item : value) {
                if (item.is_array() || item.is_object()) {
                    error = "flat DSL bracketed lists accept scalar values; use field paths or declared directives for nested items";
                    return false;
                }
            }
            if (arguments.contains(field)) {
                error = "flat DSL cannot mix a bracketed list with repeated lines for field: " + field;
                return false;
            }
            bracketed_array_fields.insert(field);
            arguments[field] = std::move(value);
            continue;
        }
        std::vector<std::variant<std::string, size_t>> path;
        if (!split_flat_path(field, path)) {
            error = "flat DSL field path is invalid: " + field;
            return false;
        }
        if (value.is_null() && path.size() == 1 && std::holds_alternative<std::string>(path.front())) {
            const auto key = std::get<std::string>(path.front());
            const auto required = schema.value("required", json::array());
            if (std::find(required.begin(), required.end(), json(key)) == required.end()) {
                // A null on an optional flat field is the model's explicit
                // "unset" value; normalize it to omission, never an empty value.
                continue;
            }
        }
        if (path.size() == 1 && std::holds_alternative<std::string>(path.front()) &&
                schema["properties"].contains(field) &&
                schema["properties"][field].value("type", std::string{}) == "array" &&
                !find_flat_directive(schema, field)) {
            if (bracketed_array_fields.count(field)) {
                error = "flat DSL cannot mix a bracketed list with repeated lines for field: " + field;
                return false;
            }
            if (!arguments.contains(field)) arguments[field] = json::array();
            if (!arguments[field].is_array()) {
                error = "flat DSL repeated field conflicts with a non-array value: " + field;
                return false;
            }
            arguments[field].push_back(std::move(value));
        } else if (!assign_flat_path(arguments, path, 0, value, error)) {
            return false;
        }
    }
    call.name = tool_name;
    call.arguments_json = arguments.dump();
    error.clear();
    return true;
}

bool compact_value_schema(const json & schema, std::string & reason, size_t depth = 0) {
    if (!schema.is_object()) {
        reason = "schema value is not an object";
        return false;
    }
    if (depth >= 12) {
        reason = "schema nesting exceeds the supported compact DSL depth";
        return false;
    }
    if (schema.contains("enum")) {
        if (!schema["enum"].is_array() || schema["enum"].empty()) {
            reason = "enum is not a non-empty array";
            return false;
        }
        for (const auto & value : schema["enum"]) {
            if (!value.is_string() && !value.is_boolean() && !value.is_number()) {
                reason = "enum contains a non-scalar value";
                return false;
            }
        }
        return true;
    }
    const auto type = schema.value("type", std::string{});
    if (type == "string" || type == "integer" || type == "number" || type == "boolean" || type == "null") return true;
    if (type == "array") {
        if (!schema.contains("items")) {
            reason = "array schema has no items definition";
            return false;
        }
        return compact_value_schema(schema["items"], reason, depth + 1);
    }
    if (type == "object") {
        const auto properties = schema.value("properties", json::object());
        if (!properties.is_object()) {
            reason = "object schema properties are invalid";
            return false;
        }
        for (const auto & property : properties.items()) {
            if (property.key().empty() || property.key().find_first_not_of(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string::npos) {
                reason = "object schema property name is not an identifier: " + property.key();
                return false;
            }
            if (!compact_value_schema(property.value(), reason, depth + 1)) return false;
        }
        return true;
    }
    reason = type.empty() ? "schema value has no supported type" : "unsupported compact DSL type: " + type;
    return false;
}

} // namespace

const char * common_agent_tool_output_format_name(common_agent_tool_output_format format) {
    switch (format) {
        case common_agent_tool_output_format::native: return "json";
        case common_agent_tool_output_format::jsonl: return "jsonl";
        case common_agent_tool_output_format::compact_dsl: return "dsl";
    }
    return "json";
}

bool common_parse_agent_tool_output_format(
        const std::string & value,
        common_agent_tool_output_format & format,
        std::string & error) {
    if (value == "json" || value == "native") format = common_agent_tool_output_format::native;
    else if (value == "jsonl") format = common_agent_tool_output_format::jsonl;
    else if (value == "dsl" || value == "compact_dsl") format = common_agent_tool_output_format::compact_dsl;
    else {
        error = "unsupported model tool output format: " + value;
        return false;
    }
    error.clear();
    return true;
}

bool common_compact_dsl_schema_supported(const std::string & schema_json, std::string & reason) {
    reason.clear();
    const auto schema = json::parse(schema_json, nullptr, false);
    if (schema.is_discarded() || !schema.is_object() || schema.value("type", std::string{}) != "object") {
        reason = "compact DSL requires an object input schema";
        return false;
    }
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object()) {
        reason = "compact DSL requires object properties";
        return false;
    }
    for (const auto & item : properties.items()) {
        if (item.key().empty()) {
            reason = "compact DSL property name is empty";
            return false;
        }
        for (const char character : item.key()) {
            if (!identifier_char(character)) {
                reason = "compact DSL property name is not an identifier: " + item.key();
                return false;
            }
        }
        if (!compact_value_schema(item.value(), reason)) return false;
        if (item.value().is_object() && item.value().contains("x-agent-flat")) {
            const auto & property = item.value();
            const auto & annotation = property["x-agent-flat"];
            const auto item_schema = property.value("items", json::object());
            const auto item_properties = item_schema.value("properties", json::object());
            if (property.value("type", std::string{}) != "array" || !annotation.is_object() ||
                    annotation.value("directive", std::string{}) != item.key() ||
                    !annotation.contains("positional") || !annotation["positional"].is_array() ||
                    annotation["positional"].empty() || !item_schema.is_object() ||
                    item_schema.value("type", std::string{}) != "object" || !item_properties.is_object()) {
                reason = "flat DSL directive annotation must name an object array and declare positional slots: " + item.key();
                return false;
            }
            std::set<std::string> slots;
            for (const auto & slot : annotation["positional"]) {
                if (!slot.is_string() || !item_properties.contains(slot.get<std::string>()) ||
                        !slots.insert(slot.get<std::string>()).second) {
                    reason = "flat DSL positional slot is unknown or repeated: " + item.key();
                    return false;
                }
            }
        }
    }
    return true;
}

bool common_parse_model_tool_call(
        common_agent_tool_output_format format,
        const std::string & text,
        common_agent_tool_call & call,
        std::string & error) {
    call = {};
    if (format == common_agent_tool_output_format::compact_dsl) {
        return parse_compact(text, call, error);
    }
    if (format != common_agent_tool_output_format::jsonl) {
        error = "native tool output must be parsed by the chat-template parser";
        return false;
    }
    const std::string line = trim(text);
    if (line.empty() || line.find('\n') != std::string::npos || line.find('\r') != std::string::npos) {
        error = "model JSONL tool output must be exactly one non-empty line";
        return false;
    }
    const auto value = json::parse(line, nullptr, false);
    if (value.is_discarded() || !value.is_object() || !value.contains("tool") ||
            !value["tool"].is_string() || !value.contains("arguments") ||
            !value["arguments"].is_object()) {
        error = "model JSONL tool output requires string tool and object arguments";
        return false;
    }
    call.name = value["tool"].get<std::string>();
    call.arguments_json = value["arguments"].dump();
    if (call.name.empty()) {
        error = "model JSONL tool output has an empty tool name";
        return false;
    }
    error.clear();
    return true;
}

bool common_parse_model_tool_call(
        common_agent_tool_output_format format,
        const std::string & text,
        const std::vector<common_chat_tool> & tools,
        common_agent_tool_call & call,
        std::string & error) {
    if (format == common_agent_tool_output_format::compact_dsl) {
        return parse_flat_compact(text, tools, call, error);
    }
    return common_parse_model_tool_call(format, text, call, error);
}

bool common_parse_compact_dsl_plan(
        const std::string & text,
        std::string & proposal_json,
        std::string & error) {
    return common_parse_compact_dsl_plan(text, {}, proposal_json, error);
}

bool common_parse_compact_dsl_plan(
        const std::string & text,
        const std::vector<common_chat_tool> & tools,
        std::string & proposal_json,
        std::string & error) {
    proposal_json.clear();
    error.clear();
    std::vector<std::string> lines;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        line = trim(line);
        if (!line.empty()) lines.push_back(std::move(line));
    }
    if (lines.size() < 2) {
        error = "compact DSL plan requires a goal and between one and six steps";
        return false;
    }
    const std::string header = lines.front();
    if (header.rfind("plan", 0) != 0 || (header.size() > 4 && !std::isspace(static_cast<unsigned char>(header[4])))) {
        error = "compact DSL plan must start with `plan goal=\"...\"`";
        return false;
    }
    size_t header_position = 4;
    while (header_position < header.size() && std::isspace(static_cast<unsigned char>(header[header_position]))) ++header_position;
    std::string goal_key;
    if (!parse_identifier(header, header_position, goal_key) || goal_key != "goal") {
        error = "compact DSL plan header requires a goal field";
        return false;
    }
    while (header_position < header.size() && std::isspace(static_cast<unsigned char>(header[header_position]))) ++header_position;
    if (header_position >= header.size() || header[header_position++] != '=') {
        error = "compact DSL plan goal must use goal=\"...\"";
        return false;
    }
    while (header_position < header.size() && std::isspace(static_cast<unsigned char>(header[header_position]))) ++header_position;
    json goal_value;
    if (!parse_value(header, header_position, goal_value, error) || !goal_value.is_string()) {
        if (error.empty()) error = "compact DSL plan goal must be a quoted string";
        return false;
    }
    while (header_position < header.size() && std::isspace(static_cast<unsigned char>(header[header_position]))) ++header_position;
    if (header_position != header.size() || goal_value.get<std::string>().empty() || goal_value.get<std::string>().size() > 256) {
        error = "compact DSL plan goal is empty, too long, or followed by extra content";
        return false;
    }

    std::vector<std::string> step_blocks;
    for (size_t index = 1; index < lines.size(); ++index) {
        if (lines[index].rfind("step ", 0) == 0) {
            step_blocks.push_back(lines[index]);
        } else if (step_blocks.empty()) {
            error = "compact DSL plan argument lines must follow a step";
            return false;
        } else {
            step_blocks.back() += "\n" + lines[index];
        }
    }
    if (step_blocks.empty() || step_blocks.size() > 6) {
        error = "compact DSL plan requires between one and six steps";
        return false;
    }

    json steps = json::array();
    std::set<std::string> aliases;
    for (const auto & step_line : step_blocks) {
        if (step_line.rfind("step ", 0) != 0) {
            error = "compact DSL plan step must start with `step`";
            return false;
        }
        // `step | open! ...` has its separator beginning at offset 4; an
        // optional `as=alias` may move it farther right. Searching from 5
        // accidentally rejected the documented no-alias form.
        const size_t separator = step_line.find(" | ", 4);
        if (separator == std::string::npos) {
            error = "compact DSL plan step requires ` | open! TOOL ...`";
            return false;
        }
        const std::string metadata = separator == 4
            ? std::string()
            : step_line.substr(5, separator - 5);
        std::string alias;
        size_t meta_position = 0;
        while (meta_position < metadata.size()) {
            while (meta_position < metadata.size() && std::isspace(static_cast<unsigned char>(metadata[meta_position]))) ++meta_position;
            if (meta_position >= metadata.size()) break;
            std::string key;
            std::string value;
            if (!parse_identifier(metadata, meta_position, key) || key != "as") {
                error = "compact DSL plan step metadata supports only `as=alias`";
                return false;
            }
            while (meta_position < metadata.size() && std::isspace(static_cast<unsigned char>(metadata[meta_position]))) ++meta_position;
            if (meta_position >= metadata.size() || metadata[meta_position++] != '=') {
                error = "compact DSL plan alias must use as=alias";
                return false;
            }
            if (!parse_identifier(metadata, meta_position, value) || value == "previous" || value.size() > 64 || !alias.empty()) {
                error = "compact DSL plan alias is invalid, reserved, or repeated";
                return false;
            }
            alias = std::move(value);
        }
        if (!alias.empty() && !aliases.insert(alias).second) {
            error = "compact DSL plan repeats alias: " + alias;
            return false;
        }
        common_agent_tool_call call;
        const std::string call_text = step_line.substr(separator + 3);
        const bool parsed_call = tools.empty()
            ? parse_compact(call_text, call, error)
            : parse_flat_compact(call_text, tools, call, error);
        if (!parsed_call) return false;
        json step = {{"tool", call.name}, {"args", json::parse(call.arguments_json)}};
        if (!alias.empty()) step["as"] = alias;
        steps.push_back(std::move(step));
    }
    proposal_json = json{{"goal", goal_value}, {"steps", std::move(steps)}}.dump();
    return true;
}

bool common_normalize_comma_separated_string(
        const std::string & value,
        std::string & normalized,
        std::string & error) {
    error.clear();
    normalized.clear();
    size_t start = 0;
    while (start <= value.size()) {
        const size_t separator = value.find(',', start);
        const std::string item = trim(value.substr(
            start, separator == std::string::npos ? std::string::npos : separator - start));
        if (item.empty()) {
            error = "comma-separated string contains an empty item";
            normalized.clear();
            return false;
        }
        if (!normalized.empty()) normalized.push_back(',');
        normalized += item;
        if (separator == std::string::npos) break;
        start = separator + 1;
    }
    return true;
}

std::string common_render_model_tool_output_instructions(
        common_agent_tool_output_format format,
        const std::vector<common_chat_tool> & tools,
        std::vector<std::string> * unsupported_names,
        std::string & error) {
    error.clear();
    if (unsupported_names != nullptr) unsupported_names->clear();
    if (format == common_agent_tool_output_format::native) return {};
    std::ostringstream out;
    out << "Tool execution is required. Emit exactly one tool call and no explanation.\n";
    if (format == common_agent_tool_output_format::jsonl) {
        out << "Use one JSON line: {\"tool\":\"TOOL_NAME\",\"arguments\":{...}}.\n";
    } else {
        out << "Use flat DSL: first write `open! TOOL_NAME`, then one `field: value` per line.\n"
            << "For scalar array fields, use either comma-separated values such as `columns: region, amount` or a bracketed list such as `columns: [region, amount]`; repeating the field on separate lines is also accepted. Quote scalar strings containing commas, for example `select: \"id, display_name\"`.\n"
            << "For nested values without a declared directive, use paths such as `operations[0].column: amount`. For annotated directives, use the listed positional slots by default; named `slot=value` entries are also accepted. Quote strings containing spaces. Omit unset fields; never use an empty value. Do not write nested object literals.\n";
    }
    out << "Available model-facing tools:\n";
    size_t rendered = 0;
    for (const auto & tool : tools) {
        std::string reason;
        if (format == common_agent_tool_output_format::compact_dsl &&
                !common_compact_dsl_schema_supported(tool.parameters, reason)) {
            if (unsupported_names != nullptr) unsupported_names->push_back(tool.name);
            continue;
        }
        std::string compact_error;
        const auto description = common_render_compact_tool_description(
            tool.name, tool.description, tool.parameters, tool.result_schema, compact_error);
        if (!compact_error.empty()) {
            error = compact_error;
            return {};
        }
        out << "- " << description << "\n";
        ++rendered;
    }
    if (rendered == 0) {
        error = "model tool output format has no representable tools";
        return {};
    }
    return out.str();
}
