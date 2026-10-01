#include "agent/tooling/schema/tool-output-codec.h"

#include "agent/tooling/schema/tool-schema-compact.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cmath>
#include <map>
#include <sstream>

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

bool parse_value(const std::string & text, size_t & position, json & value, std::string & error) {
    if (position < text.size() && text[position] == '"') {
        return parse_quoted(text, position, value, error);
    }
    const size_t start = position;
    while (position < text.size() && !std::isspace(static_cast<unsigned char>(text[position])) &&
            text[position] != ',' && text[position] != ')') {
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
    const auto number = json::parse(token, nullptr, false);
    if (!number.is_discarded() && number.is_number()) {
        value = number;
        return true;
    }
    if (token.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-") != std::string::npos) {
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

bool scalar_schema(const json & schema, std::string & reason) {
    if (!schema.is_object()) {
        reason = "schema property is not an object";
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
    if (type == "string" || type == "integer" || type == "number" || type == "boolean") return true;
    reason = type.empty() ? "schema property has no scalar type" : "unsupported compact DSL type: " + type;
    return false;
}

} // namespace

const char * common_agent_tool_output_format_name(common_agent_tool_output_format format) {
    switch (format) {
        case common_agent_tool_output_format::native: return "native";
        case common_agent_tool_output_format::jsonl: return "jsonl";
        case common_agent_tool_output_format::compact_dsl: return "compact_dsl";
    }
    return "native";
}

bool common_parse_agent_tool_output_format(
        const std::string & value,
        common_agent_tool_output_format & format,
        std::string & error) {
    if (value == "native") format = common_agent_tool_output_format::native;
    else if (value == "jsonl") format = common_agent_tool_output_format::jsonl;
    else if (value == "compact_dsl") format = common_agent_tool_output_format::compact_dsl;
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
        if (!scalar_schema(item.value(), reason)) return false;
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
        out << "Use one compact DSL line: open! TOOL_NAME key=value ... .\n"
            << "A compatible call form is TOOL_NAME(key=value, ...).\n";
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
