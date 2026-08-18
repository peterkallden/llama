#include "agent/tool-schema-compact.h"

#include <nlohmann/json.hpp>

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

std::string render_returns(const std::string & schema_json) {
    const auto schema = json::parse(schema_json, nullptr, false);
    if (!schema.is_object()) return "value";
    const auto properties = schema.value("properties", json::object());
    if (!properties.is_object() || properties.empty()) return "value";

    std::ostringstream out;
    bool first = true;
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (!first) out << ", ";
        first = false;
        out << it.key() << ':' << scalar_type(it.value());
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

std::string common_render_compact_tool_description(
        const std::string & name,
        const std::string & description,
        const std::string & input_schema_json,
        const std::string & result_schema_json,
        std::string & error) {
    const auto args = common_render_compact_tool_schema(input_schema_json, error);
    if (!error.empty()) return {};

    std::ostringstream out;
    out << name << "\n" << description << "\n";
    out << "args: " << (args.empty() ? "object" : args) << "\n";
    out << "returns: " << render_returns(result_schema_json);
    return out.str();
}
