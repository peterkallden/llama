#include "agent/tool-registry.h"
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

namespace {

bool validate_value(const json & value, const json & schema, const std::string & field, std::string & error) {
    const auto type = schema.value("type", std::string());
    const bool type_ok = type.empty() ||
        (type == "string" && value.is_string()) || (type == "integer" && value.is_number_integer()) ||
        (type == "number" && value.is_number()) || (type == "boolean" && value.is_boolean()) ||
        (type == "array" && value.is_array()) || (type == "object" && value.is_object());
    if (!type_ok) { error = "tool argument '" + field + "' has invalid type"; return false; }
    if (schema.contains("enum")) {
        bool found = false;
        for (const auto & allowed : schema["enum"]) if (allowed == value) { found = true; break; }
        if (!found) { error = "tool argument '" + field + "' is not an allowed value"; return false; }
    }
    if (value.is_string()) {
        const auto size = value.get_ref<const std::string &>().size();
        if (schema.contains("minLength") && size < schema["minLength"].get<size_t>()) { error = "tool argument '" + field + "' is too short"; return false; }
        if (schema.contains("maxLength") && size > schema["maxLength"].get<size_t>()) { error = "tool argument '" + field + "' is too long"; return false; }
    }
    if (value.is_number()) {
        const auto number = value.get<double>();
        if (schema.contains("minimum") && number < schema["minimum"].get<double>()) { error = "tool argument '" + field + "' is below its minimum"; return false; }
        if (schema.contains("maximum") && number > schema["maximum"].get<double>()) { error = "tool argument '" + field + "' exceeds its maximum"; return false; }
    }
    if (value.is_array()) {
        if (schema.contains("minItems") && value.size() < schema["minItems"].get<size_t>()) { error = "tool argument '" + field + "' has too few items"; return false; }
        if (schema.contains("maxItems") && value.size() > schema["maxItems"].get<size_t>()) { error = "tool argument '" + field + "' has too many items"; return false; }
        if (schema.contains("items")) for (size_t i = 0; i < value.size(); ++i) if (!validate_value(value[i], schema["items"], field + "[" + std::to_string(i) + "]", error)) return false;
    }
    return true;
}

} // namespace

bool common_tool_registry::register_tool(common_registered_tool tool, std::string & error) {
    if (tool.name.empty() || !tool.handler) { error = "registered tool requires a name and handler"; return false; }
    if (tools.count(tool.name)) { error = "tool is already registered"; return false; }
    const auto schema = json::parse(tool.arguments_schema, nullptr, false);
    if (!schema.is_object() || schema.value("type", std::string()) != "object") { error = "tool arguments schema must describe an object"; return false; }
    tools.emplace(tool.name, std::move(tool)); error.clear(); return true;
}

bool common_tool_registry::validate(const common_registered_tool_call & call, std::string & error) const {
    const auto it = tools.find(call.name); if (it == tools.end()) { error = "tool is not registered"; return false; }
    const auto arguments = json::parse(call.arguments_json, nullptr, false); if (!arguments.is_object()) { error = "tool arguments must be a JSON object"; return false; }
    const auto schema = json::parse(it->second.arguments_schema); const auto required = schema.value("required", json::array()); const auto properties = schema.value("properties", json::object());
    for (const auto & key : required) if (!key.is_string() || !arguments.contains(key.get<std::string>())) { error = "required tool argument is missing"; return false; }
    if (schema.value("additionalProperties", true) == false) for (auto arg = arguments.begin(); arg != arguments.end(); ++arg) if (!properties.contains(arg.key())) { error = "unexpected tool argument"; return false; }
    for (auto arg = arguments.begin(); arg != arguments.end(); ++arg) if (properties.contains(arg.key()) && !validate_value(arg.value(), properties[arg.key()], arg.key(), error)) return false;
    error.clear();
    return true;
}

bool common_tool_registry::execute(const common_registered_tool_call & call, std::string & result, std::string & error) const {
    if (!validate(call, error)) return false;
    const auto it = tools.find(call.name);
    return it->second.handler(call.arguments_json, result, error);
}

bool common_tool_registry::contains(const std::string & name) const {
    return tools.count(name) != 0;
}

bool common_tool_registry::is_read_only(const std::string & name) const {
    const auto it = tools.find(name);
    return it != tools.end() && it->second.read_only;
}

bool common_tool_registry::is_policy_gated(const std::string & name) const {
    const auto it = tools.find(name);
    return it != tools.end() && it->second.policy_gated;
}
