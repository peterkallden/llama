#include "agent/tool-registry.h"
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

bool common_tool_registry::register_tool(common_registered_tool tool, std::string & error) {
    if (tool.name.empty() || !tool.handler) { error = "registered tool requires a name and handler"; return false; }
    if (tools.count(tool.name)) { error = "tool is already registered"; return false; }
    const auto schema = json::parse(tool.arguments_schema, nullptr, false);
    if (!schema.is_object() || schema.value("type", std::string()) != "object") { error = "tool arguments schema must describe an object"; return false; }
    tools.emplace(tool.name, std::move(tool)); error.clear(); return true;
}

bool common_tool_registry::execute(const common_registered_tool_call & call, std::string & result, std::string & error) const {
    const auto it = tools.find(call.name); if (it == tools.end()) { error = "tool is not registered"; return false; }
    const auto arguments = json::parse(call.arguments_json, nullptr, false); if (!arguments.is_object()) { error = "tool arguments must be a JSON object"; return false; }
    const auto schema = json::parse(it->second.arguments_schema); const auto required = schema.value("required", json::array()); const auto properties = schema.value("properties", json::object());
    for (const auto & key : required) if (!key.is_string() || !arguments.contains(key.get<std::string>())) { error = "required tool argument is missing"; return false; }
    if (schema.value("additionalProperties", true) == false) for (auto arg = arguments.begin(); arg != arguments.end(); ++arg) if (!properties.contains(arg.key())) { error = "unexpected tool argument"; return false; }
    return it->second.handler(call.arguments_json, result, error);
}

bool common_tool_registry::contains(const std::string & name) const {
    return tools.count(name) != 0;
}

bool common_tool_registry::is_read_only(const std::string & name) const {
    const auto it = tools.find(name);
    return it != tools.end() && it->second.read_only;
}
