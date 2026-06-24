#include "agent/tool-registry.h"
#include "agent/schema-contract.h"

bool common_tool_registry::register_tool(common_registered_tool tool, std::string & error) {
    if (tool.name.empty() || !tool.handler) { error = "registered tool requires a name and handler"; return false; }
    if (tools.count(tool.name)) { error = "tool is already registered"; return false; }
    std::string ignored;
    if (tool.version == 0 || tool.executor_id.empty() || !common_schema_normalize_and_validate_object("{}", tool.arguments_schema, ignored, error)) {
        if (error == "required contract field is missing") error.clear();
        if (error.empty()) error = "registered tool has invalid capability binding";
        return false;
    }
    tools.emplace(tool.name, std::move(tool)); error.clear(); return true;
}

bool common_tool_registry::validate(const common_registered_tool_call & call, std::string & error) const {
    const auto it = tools.find(call.name); if (it == tools.end()) { error = "tool is not registered"; return false; }
    std::string normalized;
    return common_schema_normalize_and_validate_object(call.arguments_json, it->second.arguments_schema, normalized, error);
}

bool common_tool_registry::execute(const common_registered_tool_call & call, std::string & result, std::string & error) const {
    if (!validate(call, error)) return false;
    const auto it = tools.find(call.name);
    return it->second.handler(call.arguments_json, result, error);
}

bool common_tool_registry::contains(const std::string & name) const {
    return tools.count(name) != 0;
}

bool common_tool_registry::matches_binding(const std::string & name, uint32_t version, const std::string & executor_id) const {
    const auto it = tools.find(name);
    return it != tools.end() && it->second.version == version && it->second.executor_id == executor_id;
}

bool common_tool_registry::is_read_only(const std::string & name) const {
    const auto it = tools.find(name);
    return it != tools.end() && it->second.read_only;
}

bool common_tool_registry::is_policy_gated(const std::string & name) const {
    const auto it = tools.find(name);
    return it != tools.end() && it->second.policy_gated;
}
