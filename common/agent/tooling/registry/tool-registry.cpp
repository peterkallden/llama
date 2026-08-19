#include "agent/tooling/registry/tool-registry.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/dataset-contracts.h"
#include "plan/plan-json.h"

bool common_tool_registry::register_tool(common_registered_tool tool, std::string & error) {
    if (tool.name.empty() || !tool.handler) { error = "registered tool requires a name and handler"; return false; }
    if (tools.count(tool.name)) { error = "tool is already registered"; return false; }
    const auto schema = common_json_contract_value::parse(tool.arguments_schema, nullptr, false);
    if (tool.version == 0 || tool.executor_id.empty() || !schema.is_object() || schema.value("type", std::string()) != "object" ||
            (schema.contains("required") && !schema["required"].is_array()) ||
            (schema.contains("properties") && !schema["properties"].is_object())) {
        error = "registered tool has invalid capability binding";
        return false;
    }
    if (schema.contains("required")) for (const auto & field : schema["required"]) {
        if (!field.is_string()) { error = "registered tool has invalid capability binding"; return false; }
    }
    tools.emplace(tool.name, std::move(tool)); error.clear(); return true;
}

bool common_tool_registry::validate(const common_registered_tool_call & call, std::string & error) const {
    const auto it = tools.find(call.name); if (it == tools.end()) { error = "tool is not registered"; return false; }
    std::string normalized;
    if (!common_plan_normalize_tool_arguments_json(call.name, call.arguments_json, normalized, error)) return false;
    auto parsed = common_json_contract_value::parse(normalized, nullptr, false);
    if (!parsed.is_object() || !normalize_common_agent_dataset_tool_arguments(call.name, parsed, error)) return false;
    normalized = parsed.dump();
    return common_schema_normalize_and_validate_object(normalized, it->second.arguments_schema, normalized, error);
}

common_tool_execution_result common_tool_registry::execute(const common_registered_tool_call & call) const {
    std::string error;
    const auto it = tools.find(call.name);
    if (it == tools.end()) return common_tool_execution_result::failure("tool.unknown", common_tool_failure_class::validation, false, "Tool is not registered.", "tool is not registered");
    std::string normalized;
    if (!common_plan_normalize_tool_arguments_json(call.name, call.arguments_json, normalized, error)) {
        return common_tool_execution_result::failure("tool.invalid_arguments", common_tool_failure_class::validation, false, "Tool arguments do not satisfy the registered contract.", std::move(error));
    }
    auto parsed = common_json_contract_value::parse(normalized, nullptr, false);
    if (!parsed.is_object() || !normalize_common_agent_dataset_tool_arguments(call.name, parsed, error)) {
        return common_tool_execution_result::failure("tool.invalid_arguments", common_tool_failure_class::validation, false, "Tool arguments do not satisfy the registered contract.", std::move(error));
    }
    normalized = parsed.dump();
    if (!common_schema_normalize_and_validate_object(normalized, it->second.arguments_schema, normalized, error)) {
        return common_tool_execution_result::failure("tool.invalid_arguments", common_tool_failure_class::validation, false, "Tool arguments do not satisfy the registered contract.", std::move(error));
    }
    return it->second.handler(normalized);
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
