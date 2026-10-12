#pragma once

#include "agent/tooling/adapters/tool-adapters.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <string>

using common_adapter_json = nlohmann::ordered_json;

inline bool common_adapter_parse_object(
        const std::string & input, common_adapter_json & output, std::string & error) {
    output = common_adapter_json::parse(input, nullptr, false);
    if (!output.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }
    return true;
}

inline common_tool_execution_result common_adapter_success_json(const common_adapter_json & value) {
    return common_tool_execution_result::success(value.dump());
}

inline common_tool_execution_result common_adapter_success_text(std::string value) {
    return common_tool_execution_result::success(std::move(value));
}

inline common_tool_execution_result common_adapter_failure(
        std::string code, common_tool_failure_class kind, bool retryable,
        std::string summary, std::string diagnostic) {
    return common_tool_execution_result::failure(
            std::move(code), kind, retryable, std::move(summary), std::move(diagnostic));
}

inline common_tool_execution_result common_adapter_validation_failure(
        std::string code, std::string diagnostic,
        std::string summary = "Tool arguments are invalid.") {
    return common_adapter_failure(std::move(code), common_tool_failure_class::validation,
            false, std::move(summary), std::move(diagnostic));
}

inline common_tool_execution_result common_adapter_execution_failure(
        std::string code, std::string diagnostic, std::string summary) {
    return common_adapter_failure(std::move(code), common_tool_failure_class::execution,
            false, std::move(summary), std::move(diagnostic));
}

inline common_tool_execution_result common_adapter_not_found_failure(
        std::string code, std::string diagnostic, std::string summary) {
    return common_adapter_failure(std::move(code), common_tool_failure_class::not_found,
            false, std::move(summary), std::move(diagnostic));
}

inline common_tool_execution_result common_adapter_limit_failure(
        std::string code, std::string diagnostic, std::string summary) {
    return common_adapter_failure(std::move(code), common_tool_failure_class::limit,
            false, std::move(summary), std::move(diagnostic));
}

inline bool common_adapter_register_definition(
        const common_tool_definition & definition,
        common_tool_registry & registry,
        std::function<common_tool_execution_result(const std::string &)> handler,
        std::string & error, bool read_only = true, bool policy_gated = false) {
    common_registered_tool tool;
    tool.name = definition.name;
    tool.version = definition.version;
    tool.executor_id = definition.executor_id;
    tool.arguments_schema = definition.input_schema_json;
    tool.read_only = read_only;
    tool.policy_gated = policy_gated;
    tool.handler = std::move(handler);
    return registry.register_tool(std::move(tool), error);
}
