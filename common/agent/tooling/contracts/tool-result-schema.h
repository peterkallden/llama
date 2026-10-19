#pragma once

#include <nlohmann/json.hpp>
#include <string>

// Native providers retain the canonical result in a common success payload.
// MCP schemas describe that actual structuredContent, including its envelope.
inline nlohmann::ordered_json common_tool_success_payload_schema(const std::string & result_schema_json) {
    using json = nlohmann::ordered_json;
    const auto result = json::parse(result_schema_json, nullptr, false);
    if (!result.is_object()) return json();
    return {
        {"type", "object"},
        {"required", {"ok", "result"}},
        {"properties", {
            {"ok", {{"type", "boolean"}, {"const", true}}},
            {"result", result},
            {"summary", {{"type", "string"}}},
            {"resources", {{"type", "array"}, {"items", {{"type", "object"}}}}},
            {"dataset_refs", {{"type", "array"}, {"items", {{"type", "object"}}}}},
        }},
    };
}
