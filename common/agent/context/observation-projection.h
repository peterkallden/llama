#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>

#include <nlohmann/json.hpp>

using common_agent_projection_json = nlohmann::ordered_json;

inline common_agent_projection_json common_agent_project_observation_json(
        const common_agent_projection_json & value, size_t depth, size_t max_depth,
        size_t max_object_fields, size_t max_array_items, size_t max_string_chars) {
    if (value.is_string()) {
        auto text = value.get<std::string>();
        if (text.size() > max_string_chars) text.resize(max_string_chars);
        return text;
    }
    if (value.is_primitive() || value.is_null()) return value;
    if (depth >= max_depth) {
        return value.is_array()
            ? common_agent_projection_json::array()
            : common_agent_projection_json{{"omitted_field_count", value.size()}};
    }
    if (value.is_array()) {
        common_agent_projection_json out = common_agent_projection_json::array();
        for (size_t i = 0; i < std::min(value.size(), max_array_items); ++i)
            out.push_back(common_agent_project_observation_json(value[i], depth + 1, max_depth, max_object_fields, max_array_items, max_string_chars));
        return out;
    }
    common_agent_projection_json out = common_agent_projection_json::object();
    // Result status, references, and bounded summaries are more useful for
    // continuation than arbitrary early fields in a large provider payload.
    static const std::array<const char *, 14> priority = {"status", "error", "failure_code", "retryable",
        "resource_ref", "resource_refs", "resource_uri", "dataset_ref", "dataset_refs", "dataset_uri",
        "summary", "result_summary", "content_summary", "result_truncated"};
    for (const auto * key : priority) if (value.contains(key) && out.size() < max_object_fields)
        out[key] = common_agent_project_observation_json(value.at(key), depth + 1, max_depth, max_object_fields, max_array_items, max_string_chars);
    for (const auto & item : value.items()) {
        if (out.size() >= max_object_fields) break;
        if (out.contains(item.key())) continue;
        out[item.key()] = common_agent_project_observation_json(item.value(), depth + 1, max_depth, max_object_fields, max_array_items, max_string_chars);
    }
    const size_t omitted = value.size() > out.size() ? value.size() - out.size() : 0;
    if (omitted > 0 && out.size() < max_object_fields) out["omitted_field_count"] = omitted;
    return out;
}
