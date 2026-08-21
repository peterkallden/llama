#include "agent/tool-family-index.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>

using json = nlohmann::ordered_json;

namespace {

std::string family_id_for_tool(const std::string & name) {
    const auto dot = name.find('.');
    if (dot != std::string::npos && dot > 0) return name.substr(0, dot);

    // A few older/native names use an underscore namespace instead of the
    // dotted form. Keep the grouping deterministic while preserving the
    // original tool name in the generated index.
    const auto underscore = name.find('_');
    if (underscore != std::string::npos && underscore > 0) return name.substr(0, underscore);
    return name;
}

std::string family_description_for_id(const std::string & id) {
    static const std::unordered_map<std::string, std::string> descriptions = {
        {"data",       "Query and transform datasets"},
        {"dataset",    "Choose and inspect datasets for analysis"},
        {"diagnostics", "Explain failures and inspect execution evidence"},
        {"document",   "Extract structured content from documents"},
        {"mcp",        "Discover and invoke MCP resources and tools"},
        {"repository", "Search and inspect repository-backed content"},
        {"resource",   "Resolve and inspect external resources"},
        {"statistics", "Describe datasets and compute summaries"},
        {"web",        "Search and retrieve information from the web"},
    };
    const auto it = descriptions.find(id);
    return it != descriptions.end()
        ? it->second
        : "Operations provided by the " + id + " tool family";
}

} // namespace

std::vector<common_tool_family_index> common_generate_tool_family_index(
        const std::vector<common_chat_tool> & tools) {
    std::map<std::string, std::set<std::string>> grouped;
    for (const auto & tool : tools) {
        if (tool.name.empty()) continue;
        grouped[family_id_for_tool(tool.name)].insert(tool.name);
    }

    std::vector<common_tool_family_index> result;
    result.reserve(grouped.size());
    for (auto & entry : grouped) {
        common_tool_family_index family;
        family.id = std::move(entry.first);
        family.description = family_description_for_id(family.id);
        family.tool_names.assign(entry.second.begin(), entry.second.end());
        result.push_back(std::move(family));
    }
    return result;
}

std::string common_render_tool_family_index(
        const std::vector<common_tool_family_index> & families,
        size_t max_chars) {
    std::string rendered = "tool families:";
    for (const auto & family : families) {
        std::set<std::string> operations;
        for (const auto & tool_name : family.tool_names) {
            const auto dot = tool_name.find('.');
            const auto underscore = tool_name.find('_');
            const auto separator = dot != std::string::npos ? dot : underscore;
            if (separator != std::string::npos && separator + 1 < tool_name.size()) {
                operations.insert(tool_name.substr(separator + 1));
            }
        }
        std::string entry = "\n- " + family.id + ": " + family.description + "; operations: ";
        if (!operations.empty()) {
            size_t operation_index = 0;
            for (const auto & operation : operations) {
                if (operation_index++ > 0) entry += ", ";
                entry += operation;
            }
        } else {
            entry += family.id;
        }
        if (rendered.size() + entry.size() > max_chars) break;
        rendered += entry;
    }
    return rendered;
}

std::vector<common_chat_tool> common_filter_tools_by_families(
        const std::vector<common_chat_tool> & tools,
        const std::vector<std::string> & family_ids) {
    const std::set<std::string> selected(family_ids.begin(), family_ids.end());
    std::vector<common_chat_tool> result;
    for (const auto & tool : tools) {
        if (selected.count(family_id_for_tool(tool.name))) result.push_back(tool);
    }
    return result;
}

std::string common_tool_family_selection_schema() {
    return R"({"type":"object","additionalProperties":false,"required":["needs_tools","families"],"properties":{"needs_tools":{"type":"boolean"},"families":{"type":"array","maxItems":8,"uniqueItems":true,"items":{"type":"string","minLength":1,"maxLength":64}},"reason":{"type":"string","maxLength":256}}})";
}

bool common_parse_tool_family_selection(
        const std::string & json_text,
        common_tool_family_selection & selection,
        std::string & error) {
    selection = {};
    const auto value = json::parse(json_text, nullptr, false);
    if (value.is_discarded() || !value.is_object()) {
        error = "tool family selection must be a JSON object";
        return false;
    }
    if (!value.contains("needs_tools") || !value["needs_tools"].is_boolean() ||
            !value.contains("families") || !value["families"].is_array()) {
        error = "tool family selection requires needs_tools:boolean and families:string[]";
        return false;
    }
    if (value["families"].size() > 8) {
        error = "tool family selection contains too many families";
        return false;
    }
    selection.needs_tools = value["needs_tools"].get<bool>();
    std::set<std::string> seen;
    for (const auto & item : value["families"]) {
        if (!item.is_string() || item.get<std::string>().empty()) {
            error = "tool family selection families must contain non-empty strings";
            return false;
        }
        if (!seen.insert(item.get<std::string>()).second) {
            error = "tool family selection families must be unique";
            return false;
        }
        selection.family_ids.push_back(item.get<std::string>());
    }
    error.clear();
    return true;
}
