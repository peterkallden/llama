#include "agent/tool-family-index.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

using json = nlohmann::ordered_json;

namespace {

std::string family_id_for_tool(const std::string & name) {
    if (name == "calculator") return "math";
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
        {"diagnostics", "Analyze compiler and test failures, native crashes and debugger dumps"},
        {"document",   "Extract structured content from documents"},
        {"mcp",        "Discover and invoke MCP resources and tools"},
        {"repository", "Search and inspect repository-backed content"},
        {"math",       "Perform bounded arithmetic calculations"},
        {"memory",     "Search and manage scoped runtime memory"},
        {"resource",   "Inspect and read host-owned resources"},
        {"statistics", "Describe datasets and compute summaries"},
        {"time",       "Read current time and date information"},
        {"web",        "Search and retrieve information from the public web"},
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
        // Keep tool_names available for host-side family filtering, but do not
        // expose individual tools in the first routing context. Exact names
        // and contracts are rendered only after family selection.
        const std::string entry = "\n- " + family.id + ": " + family.description;
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

bool common_parse_tool_family_selection_text(
        const std::string & text,
        const std::vector<common_tool_family_index> & families,
        common_tool_family_selection & selection,
        std::string & error) {
    selection = {};
    std::string line = text.substr(0, text.find('\n'));
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
    size_t first = 0;
    while (first < line.size() && std::isspace(static_cast<unsigned char>(line[first]))) ++first;
    line.erase(0, first);
    std::string upper = line;
    for (char & ch : upper) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (upper.rfind("NO_TOOLS", 0) == 0 || upper.rfind("NO TOOLS", 0) == 0) {
        selection.needs_tools = false;
        return true;
    }
    if (upper.rfind("TOOLS:", 0) != 0) {
        error = "tool family selection must start with NO_TOOLS or TOOLS:";
        return false;
    }
    std::set<std::string> available;
    for (const auto & family : families) available.insert(family.id);
    std::string ids = line.substr(6);
    for (char & ch : ids) if (ch == ',' || ch == ';') ch = ' ';
    std::istringstream stream(ids);
    std::string id;
    selection.needs_tools = true;
    while (stream >> id) {
        if (!available.count(id)) {
            error = "tool family selection returned unknown family: " + id;
            return false;
        }
        if (std::find(selection.family_ids.begin(), selection.family_ids.end(), id) != selection.family_ids.end()) {
            error = "tool family selection contains duplicate family: " + id;
            return false;
        }
        selection.family_ids.push_back(id);
    }
    if (selection.family_ids.empty()) {
        error = "tool family selection must include at least one family after TOOLS:";
        return false;
    }
    error.clear();
    return true;
}
