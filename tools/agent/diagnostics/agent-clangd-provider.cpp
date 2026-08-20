#include "agent-clangd-provider.h"

#include <filesystem>

namespace {
using json = agent_clangd_json;

common_tool_execution_result provider_error(const std::string & code, const std::string & detail) {
    return common_tool_execution_result::failure(code, common_tool_failure_class::execution, false, "Clangd semantic diagnostics are unavailable.", detail);
}

std::string uri_path(const std::string & uri) {
    if (uri.rfind("file://", 0) == 0) return uri.substr(7);
    return uri;
}

json location_json(const json & location) {
    const auto & range = location.value("range", json::object());
    const auto & start = range.value("start", json::object());
    return {
        {"uri", location.value("uri", location.value("targetUri", std::string()))},
        {"path", uri_path(location.value("uri", location.value("targetUri", std::string())))},
        {"line", start.value("line", 0) + 1},
        {"column", start.value("character", 0) + 1},
    };
}
}

agent_clangd_diagnostics_provider::agent_clangd_diagnostics_provider(agent_clangd_session_config config)
    : session_(std::make_unique<agent_clangd_session>(config)), repository_root_(std::move(config.repository_root)) {}

common_tool_execution_result agent_clangd_diagnostics_provider::symbol(const std::string & arguments_json) {
    const auto arguments = json::parse(arguments_json, nullptr, false);
    if (!arguments.is_object() || !arguments.contains("symbol") || !arguments["symbol"].is_string()) return common_tool_execution_result::failure("tool.diagnostics.symbol.invalid_arguments", common_tool_failure_class::validation, false, "Symbol arguments are invalid.");
    return workspace_symbol(arguments);
}

common_tool_execution_result agent_clangd_diagnostics_provider::workspace_symbol(const json & arguments) {
    json response;
    std::string error;
    if (!session_->request("workspace/symbol", {{"query", arguments["symbol"]}}, response, error)) return provider_error("tool.diagnostics.symbol.provider_unavailable", error);
    json definitions = json::array();
    const auto result = response.value("result", json::array());
    const auto limit = std::min<size_t>(arguments.value("max_results", 64), 64);
    if (result.is_array()) for (const auto & item : result) {
        if (definitions.size() >= limit || !item.is_object() || !item.contains("location")) break;
        auto definition = location_json(item["location"]);
        definition["name"] = item.value("name", arguments["symbol"]);
        definition["kind"] = item.value("kind", 0);
        definitions.push_back(std::move(definition));
    }
    return common_tool_execution_result::success(json({{"backend", "clangd"}, {"semantic", true}, {"symbol", arguments["symbol"]}, {"definitions", definitions}, {"count", definitions.size()}, {"truncated", definitions.size() >= limit}}).dump());
}

bool agent_clangd_diagnostics_provider::resolve_location(const json & arguments, json & location, std::string & error) {
    if (arguments.contains("definition_path") && arguments.contains("definition_line")) {
        const auto path = std::filesystem::path(arguments["definition_path"].get<std::string>()).is_absolute()
            ? std::filesystem::path(arguments["definition_path"].get<std::string>())
            : std::filesystem::path(repository_root_) / arguments["definition_path"].get<std::string>();
        location = {{"uri", "file:///" + path.generic_string()}, {"range", {{"start", {{"line", arguments["definition_line"].get<int>() - 1}, {"character", arguments.value("definition_column", 1) - 1}}}}}};
        return true;
    }
    if (arguments.contains("path_hint") && arguments.contains("line")) {
        const auto path = std::filesystem::path(arguments["path_hint"].get<std::string>()).is_absolute()
            ? std::filesystem::path(arguments["path_hint"].get<std::string>())
            : std::filesystem::path(repository_root_) / arguments["path_hint"].get<std::string>();
        location = {{"uri", "file:///" + path.generic_string()}, {"range", {{"start", {{"line", arguments["line"].get<int>() - 1}, {"character", arguments.value("column", 1) - 1}}}}}};
        return true;
    }
    error = "semantic diagnostics request requires a source location";
    return false;
}

common_tool_execution_result agent_clangd_diagnostics_provider::location_query(const json & arguments, const std::string & method) {
    json location;
    std::string error;
    if (!resolve_location(arguments, location, error)) return common_tool_execution_result::failure("tool.diagnostics.semantic.invalid_arguments", common_tool_failure_class::validation, false, "A source location is required for this semantic query.", error);
    json response;
    const auto params = json{{"textDocument", {{"uri", location["uri"]}}}, {"position", location["range"]["start"]}};
    if (!session_->request(method, params, response, error)) return provider_error("tool.diagnostics.semantic.provider_unavailable", error);
    json entries = json::array();
    const auto result = response.value("result", json::array());
    const auto limit = std::min<size_t>(arguments.value("max_results", 128), 128);
    if (result.is_array()) for (const auto & item : result) {
        if (entries.size() >= limit || !item.is_object()) break;
        auto entry = item.contains("location") ? location_json(item["location"]) : location_json(item);
        if (item.contains("from")) entry["from"] = location_json(item["from"]);
        if (item.contains("to")) entry["to"] = location_json(item["to"]);
        if (item.contains("name")) entry["name"] = item["name"];
        entries.push_back(std::move(entry));
    }
    return common_tool_execution_result::success(json({{"backend", "clangd"}, {"semantic", true}, {"results", entries}, {"count", entries.size()}, {"truncated", entries.size() >= limit}}).dump());
}

common_tool_execution_result agent_clangd_diagnostics_provider::references(const std::string & arguments_json) {
    const auto arguments = json::parse(arguments_json, nullptr, false);
    if (!arguments.is_object() || !arguments.contains("symbol")) return common_tool_execution_result::failure("tool.diagnostics.references.invalid_arguments", common_tool_failure_class::validation, false, "Reference arguments are invalid.");
    return location_query(arguments, "textDocument/references");
}

common_tool_execution_result agent_clangd_diagnostics_provider::call_hierarchy(const std::string & arguments_json) {
    const auto arguments = json::parse(arguments_json, nullptr, false);
    if (!arguments.is_object() || !arguments.contains("symbol")) return common_tool_execution_result::failure("tool.diagnostics.call_hierarchy.invalid_arguments", common_tool_failure_class::validation, false, "Call hierarchy arguments are invalid.");
    json location;
    std::string error;
    if (!resolve_location(arguments, location, error)) return common_tool_execution_result::failure("tool.diagnostics.call_hierarchy.invalid_arguments", common_tool_failure_class::validation, false, "A source location is required for call hierarchy.", error);
    json prepared;
    if (!session_->request("textDocument/prepareCallHierarchy", {{"textDocument", {{"uri", location["uri"]}}}, {"position", location["range"]["start"]}}, prepared, error)) return provider_error("tool.diagnostics.call_hierarchy.provider_unavailable", error);
    const auto items = prepared.value("result", json::array());
    if (!items.is_array() || items.empty()) return common_tool_execution_result::success(json({{"backend", "clangd"}, {"semantic", true}, {"callers", json::array()}, {"callees", json::array()}, {"count", 0}}).dump());
    const auto item = items[0];
    const auto direction = arguments.value("direction", std::string("both"));
    json callers = json::array(), callees = json::array();
    const auto limit = std::min<size_t>(arguments.value("max_results", 128), 128);
    auto collect = [&](const char * method, json & output) {
        json response;
        if (!session_->request(method, {{"item", item}}, response, error)) return false;
        const auto result = response.value("result", json::array());
        if (!result.is_array()) return true;
        for (const auto & entry : result) {
            if (output.size() >= limit || !entry.is_object()) break;
            json normalized = entry;
            if (entry.contains("from")) normalized["from"] = location_json(entry["from"]);
            if (entry.contains("to")) normalized["to"] = location_json(entry["to"]);
            output.push_back(std::move(normalized));
        }
        return true;
    };
    if ((direction == "callers" || direction == "both") && !collect("callHierarchy/incomingCalls", callers)) return provider_error("tool.diagnostics.call_hierarchy.provider_unavailable", error);
    if ((direction == "callees" || direction == "both") && !collect("callHierarchy/outgoingCalls", callees)) return provider_error("tool.diagnostics.call_hierarchy.provider_unavailable", error);
    return common_tool_execution_result::success(json({{"backend", "clangd"}, {"semantic", true}, {"callers", callers}, {"callees", callees}, {"count", callers.size() + callees.size()}}).dump());
}
