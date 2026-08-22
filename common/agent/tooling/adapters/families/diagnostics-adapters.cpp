#include "agent/tooling/adapters/families/diagnostics-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>

namespace {

using json = common_adapter_json;

bool text_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    char buffer[1024];
    input.read(buffer, sizeof(buffer));
    return input.good() || input.eof()
        ? std::find(buffer, buffer + input.gcount(), '\0') == buffer + input.gcount()
        : false;
}

bool repository_path(const std::string & root, const std::string & relative,
        std::filesystem::path & output, std::string & error) {
    if (root.empty()) { error = "diagnostics requires a repository root"; return false; }
    std::error_code fs_error;
    const auto base = std::filesystem::weakly_canonical(root, fs_error);
    if (fs_error) { error = "repository root could not be resolved"; return false; }
    const auto requested = relative.empty() ? base : std::filesystem::weakly_canonical(base / relative, fs_error);
    if (fs_error) { error = "diagnostics path could not be resolved"; return false; }
    const auto base_text = base.generic_string();
    const auto requested_text = requested.generic_string();
    if (requested_text != base_text && requested_text.rfind(base_text + "/", 0) != 0) {
        error = "diagnostics path escapes the runtime root"; return false;
    }
    output = requested; return true;
}

std::string lower_copy(std::string value) {
    for (char & c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

std::string normalize_failure_message(std::string value) {
    value = std::regex_replace(value, std::regex(R"([A-Za-z]:[\\/][^ :]+)"), "<path>");
    value = std::regex_replace(value, std::regex(R"([^\s:]+\.(?:c|cc|cpp|cxx|h|hh|hpp):\d+(?::\d+)?)"), "<location>");
    value = std::regex_replace(value, std::regex(R"((expected\s+)\d+)"), "$1<value>");
    value = std::regex_replace(value, std::regex(R"(0x[0-9A-Fa-f]+|\b\d{4,}\b)"), "<number>");
    value = std::regex_replace(value, std::regex(R"(\s+)"), " ");
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

std::string classify_failure(const std::string & line) {
    const auto lower = lower_copy(line);
    if (lower.find("timeout") != std::string::npos || lower.find("timed out") != std::string::npos) return "timeout";
    if (lower.find("segmentation fault") != std::string::npos || lower.find("access violation") != std::string::npos || lower.find("crash") != std::string::npos) return "crash";
    if (lower.find("linker") != std::string::npos || lower.find("undefined reference") != std::string::npos || lower.find("unresolved external") != std::string::npos) return "link_failure";
    if (lower.find("compile") != std::string::npos || lower.find("compiler") != std::string::npos) return "compile_failure";
    if (lower.find("permission") != std::string::npos || lower.find("access denied") != std::string::npos) return "permission";
    if (lower.find("network") != std::string::npos || lower.find("connection") != std::string::npos) return "network";
    if (lower.find("assert") != std::string::npos || lower.find("expect") != std::string::npos) return "assertion_failure";
    return "test_failure";
}

common_tool_execution_result symbol_fallback(const common_native_tool_bindings & bindings, const json & args, bool references) {
    const auto symbol = args.value("symbol", std::string{});
    const int limit = args.value("max_results", references ? 128 : 64);
    if (bindings.repository_root.empty()) return common_adapter_execution_failure("tool.diagnostics.repository_unavailable", "diagnostics requires a repository root", "Diagnostics are unavailable without a host-owned repository.");
    if (symbol.empty() || symbol.size() > 256 || limit < 1 || limit > (references ? 128 : 64)) return common_adapter_validation_failure(references ? "tool.diagnostics.references.invalid_arguments" : "tool.diagnostics.symbol.invalid_arguments", "diagnostics symbol arguments are out of bounds");
    std::filesystem::path root; std::string error;
    const auto hint = args.value(references ? "definition_path" : "path_hint", std::string{});
    if (!repository_path(bindings.repository_root, hint, root, error)) return common_adapter_validation_failure("tool.diagnostics.invalid_path", std::move(error));
    if (!std::filesystem::is_regular_file(root) && !std::filesystem::is_directory(root)) return common_adapter_not_found_failure("tool.diagnostics.path_not_found", "diagnostics path was not found", "The diagnostics path was not found.");
    json matches = json::array();
    auto scan = [&](const std::filesystem::path & file) {
        std::error_code file_error;
        if (matches.size() >= static_cast<size_t>(limit) || !std::filesystem::is_regular_file(file) || file_error || std::filesystem::file_size(file, file_error) > 512 * 1024 || !text_file(file)) return;
        std::ifstream stream(file); std::string line;
        for (int number = 1; std::getline(stream, line) && matches.size() < static_cast<size_t>(limit); ++number) {
            for (size_t offset = line.find(symbol); offset != std::string::npos && matches.size() < static_cast<size_t>(limit); offset = line.find(symbol, offset + symbol.size())) {
                const auto relative = std::filesystem::relative(file, bindings.repository_root).generic_string();
                matches.push_back({{"path", relative}, {"line", number}, {"column", offset + 1}, {"symbol", symbol}, {"kind", references ? "reference" : ((line.find("class " + symbol) != std::string::npos || line.find("struct " + symbol) != std::string::npos || line.find("enum " + symbol) != std::string::npos) ? "definition" : "text_match")}});
            }
        }
    };
    if (std::filesystem::is_regular_file(root)) scan(root);
    else for (auto it = std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::skip_permission_denied); it != std::filesystem::recursive_directory_iterator() && matches.size() < static_cast<size_t>(limit); ++it) scan(it->path());
    return common_adapter_success_json({{"backend", "text-fallback"}, {"semantic", false}, {"symbol", symbol}, {references ? "references" : "definitions", matches}, {"count", matches.size()}, {"truncated", matches.size() >= static_cast<size_t>(limit)}});
}

} // namespace

bool common_try_register_diagnostics_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    const auto id = definition.executor_id;
    const bool is_diagnostics = id == "builtin.diagnostics.compile" || id == "builtin.diagnostics.symbol" || id == "builtin.diagnostics.references" || id == "builtin.diagnostics.call_hierarchy" || id == "builtin.diagnostics.test_failures" || id == "builtin.diagnostics.format" || id == "builtin.diagnostics.include_graph" || id == "builtin.diagnostics.native_crash";
    if (!is_diagnostics) return false;
    if (id == "builtin.diagnostics.compile") {
        installed = common_adapter_register_definition(definition, registry, [](const std::string & input) {
            std::string error; json args;
            if (!common_adapter_parse_object(input, args, error) || !args.contains("output") || !args["output"].is_string()) return common_adapter_validation_failure("tool.diagnostics.compile.invalid_output", "diagnostics.compile requires compiler output");
            json diagnostics = json::array(); const std::regex gcc(R"(^(.+):(\d+):(\d+):\s*(error|warning|note):\s*(.*)$)"); const std::regex msvc(R"(^(.+)\((\d+),(\d+)\):\s*(error|warning|note)\s*[^:]*:\s*(.*)$)");
            std::istringstream lines(args["output"].get<std::string>()); std::string line; while (std::getline(lines, line) && diagnostics.size() < 256) { std::smatch match; if (std::regex_match(line, match, gcc) || std::regex_match(line, match, msvc)) diagnostics.push_back({{"path", match[1].str()}, {"line", std::stoi(match[2].str())}, {"column", std::stoi(match[3].str())}, {"severity", match[4].str()}, {"message", match[5].str()}}); }
            return common_adapter_success_json({{"diagnostics", diagnostics}, {"count", diagnostics.size()}});
        }, error);
    } else if (id == "builtin.diagnostics.symbol" || id == "builtin.diagnostics.references") {
        const bool references = id == "builtin.diagnostics.references";
        installed = common_adapter_register_definition(definition, registry, [bindings, references](const std::string & input) {
            std::string error; json args;
            if (!common_adapter_parse_object(input, args, error) || !args.contains("symbol") || !args["symbol"].is_string()) return common_adapter_validation_failure(references ? "tool.diagnostics.references.invalid_arguments" : "tool.diagnostics.symbol.invalid_arguments", "diagnostics requires a symbol");
            if (references ? static_cast<bool>(bindings.diagnostics_references) : static_cast<bool>(bindings.diagnostics_symbol)) return references ? bindings.diagnostics_references(input) : bindings.diagnostics_symbol(input);
            return symbol_fallback(bindings, args, references);
        }, error);
    } else if (id == "builtin.diagnostics.call_hierarchy") {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json args;
            if (!common_adapter_parse_object(input, args, error) || !args.contains("symbol") || !args["symbol"].is_string()) return common_adapter_validation_failure("tool.diagnostics.call_hierarchy.invalid_arguments", "diagnostics.call_hierarchy requires a symbol");
            const auto direction = args.value("direction", std::string("both")); const auto depth = args.value("max_depth", 3); const auto limit = args.value("max_results", 128);
            if ((direction != "callers" && direction != "callees" && direction != "both") || depth < 1 || depth > 8 || limit < 1 || limit > 128) return common_adapter_validation_failure("tool.diagnostics.call_hierarchy.invalid_arguments", "diagnostics.call_hierarchy arguments are out of bounds");
            if (bindings.diagnostics_call_hierarchy) return bindings.diagnostics_call_hierarchy(input);
            return common_adapter_execution_failure("tool.diagnostics.call_hierarchy.unavailable", "diagnostics.call_hierarchy requires a semantic provider", "Call hierarchy is unavailable until a clangd or project-index provider is configured.");
        }, error);
    } else if (id == "builtin.diagnostics.test_failures") {
        installed = common_adapter_register_definition(definition, registry, [](const std::string & input) {
            std::string error; json args;
            if (!common_adapter_parse_object(input, args, error) || !args.contains("result") || !args["result"].is_string()) return common_adapter_validation_failure("tool.diagnostics.test_failures.invalid_result", "diagnostics.test_failures requires a bounded test result");
            json groups = json::array(); std::map<std::string, size_t> indices; std::istringstream lines(args["result"].get<std::string>()); std::string line;
            while (std::getline(lines, line)) { const auto lower = lower_copy(line); if (lower.find("failed") == std::string::npos && lower.find("failure") == std::string::npos && lower.find("error") == std::string::npos && lower.find("timeout") == std::string::npos && lower.find("assert") == std::string::npos) continue; const auto classification = classify_failure(line); const auto normalized = normalize_failure_message(line); const auto key = classification + "|" + normalized; const auto found = indices.find(key); if (found == indices.end()) { if (groups.size() >= 64) continue; indices.emplace(key, groups.size()); groups.push_back({{"classification", classification}, {"message", normalized}, {"count", 1}, {"examples", json::array({line.substr(0, 1024)})}}); } else { auto & group = groups[found->second]; group["count"] = group["count"].get<size_t>() + 1; if (group["examples"].size() < 3) group["examples"].push_back(line.substr(0, 1024)); } }
            return common_adapter_success_json({{"failure_groups", groups}, {"count", groups.size()}, {"backend", "bounded-text"}});
        }, error);
    } else if (id == "builtin.diagnostics.native_crash") {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            std::string error; json args;
            if (!common_adapter_parse_object(input, args, error) || !args.contains("executable") || !args["executable"].is_string() || !args.contains("dump") || !args["dump"].is_string())
                return common_adapter_validation_failure("tool.diagnostics.native_crash.invalid_arguments", "diagnostics.native_crash requires executable and dump resources");
            if (args["executable"].get<std::string>().empty() || args["executable"].get<std::string>().size() > 1024 || args["dump"].get<std::string>().empty() || args["dump"].get<std::string>().size() > 1024)
                return common_adapter_validation_failure("tool.diagnostics.native_crash.invalid_arguments", "diagnostics.native_crash resource references are out of bounds");
            if (bindings.diagnostics_native_crash) return bindings.diagnostics_native_crash(input);
            return common_adapter_execution_failure("tool.diagnostics.native_crash.backend_unavailable", "native crash diagnostics provider is not configured", "Native crash analysis is unavailable on this host.");
        }, error);
    } else if (id == "builtin.diagnostics.format") {
        installed = common_adapter_register_definition(definition, registry, [](const std::string & input) { std::string error; json args; if (!common_adapter_parse_object(input, args, error) || !args.contains("output") || !args["output"].is_string()) return common_adapter_validation_failure("tool.diagnostics.format.invalid_output", "diagnostics.format requires formatter output"); const auto text = args["output"].get<std::string>(); json files = json::array(); std::istringstream lines(text); std::string line; while (std::getline(lines, line) && files.size() < 256) if (line.find("would reformat") != std::string::npos || line.find("needs formatting") != std::string::npos) files.push_back(line); return common_adapter_success_json({{"formatted", files.empty()}, {"files", files}, {"raw_output", text.substr(0, 65536)}}); }, error);
    } else {
        installed = common_adapter_register_definition(definition, registry, [](const std::string & input) { std::string error; json args; if (!common_adapter_parse_object(input, args, error) || !args.contains("output") || !args["output"].is_string()) return common_adapter_validation_failure("tool.diagnostics.include_graph.invalid_output", "diagnostics.include_graph requires dependency output"); json nodes = json::array(), edges = json::array(); std::set<std::string> seen; std::istringstream lines(args["output"].get<std::string>()); std::string line; while (std::getline(lines, line) && edges.size() < 512) { const auto separator = line.find(" -> "); if (separator == std::string::npos) continue; const auto from = line.substr(0, separator), to = line.substr(separator + 4); if (from.empty() || to.empty()) continue; if (seen.insert(from).second) nodes.push_back(from); if (seen.insert(to).second) nodes.push_back(to); edges.push_back({{"from", from}, {"to", to}}); } return common_adapter_success_json({{"nodes", nodes}, {"edges", edges}, {"cycles", json::array()}}); }, error);
    }
    return installed;
}
