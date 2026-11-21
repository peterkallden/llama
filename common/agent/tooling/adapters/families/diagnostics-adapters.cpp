#include "agent/tooling/adapters/families/diagnostics-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <regex>
#include <set>
#include <sstream>

namespace {

using json = common_adapter_json;

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

} // namespace

bool common_try_register_diagnostics_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    const auto id = definition.executor_id;
    const bool is_diagnostics = id == "builtin.diagnostics.compile" || id == "builtin.diagnostics.test_failures" || id == "builtin.diagnostics.format" || id == "builtin.diagnostics.include_graph" || id == "builtin.diagnostics.native_crash";
    if (!is_diagnostics) return false;
    if (id == "builtin.diagnostics.compile") {
        installed = common_adapter_register_definition(definition, registry, [](const std::string & input) {
            std::string error; json args;
            if (!common_adapter_parse_object(input, args, error) || !args.contains("output") || !args["output"].is_string()) return common_adapter_validation_failure("tool.diagnostics.compile.invalid_output", "diagnostics.compile requires compiler output");
            json diagnostics = json::array(); const std::regex gcc(R"(^(.+):(\d+):(\d+):\s*(error|warning|note):\s*(.*)$)"); const std::regex msvc(R"(^(.+)\((\d+),(\d+)\):\s*(error|warning|note)\s*[^:]*:\s*(.*)$)");
            std::istringstream lines(args["output"].get<std::string>()); std::string line; while (std::getline(lines, line) && diagnostics.size() < 256) { std::smatch match; if (std::regex_match(line, match, gcc) || std::regex_match(line, match, msvc)) diagnostics.push_back({{"path", match[1].str()}, {"line", std::stoi(match[2].str())}, {"column", std::stoi(match[3].str())}, {"severity", match[4].str()}, {"message", match[5].str()}}); }
            return common_adapter_success_json({{"diagnostics", diagnostics}, {"count", diagnostics.size()}});
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
        installed = common_adapter_register_definition(definition, registry, [](const std::string & input) {
            std::string error; json args;
            if (!common_adapter_parse_object(input, args, error) || !args.contains("output") || !args["output"].is_string()) return common_adapter_validation_failure("tool.diagnostics.format.invalid_output", "diagnostics.format requires formatter output");
            const auto text = args["output"].get<std::string>();
            json files = json::array(); std::istringstream lines(text); std::string line;
            while (std::getline(lines, line) && files.size() < 256) {
                const auto lower = lower_copy(line);
                if (lower.find("would reformat") != std::string::npos || lower.find("needs formatting") != std::string::npos ||
                        lower.find("formatting differs") != std::string::npos || lower.find("clang-format-violations") != std::string::npos) files.push_back(line);
            }
            return common_adapter_success_json({{"formatted", files.empty()}, {"files", files}, {"raw_output", text.substr(0, 65536)}});
        }, error);
    } else {
        installed = common_adapter_register_definition(definition, registry, [](const std::string & input) {
            std::string error; json args;
            if (!common_adapter_parse_object(input, args, error) || !args.contains("output") || !args["output"].is_string()) return common_adapter_validation_failure("tool.diagnostics.include_graph.invalid_output", "diagnostics.include_graph requires clang -H trace or explicit parent -> child edges");
            json nodes = json::array(), edges = json::array();
            std::set<std::string> seen;
            std::vector<std::string> include_stack;
            std::istringstream lines(args["output"].get<std::string>()); std::string line;
            size_t parsed = 0;
            while (std::getline(lines, line) && edges.size() < 512) {
                const auto separator = line.find(" -> ");
                if (separator != std::string::npos) {
                    const auto from = line.substr(0, separator), to = line.substr(separator + 4);
                    if (from.empty() || to.empty()) continue;
                    if (seen.insert(from).second) nodes.push_back(from);
                    if (seen.insert(to).second) nodes.push_back(to);
                    edges.push_back({{"from", from}, {"to", to}});
                    ++parsed;
                    continue;
                }
                size_t depth = 0;
                while (depth < line.size() && line[depth] == '.') ++depth;
                if (depth == 0 || depth >= line.size() || line[depth] != ' ') continue;
                const auto header = line.substr(depth + 1);
                if (header.empty()) continue;
                if (seen.insert(header).second) nodes.push_back(header);
                if (depth > 1 && include_stack.size() >= depth - 1)
                    edges.push_back({{"from", include_stack[depth - 2]}, {"to", header}});
                if (include_stack.size() < depth) include_stack.resize(depth);
                include_stack[depth - 1] = header;
                ++parsed;
            }
            if (parsed == 0) return common_adapter_validation_failure("tool.diagnostics.include_graph.unrecognized_output", "No clang -H include trace or parent -> child edge was found.");
            return common_adapter_success_json({{"nodes", nodes}, {"edges", edges}, {"cycles", json::array()}});
        }, error);
    }
    return installed;
}
