#include "agent/tooling/adapters/families/code-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/visualization/visualization-contract.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

namespace {

using json = common_adapter_json;

bool nonnegative_integer(const json & value, uint64_t maximum, uint64_t & output) {
    if (!value.is_number_integer()) return false;
    if (value.is_number_unsigned()) {
        output = value.get<uint64_t>();
    } else {
        const auto signed_value = value.get<int64_t>();
        if (signed_value < 0) return false;
        output = static_cast<uint64_t>(signed_value);
    }
    return output <= maximum;
}

bool text_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    char buffer[1024];
    input.read(buffer, sizeof(buffer));
    return input.good() || input.eof()
        ? std::find(buffer, buffer + input.gcount(), '\0') == buffer + input.gcount()
        : false;
}

bool resolve_repository_path(const std::string & root, const std::string & relative,
        std::filesystem::path & output, std::string & error) {
    if (root.empty()) { error = "code analysis requires a host-owned repository"; return false; }
    std::error_code fs_error;
    const auto base = std::filesystem::weakly_canonical(root, fs_error);
    if (fs_error) { error = "repository root could not be resolved"; return false; }
    const auto requested = relative.empty() ? base : std::filesystem::weakly_canonical(base / relative, fs_error);
    if (fs_error) { error = "code path could not be resolved"; return false; }
    const auto relative_path = requested.lexically_relative(base);
    if (relative_path.empty() && requested != base) {
        error = "code path escapes the host-owned repository";
        return false;
    }
    if (!relative_path.empty() && (*relative_path.begin() == ".." || relative_path.is_absolute())) {
        error = "code path escapes the host-owned repository";
        return false;
    }
    output = requested;
    return true;
}

common_tool_execution_result symbols_fallback(
        const common_native_tool_bindings & bindings, const json & args, bool references, bool legacy) {
    const auto symbol = args.value("symbol", std::string{});
    const int limit = args.value("max_results", references ? 128 : 64);
    if (symbol.empty() || symbol.size() > 256 || limit < 1 || limit > (references ? 128 : 64))
        return common_adapter_validation_failure("tool.code.invalid_arguments", "code symbol arguments are out of bounds");
    std::filesystem::path root;
    std::string error;
    const auto hint = references
        ? args.value("definition_path", args.value("path_hint", std::string{}))
        : args.value("path_hint", std::string{});
    if (!resolve_repository_path(bindings.repository_root, hint, root, error))
        return common_adapter_validation_failure("tool.code.invalid_path", std::move(error));
    if (!std::filesystem::is_regular_file(root) && !std::filesystem::is_directory(root))
        return common_adapter_not_found_failure("tool.code.path_not_found", "code path was not found", "The selected code path was not found.");

    json matches = json::array();
    size_t files_scanned = 0;
    size_t bytes_scanned = 0;
    size_t files_skipped = 0;
    size_t entries_visited = 0;
    bool scan_limited = false;
    bool coverage_complete = true;
    std::error_code base_error;
    const auto canonical_base = std::filesystem::weakly_canonical(bindings.repository_root, base_error);
    if (base_error) return common_adapter_execution_failure("tool.code.repository_unavailable",
        "repository root changed during code analysis", "The authorized repository is unavailable.");
    auto scan = [&](const std::filesystem::path & file) {
        std::error_code file_error;
        if (matches.size() >= static_cast<size_t>(limit) || files_scanned >= 2048 || bytes_scanned >= 64 * 1024 * 1024) {
            scan_limited = true;
            return;
        }
        if (!std::filesystem::is_regular_file(file, file_error)) {
            if (file_error) {
                ++files_skipped;
                coverage_complete = false;
            }
            return;
        }
        file_error.clear();
        const auto canonical_file = std::filesystem::weakly_canonical(file, file_error);
        if (file_error) {
            ++files_skipped;
            coverage_complete = false;
            return;
        }
        const auto relative_file = canonical_file.lexically_relative(canonical_base);
        if ((relative_file.empty() && canonical_file != canonical_base) ||
                (!relative_file.empty() && (*relative_file.begin() == ".." || relative_file.is_absolute()))) {
            ++files_skipped;
            coverage_complete = false;
            return;
        }
        const auto size = std::filesystem::file_size(canonical_file, file_error);
        if (file_error || size > 512 * 1024 ||
                size > 64 * 1024 * 1024 - bytes_scanned || !text_file(canonical_file)) {
            if (!file_error && size <= 512 * 1024 && size > 64 * 1024 * 1024 - bytes_scanned) {
                scan_limited = true;
            }
            ++files_skipped;
            coverage_complete = false;
            return;
        }
        ++files_scanned;
        bytes_scanned += static_cast<size_t>(size);
        std::ifstream stream(canonical_file);
        std::string line;
        for (int number = 1; std::getline(stream, line) && matches.size() < static_cast<size_t>(limit); ++number) {
            for (size_t offset = line.find(symbol); offset != std::string::npos && matches.size() < static_cast<size_t>(limit);
                    offset = line.find(symbol, offset + symbol.size())) {
                matches.push_back({{"path", canonical_file.lexically_relative(
                        canonical_base).generic_string()},
                    {"line", number}, {"column", offset + 1}, {"symbol", symbol},
                    {"kind", legacy ? (references ? "reference" : "text_match") :
                        (references ? "text_reference" : "text_match")}});
            }
        }
        if (stream.bad()) {
            ++files_skipped;
            coverage_complete = false;
        }
    };
    std::error_code root_error;
    if (std::filesystem::is_regular_file(root, root_error)) scan(root);
    else if (!root_error && std::filesystem::is_directory(root, root_error)) {
        root_error.clear();
        std::error_code iterator_error;
        auto it = std::filesystem::recursive_directory_iterator(root,
            std::filesystem::directory_options::none, iterator_error);
        if (iterator_error) coverage_complete = false;
        const std::filesystem::recursive_directory_iterator end;
        for (; it != end && matches.size() < static_cast<size_t>(limit) &&
                files_scanned < 2048 && bytes_scanned < 64 * 1024 * 1024 &&
                entries_visited < 8192;) {
            ++entries_visited;
            scan(it->path());
            it.increment(iterator_error);
            if (iterator_error) {
                coverage_complete = false;
                iterator_error.clear();
            }
        }
        if (it != end && (files_scanned >= 2048 || bytes_scanned >= 64 * 1024 * 1024 ||
                entries_visited >= 8192)) {
            scan_limited = true;
            coverage_complete = false;
        }
    } else {
        return common_adapter_not_found_failure("tool.code.path_not_found",
            "code path disappeared during analysis", "The selected code path was not found.");
    }
    if (legacy) return common_adapter_success_json({{"backend", "text-fallback"}, {"semantic", false},
        {"symbol", symbol}, {references ? "references" : "definitions", matches},
        {"count", matches.size()}, {"truncated", matches.size() >= static_cast<size_t>(limit)}});
    json limitations = json::array({
        "Text matches do not resolve symbols or prove call relationships.",
        "Source locations refer to the host's current repository state; this response does not pin a commit or content snapshot."});
    if (files_skipped > 0) limitations.push_back("Some repository files were not inspected; see files_skipped.");
    if (scan_limited) limitations.push_back("The configured file or byte scan budget was reached.");
    return common_adapter_success_json({{"schema", "code-analysis.v1"},
        {"operation", references ? "references" : "symbols"},
        {"backend", "text-fallback"}, {"method", "text_match"}, {"semantic", false},
        {"symbol", symbol}, {"items", matches},
        {"count", matches.size()}, {"truncated", matches.size() >= static_cast<size_t>(limit)},
        {"complete", coverage_complete && !scan_limited && matches.size() < static_cast<size_t>(limit)},
        {"scan_limited", scan_limited}, {"files_scanned", files_scanned},
        {"files_skipped", files_skipped}, {"entries_visited", entries_visited}, {"bytes_scanned", bytes_scanned},
        {"limitations", limitations}});
}

std::string location_id(const json & location) {
    if (!location.is_object()) return {};
    if (!location.contains("path") || !location["path"].is_string() ||
            !location.contains("line") || !location["line"].is_number_integer() ||
            (location.contains("column") && !location["column"].is_number_integer())) return {};
    const auto path = location.value("path", std::string{});
    uint64_t line = 0;
    uint64_t column = 0;
    if (path.empty() || path.size() > 480 ||
            !nonnegative_integer(location["line"], 1000000, line) || line == 0 ||
            (location.contains("column") && !nonnegative_integer(location["column"], 1000000, column))) return {};
    return "function:" + path + ":" + std::to_string(line) + ":" + std::to_string(column);
}

json call_visualization(const json & output, bool truncated, size_t limit) {
    common_agent_visualization_relationship graph;
    graph.title = "Static call relationships";
    graph.complete = !truncated;
    graph.truncated = truncated;
    graph.limitations.push_back(
        "Source locations refer to the host's current repository state; this response does not pin a commit or content snapshot.");
    const auto add_limitation = [&](const std::string & value) {
        if (std::find(graph.limitations.begin(), graph.limitations.end(), value) == graph.limitations.end())
            graph.limitations.push_back(value);
    };
    if (!graph.complete) graph.limitations.push_back("Call relationships were limited by the configured result bound.");
    bool invalid_location = false;

    std::map<std::string, std::string> labels;
    json edges = json::array();
    const auto collect = [&](const char * field, const char * relation) {
        const auto values = output.value(field, json::array());
        if (!values.is_array()) return;
        for (const auto & value : values) {
            if (edges.size() >= limit) {
                graph.complete = false;
                graph.truncated = true;
                add_limitation("The relationship projection reached its edge limit.");
                break;
            }
            if (!value.is_object() || !value.contains("from") || !value.contains("to")) {
                invalid_location = true;
                continue;
            }
            const auto from_id = location_id(value["from"]);
            const auto to_id = location_id(value["to"]);
            if (from_id.empty() || to_id.empty()) {
                invalid_location = true;
                continue;
            }
            const auto remember = [&](const std::string & id, const json & location) {
                const auto path = location.value("path", std::string{});
                const auto line = location["line"].get<uint64_t>();
                const auto column = location.contains("column") ? location["column"].get<uint64_t>() : 0;
                labels.emplace(id, path + ":" + std::to_string(line) + ":" + std::to_string(column));
            };
            remember(from_id, value["from"]);
            remember(to_id, value["to"]);
            const auto from_path = value["from"].value("path", std::string{});
            const auto to_path = value["to"].value("path", std::string{});
            const auto from_line = value["from"].value("line", 0);
            const auto to_line = value["to"].value("line", 0);
            if (from_path.size() + 16 > 512 || to_path.size() + 16 > 512) {
                graph.complete = false;
                graph.truncated = true;
                add_limitation("Some source anchors exceeded visualization limits.");
                continue;
            }
            edges.push_back({{"id", "call:" + std::to_string(edges.size())},
                {"source", from_id}, {"target", to_id}, {"directed", true},
                {"relation", relation}, {"label", "calls"},
                {"provenance_refs", json::array({
                    from_path + "#L" + std::to_string(from_line),
                    to_path + "#L" + std::to_string(to_line)})}});
        }
    };
    collect("callers", "static_call");
    collect("callees", "static_call");
    if (invalid_location) {
        graph.complete = false;
        graph.truncated = true;
        add_limitation("Some provider call relationships lacked usable source locations.");
    }
    for (const auto & entry : labels) {
        if (entry.first.size() > 256 || entry.second.size() > 512) {
            graph.complete = false;
            graph.truncated = true;
            add_limitation("Some source locations exceeded visualization limits.");
            continue;
        }
        graph.nodes.push_back({entry.first, entry.second, "function"});
    }
    graph.edges.reserve(edges.size());
    for (const auto & edge : edges) {
        if (edge["source"].get<std::string>().size() > 256 ||
                edge["target"].get<std::string>().size() > 256) continue;
        graph.edges.push_back({edge["id"].get<std::string>(),
            edge["source"].get<std::string>(), edge["target"].get<std::string>(),
            edge["label"].get<std::string>(), true, edge["relation"].get<std::string>(),
            edge["provenance_refs"].get<std::vector<std::string>>()});
    }
    std::set<std::string> node_ids;
    for (const auto & node : graph.nodes) node_ids.insert(node.id);
    graph.edges.erase(std::remove_if(graph.edges.begin(), graph.edges.end(), [&](const auto & edge) {
        return node_ids.count(edge.source) == 0 || node_ids.count(edge.target) == 0;
    }), graph.edges.end());
    return common_agent_visualization_relationship_to_json(graph);
}

bool valid_optional_string(const json & value, const char * key, size_t maximum,
        std::string & error) {
    if (!value.contains(key)) return true;
    if (!value[key].is_string() || value[key].get_ref<const std::string &>().size() > maximum) {
        error = std::string(key) + " must be a bounded string";
        return false;
    }
    return true;
}

bool valid_optional_integer(const json & value, const char * key, int minimum,
        int maximum, std::string & error) {
    if (!value.contains(key)) return true;
    if (!value[key].is_number_integer()) {
        error = std::string(key) + " must be an integer";
        return false;
    }
    bool in_range = false;
    if (value[key].is_number_unsigned()) {
        const auto number = value[key].get<uint64_t>();
        in_range = number >= static_cast<uint64_t>(minimum) &&
            number <= static_cast<uint64_t>(maximum);
    } else {
        const auto number = value[key].get<int64_t>();
        in_range = number >= minimum && number <= maximum;
    }
    if (!in_range) {
        error = std::string(key) + " is outside its permitted range";
        return false;
    }
    return true;
}

bool has_executor(const std::string & id, const char * canonical, const char * legacy) {
    return id == canonical || id == legacy;
}

common_tool_execution_result canonical_provider_result(
        common_tool_execution_result result, bool legacy) {
    static const std::string legacy_prefix = "tool.diagnostics.";
    if (!legacy && !result.ok && result.failure_code.rfind(legacy_prefix, 0) == 0) {
        result.failure_code.replace(0, legacy_prefix.size(), "tool.code.");
    }
    return result;
}

} // namespace

bool common_try_register_code_tool_adapter(
        const common_tool_definition & definition,
        const common_native_tool_bindings & bindings,
        common_tool_registry & registry,
        bool & installed,
        std::string & error) {
    installed = false;
    const auto & id = definition.executor_id;
    const bool inspect = id == "builtin.code.inspect";
    const bool symbols = has_executor(id, "builtin.code.symbols", "builtin.diagnostics.symbol");
    const bool references = has_executor(id, "builtin.code.references", "builtin.diagnostics.references");
    const bool calls = has_executor(id, "builtin.code.calls", "builtin.diagnostics.call_hierarchy");
    if (!inspect && !symbols && !references && !calls) return false;

    if (inspect) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string &) {
            std::error_code root_error;
            const bool repository_available = !bindings.repository_root.empty() &&
                std::filesystem::is_directory(bindings.repository_root, root_error) && !root_error;
            const bool symbols_semantic = static_cast<bool>(bindings.code_symbols);
            const bool references_semantic = static_cast<bool>(bindings.code_references);
            const bool calls_semantic = static_cast<bool>(bindings.code_calls);
            const bool semantic_provider = repository_available &&
                (symbols_semantic || references_semantic || calls_semantic);
            return common_adapter_success_json({{"schema", "code-capabilities.v1"},
                {"repository_available", repository_available},
                {"backend", semantic_provider ? "semantic_provider" : "bounded_text"},
                {"capabilities", {{"symbols", repository_available ? (symbols_semantic ? "semantic" : "text_fallback") : "unavailable"},
                    {"references", repository_available ? (references_semantic ? "semantic" : "text_fallback") : "unavailable"},
                    {"calls", repository_available && calls_semantic ? "semantic" : "unavailable"}}},
                {"execution", "read_only"},
                {"limitations", json::array({
                    "Text fallback finds text occurrences only; it does not resolve code semantics.",
                    "Code analysis uses the host's current repository state; the current result contract does not pin a commit or content snapshot."})}});
        }, error);
    } else if (symbols || references) {
        const bool legacy = id.rfind("builtin.diagnostics.", 0) == 0;
        installed = common_adapter_register_definition(definition, registry, [bindings, references, legacy](const std::string & input) {
            std::string parse_error;
            json args;
            if (!common_adapter_parse_object(input, args, parse_error) || !args.contains("symbol") ||
                    !args["symbol"].is_string() || args["symbol"].get_ref<const std::string &>().empty() ||
                    args["symbol"].get_ref<const std::string &>().size() > 256)
                return common_adapter_validation_failure("tool.code.invalid_arguments", "code operation requires a symbol of at most 256 bytes");
            if (!valid_optional_string(args, references ? "definition_path" : "path_hint", 512, parse_error) ||
                    !valid_optional_string(args, references ? "path_hint" : "definition_path", 512, parse_error) ||
                    !valid_optional_integer(args, "definition_line", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "definition_column", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "line", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "column", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "max_results", 1, references ? 128 : 64, parse_error))
                return common_adapter_validation_failure("tool.code.invalid_arguments", std::move(parse_error));
            const auto & semantic = references
                ? (bindings.code_references ? bindings.code_references : (legacy ? bindings.diagnostics_references : bindings.code_references))
                : (bindings.code_symbols ? bindings.code_symbols : (legacy ? bindings.diagnostics_symbol : bindings.code_symbols));
            if (semantic) {
                auto result = semantic(input);
                if (!result.ok) return canonical_provider_result(std::move(result), legacy);
                auto output = json::parse(result.output, nullptr, false);
                const bool semantic_empty = legacy && output.is_object() &&
                    output.contains("backend") && output["backend"].is_string() &&
                    output["backend"].get_ref<const std::string &>() == "clangd" &&
                    output.contains("semantic") && output["semantic"].is_boolean() && output["semantic"].get<bool>() &&
                    output.contains("count") && output["count"].is_number_integer() &&
                    output["count"].is_number_unsigned() && output["count"].get<uint64_t>() == 0;
                if (semantic_empty)
                    return symbols_fallback(bindings, args, references, true);
                if (legacy) return result;
                if (!output.is_object())
                    return common_adapter_execution_failure("tool.code.invalid_provider_result",
                        "semantic provider returned an invalid result", "Code analysis returned an invalid provider response.");
                auto items = references
                    ? output.value("results", output.value("references", json::array()))
                    : output.value("definitions", json::array());
                if (!items.is_array() ||
                        (output.contains("backend") && !output["backend"].is_string()) ||
                        (output.contains("truncated") && !output["truncated"].is_boolean()) ||
                        (output.contains("count") && !output["count"].is_number_integer()))
                    return common_adapter_execution_failure("tool.code.invalid_provider_result",
                        "semantic provider result has invalid metadata or omitted its result array", "Code analysis returned an invalid provider response.");
                const bool truncated = output.value("truncated", false);
                uint64_t count = items.size();
                if (output.contains("count") && !nonnegative_integer(output["count"], items.size(), count))
                    return common_adapter_execution_failure("tool.code.invalid_provider_result",
                        "semantic provider result count did not match its result array", "Code analysis returned an invalid provider response.");
                if (count != items.size())
                    return common_adapter_execution_failure("tool.code.invalid_provider_result",
                        "semantic provider result count did not match its result array", "Code analysis returned an invalid provider response.");
                return common_adapter_success_json({{"schema", "code-analysis.v1"},
                    {"operation", references ? "references" : "symbols"},
                    {"backend", output.value("backend", std::string("semantic_provider"))},
                    {"method", "semantic"}, {"semantic", true}, {"symbol", args["symbol"]},
                    {"items", items}, {"count", count},
                    {"truncated", truncated}, {"complete", !truncated},
                    {"limitations", json::array({
                        "Source locations refer to the host's current repository state; this response does not pin a commit or content snapshot."})}});
            }
            return symbols_fallback(bindings, args, references, legacy);
        }, error);
    } else if (calls) {
        const bool legacy = id.rfind("builtin.diagnostics.", 0) == 0;
        installed = common_adapter_register_definition(definition, registry, [bindings, legacy](const std::string & input) {
            std::string parse_error;
            json args;
            if (!common_adapter_parse_object(input, args, parse_error) || !args.contains("symbol") ||
                    !args["symbol"].is_string() || args["symbol"].get_ref<const std::string &>().empty() ||
                    args["symbol"].get_ref<const std::string &>().size() > 256)
                return common_adapter_validation_failure("tool.code.calls.invalid_arguments", "code.calls requires a symbol of at most 256 bytes");
            const int maximum_results = legacy ? 128 : 16;
            if (!valid_optional_string(args, "path_hint", 512, parse_error) ||
                    !valid_optional_string(args, "definition_path", 512, parse_error) ||
                    !valid_optional_integer(args, "line", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "column", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "definition_line", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "definition_column", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "max_results", 1, maximum_results, parse_error) ||
                    (args.contains("direction") && !args["direction"].is_string()))
                return common_adapter_validation_failure("tool.code.calls.invalid_arguments",
                    parse_error.empty() ? "direction must be a string" : std::move(parse_error));
            const auto direction = args.value("direction", std::string("both"));
            const auto limit = args.value("max_results", maximum_results);
            if ((direction != "callers" && direction != "callees" && direction != "both") || limit < 1 || limit > maximum_results)
                return common_adapter_validation_failure("tool.code.calls.invalid_arguments", "code.calls arguments are out of bounds");
            const auto & semantic = bindings.code_calls ? bindings.code_calls : (legacy ? bindings.diagnostics_call_hierarchy : bindings.code_calls);
            if (!semantic) return common_adapter_execution_failure(legacy ? "tool.diagnostics.call_hierarchy.unavailable" : "tool.code.calls.unavailable",
                legacy ? "diagnostics.call_hierarchy requires a semantic provider" : "semantic call analysis is not configured",
                "Call relationships are unavailable until a semantic code provider is configured.");
            args["max_results"] = limit;
            auto result = semantic(args.dump());
            if (!result.ok) return canonical_provider_result(std::move(result), legacy);
            if (legacy) return result;
            auto output = json::parse(result.output, nullptr, false);
            if (!output.is_object() || !output.value("callers", json::array()).is_array() ||
                    !output.value("callees", json::array()).is_array() ||
                    !output.contains("count") || !output["count"].is_number_integer() ||
                    (output.contains("backend") && !output["backend"].is_string()) ||
                    (output.contains("truncated") && !output["truncated"].is_boolean()))
                return common_adapter_execution_failure("tool.code.invalid_provider_result",
                    "semantic provider returned an invalid call hierarchy", "Code analysis returned an invalid provider response.");
            const auto actual_count = output["callers"].size() + output["callees"].size();
            const bool count_matches = output["count"].is_number_unsigned()
                ? output["count"].get<uint64_t>() == actual_count
                : output["count"].get<int64_t>() >= 0 &&
                    static_cast<uint64_t>(output["count"].get<int64_t>()) == actual_count;
            if (!count_matches)
                return common_adapter_execution_failure("tool.code.invalid_provider_result",
                    "semantic provider result count did not match its call relationships", "Code analysis returned an invalid provider response.");
            const auto count = actual_count;
            const bool provider_truncated =
                ((direction == "callers" || direction == "both") && output["callers"].size() >= static_cast<size_t>(limit)) ||
                ((direction == "callees" || direction == "both") && output["callees"].size() >= static_cast<size_t>(limit));
            const auto graph = call_visualization(output, provider_truncated, static_cast<size_t>(limit));
            const bool truncated = provider_truncated || !graph.value("complete", true);
            auto limitations = json::array({
                "The provider returns one-hop call relationships, not a proof of feasible execution."});
            for (const auto & limitation : graph.value("limitations", json::array())) limitations.push_back(limitation);
            auto canonical_output = json{{"schema", "code-analysis.v1"},
                {"operation", "calls"}, {"backend", output.value("backend", std::string("semantic_provider"))},
                {"method", "semantic_call_hierarchy"}, {"semantic", true}, {"depth", 1},
                {"symbol", args["symbol"]}, {"count", count}, {"truncated", truncated},
                {"complete", !truncated},
                {"limitations", limitations},
                {"visualization", graph}};
            const auto serialized = canonical_output.dump();
            if (serialized.size() > 64 * 1024)
                return common_adapter_limit_failure("tool.code.calls.output_limit",
                    "call hierarchy exceeded the bounded response size", "Code analysis returned too much graph data; reduce max_results.");
            return common_adapter_success_json(canonical_output);
        }, error);
    }
    return installed;
}
