#include "agent/tooling/adapters/families/code-adapters.h"

#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/visualization/visualization-contract.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>

namespace {

using json = common_adapter_json;

void add_limitation(std::vector<std::string> & limitations, const std::string & value) {
    if (std::find(limitations.begin(), limitations.end(), value) != limitations.end()) return;
    if (limitations.size() < 31) limitations.push_back(value);
    else if (limitations.size() == 31) limitations.push_back("Additional analysis limitations were omitted.");
}

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

bool provider_coverage(const json & output, bool & complete, bool & truncated, json & limitations) {
    if ((output.contains("backend") && !output["backend"].is_string()) ||
            (output.contains("semantic") && (!output["semantic"].is_boolean() || !output["semantic"].get<bool>())) ||
            (output.contains("complete") && !output["complete"].is_boolean()) ||
            (output.contains("truncated") && !output["truncated"].is_boolean())) return false;
    truncated = output.value("truncated", false);
    complete = output.value("complete", !truncated);
    if (complete && truncated) return false;
    limitations = output.value("limitations", json::array());
    if (!limitations.is_array() || limitations.size() > 24) return false;
    for (const auto & value : limitations)
        if (!value.is_string() || value.get_ref<const std::string &>().empty() ||
                value.get_ref<const std::string &>().size() > 512 ||
                value.get_ref<const std::string &>().find('\0') != std::string::npos) return false;
    if ((!complete || truncated) && limitations.empty()) limitations.push_back("The semantic provider reported incomplete coverage.");
    return true;
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
    std::error_code initial_error;
    const auto root_status = std::filesystem::status(root, initial_error);
    if (initial_error || (!std::filesystem::is_regular_file(root_status) && !std::filesystem::is_directory(root_status)))
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
        if (!stream) { ++files_skipped; coverage_complete = false; return; }
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
    if (scan_limited) limitations.push_back("The configured traversal, file or byte scan budget was reached.");
    return common_adapter_success_json({{"schema", "code-analysis.v1"},
        {"operation", references ? "references" : "symbols"},
        {"backend", "text-fallback"}, {"method", "text_match"}, {"semantic", false},
        {"column_encoding", "utf-8-bytes"},
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
    if (path.empty() || path.size() > 480 || path.find('\0') != std::string::npos ||
            !nonnegative_integer(location["line"], 1000000, line) || line == 0 ||
            (location.contains("column") && !nonnegative_integer(location["column"], 1000000, column))) return {};
    return "function:" + path + ":" + std::to_string(line) + ":" + std::to_string(column);
}

json call_visualization(const json & output, bool complete, bool truncated, size_t limit) {
    common_agent_visualization_relationship graph;
    graph.title = "Static call relationships";
    graph.complete = complete;
    graph.truncated = truncated;
    graph.limitations.push_back(
        "Source locations refer to the host's current repository state; this response does not pin a commit or content snapshot.");
    const auto add_graph_limitation = [&](const std::string & value) {
        add_limitation(graph.limitations, value);
    };
    graph.limitations.push_back("Static call relationships do not establish feasible paths or observed execution.");
    if (truncated) graph.limitations.push_back("Call relationships were limited by the configured result bound.");
    for (const auto & value : output.value("limitations", json::array())) add_graph_limitation(value.get<std::string>());
    bool invalid_location = false;

    std::map<std::string, std::string> labels;
    json edges = json::array();
    std::map<std::pair<std::string, std::string>, size_t> relationships;
    const auto collect = [&](const char * field, const char * relation) {
        const auto values = output.value(field, json::array());
        if (!values.is_array()) return;
        for (const auto & value : values) {
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
            const auto existing = relationships.find({from_id, to_id});
            if (existing == relationships.end() && edges.size() >= limit) {
                graph.complete = false; graph.truncated = true;
                add_graph_limitation("The relationship projection reached its edge limit.");
                continue;
            }
            const auto remember = [&](const std::string & id, const json & location) {
                const auto path = location.value("path", std::string{});
                const auto line = location["line"].get<uint64_t>();
                const auto column = location.contains("column") ? location["column"].get<uint64_t>() : 0;
                auto label = path + ":" + std::to_string(line) + ":" + std::to_string(column);
                if (location.contains("name") && location["name"].is_string() &&
                        location["name"].get_ref<const std::string &>().size() + label.size() + 3 <= 512)
                    label = location["name"].get<std::string>() + " @ " + label;
                labels.emplace(id, std::move(label));
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
                add_graph_limitation("Some source anchors exceeded visualization limits.");
                continue;
            }
            json refs = json::array({from_path + "#L" + std::to_string(from_line),
                to_path + "#L" + std::to_string(to_line)});
            if (value.contains("call_sites")) {
                if (!value["call_sites"].is_array()) invalid_location = true;
                else for (const auto & site : value["call_sites"]) {
                    if (location_id(site).empty() || site["path"] != from_path) { invalid_location = true; continue; }
                    uint64_t end_line = 0, end_column = 0;
                    const auto line = site["line"].get<uint64_t>();
                    const auto column = site.value("column", uint64_t{0});
                    if (!site.contains("end_line") || !site.contains("end_column") || column == 0 ||
                            !nonnegative_integer(site["end_line"], 1000000, end_line) ||
                            !nonnegative_integer(site["end_column"], 1000000, end_column) || end_column == 0 ||
                            end_line < line || (end_line == line && end_column < column)) {
                        invalid_location = true; continue;
                    }
                    const auto ref = from_path + "#L" + std::to_string(line) + "C" + std::to_string(column) +
                        "-L" + std::to_string(end_line) + "C" + std::to_string(end_column);
                    if (ref.size() > 512) { graph.complete = false; graph.truncated = true;
                        add_graph_limitation("Some call-site references exceeded the visualization limit."); continue; }
                    if (std::find(refs.begin(), refs.end(), ref) != refs.end()) continue;
                    if (refs.size() >= 64) { graph.complete = false; graph.truncated = true;
                        add_graph_limitation("Some call-site references exceeded the visualization limit."); break; }
                    refs.push_back(ref);
                }
            }
            if (existing != relationships.end()) {
                auto & retained = edges[existing->second]["provenance_refs"];
                for (const auto & ref : refs) {
                    if (std::find(retained.begin(), retained.end(), ref) != retained.end()) continue;
                    if (retained.size() >= 64) { graph.complete = false; graph.truncated = true;
                        add_graph_limitation("Some call-site references exceeded the visualization limit."); break; }
                    retained.push_back(ref);
                }
                continue;
            }
            relationships.emplace(std::make_pair(from_id, to_id), edges.size());
            edges.push_back({{"id", "call:" + std::to_string(edges.size())},
                {"source", from_id}, {"target", to_id}, {"directed", true},
                {"relation", relation}, {"label", "calls"}, {"provenance_refs", refs}});
        }
    };
    collect("callers", "static_call");
    collect("callees", "static_call");
    if (invalid_location) {
        graph.complete = false;
        add_graph_limitation("Some provider call relationships lacked usable source locations.");
    }
    for (const auto & entry : labels) {
        if (entry.first.size() > 256 || entry.second.size() > 512) {
            graph.complete = false;
            graph.truncated = true;
            add_graph_limitation("Some source locations exceeded visualization limits.");
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

struct call_neighborhood_result {
    json relationships;
    bool complete = false;
    bool truncated = false;
    std::optional<common_tool_execution_result> failure;
};

call_neighborhood_result call_neighborhood(const common_native_tool_bindings & bindings, const json & args,
        const std::string & direction, size_t max_depth, size_t max_nodes, size_t max_edges,
        uint32_t timeout_ms) {
    struct pending_location {
        json location;
        std::string symbol;
        size_t depth = 0;
        bool provider_location = false;
    };
    std::deque<pending_location> pending;
    const bool has_definition = args.contains("definition_path") && args.contains("definition_line");
    const auto path_key = has_definition ? "definition_path" : "path_hint";
    const auto line_key = has_definition ? "definition_line" : "line";
    const auto column_key = has_definition ? "definition_column" : "column";
    json seed = {{"path", args.value(path_key, std::string{})}, {"line", args.value(line_key, 0)}};
    if (args.contains(column_key)) seed["column"] = args[column_key];
    pending.push_back({seed, args.value("symbol", std::string{}), 0});

    json relationships = {{"callers", json::array()}, {"callees", json::array()}};
    json continuations = json::array(), failures = json::array();
    std::set<std::string> continuation_ids;
    std::set<std::string> discovered_nodes, queued_locations, expanded_locations;
    std::set<std::pair<std::string, std::string>> edges;
    std::vector<std::string> limitations;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    size_t requests = 0;
    bool complete = true, truncated = false;
    const auto stop_at_bound = [&](const std::string & reason) {
        complete = false;
        truncated = true;
        add_limitation(limitations, reason);
    };
    const auto add_continuation = [&](const pending_location & location, const char * reason) {
        const auto id = location_id(location.location);
        if (id.empty() || location.symbol.empty() || location.symbol.size() > 256 ||
                location.symbol.find('\0') != std::string::npos ||
                (location.location.contains("column") && location.location["column"] == 0) ||
                continuation_ids.count(id) || continuations.size() >= max_nodes) return;
        json arguments = {{"symbol", location.symbol}, {"definition_path", location.location["path"]},
            {"definition_line", location.location["line"]}, {"direction", direction}};
        if (location.location.contains("column")) arguments["definition_column"] = location.location["column"];
        continuations.push_back({{"reason", reason}, {"arguments", std::move(arguments)}});
        continuation_ids.insert(id);
    };
    pending_location active;
    const auto invalid_page = [&](const std::string & reason) {
        if (edges.empty()) return call_neighborhood_result{nullptr, false, false,
            common_adapter_execution_failure("tool.code.impact.provider_failed", reason,
                "The semantic provider returned an invalid bounded code analysis.")};
        add_limitation(limitations, "A provider response was invalid after partial results were collected.");
        add_continuation(active, "invalid_provider_response");
        for (const auto & remaining : pending) add_continuation(remaining, "unexpanded");
        failures.push_back({{"code", "tool.code.impact.provider_failed"}, {"class", "execution"},
            {"retryable", false}, {"summary", "The semantic provider returned an invalid bounded code analysis."},
            {"symbol", active.symbol}, {"path", active.location["path"]}, {"line", active.location["line"]}});
        relationships["limitations"] = limitations;
        relationships["provider_requests"] = requests;
        relationships["continuations"] = continuations;
        relationships["provider_failures"] = failures;
        return call_neighborhood_result{relationships, false, truncated, std::nullopt};
    };

    while (!pending.empty()) {
        auto current = std::move(pending.front());
        pending.pop_front();
        active = current;
        const auto current_id = location_id(current.location);
        if (expanded_locations.count(current_id)) continue;
        if (current.depth >= max_depth) {
            stop_at_bound("The requested call-neighborhood depth limit was reached; boundary nodes were not expanded.");
            add_continuation(current, "depth_limit");
            continue;
        }
        if (requests >= max_nodes || std::chrono::steady_clock::now() >= deadline) {
            stop_at_bound("The call-neighborhood provider-request or elapsed-time budget was reached.");
            add_continuation(current, requests >= max_nodes ? "request_limit" : "time_limit");
            for (const auto & remaining : pending) add_continuation(remaining, "budget_limit");
            break;
        }
        expanded_locations.insert(current_id);
        json request = {{"symbol", current.symbol}, {"direction", direction}, {"max_results", 16}};
        // Provider-returned locations are definitions, even when the initial selection was a hint.
        request["definition_path"] = current.location["path"];
        request["definition_line"] = current.location["line"];
        if (current.location.contains("column")) request["definition_column"] = current.location["column"];
        ++requests;
        auto result = bindings.code_calls(request.dump());
        if (!result.ok) {
            if (edges.empty()) return {nullptr, false, false, std::move(result)};
            complete = false;
            add_limitation(limitations, "A provider request failed after partial results were collected.");
            add_continuation(current, "provider_failure");
            if (failures.size() < max_nodes) failures.push_back({
                {"code", result.failure_code.substr(0, 256)},
                {"class", common_tool_failure_class_name(result.failure_class)},
                {"retryable", result.retryable},
                {"summary", result.safe_summary.substr(0, 512)},
                {"symbol", current.symbol}, {"path", current.location["path"]}, {"line", current.location["line"]}});
            continue;
        }
        if (result.output.size() > 64 * 1024) {
            return invalid_page("semantic call provider exceeded its neighborhood page byte limit");
        }
        auto output = json::parse(result.output, nullptr, false);
        if (!output.is_object() || !output.contains("callers") || !output["callers"].is_array() ||
                !output.contains("callees") || !output["callees"].is_array() ||
                output["callers"].size() + output["callees"].size() > 16 ||
                !output.contains("count")) {
            return invalid_page("semantic call provider returned an invalid neighborhood page");
        }
        const auto result_count = output["callers"].size() + output["callees"].size();
        uint64_t reported_count = 0;
        if (!nonnegative_integer(output["count"], result_count, reported_count) || reported_count != result_count) {
            return invalid_page("semantic call provider returned an inconsistent relationship count");
        }
        bool page_complete = true, page_truncated = false;
        json page_limitations;
        if (!provider_coverage(output, page_complete, page_truncated, page_limitations)) {
            return invalid_page("semantic call provider returned invalid coverage metadata");
        }
        // Older providers omit coverage flags. A full page cannot then prove exhaustion.
        if (!output.contains("truncated") && result_count == 16) page_truncated = true;
        complete = complete && page_complete && !page_truncated;
        truncated = truncated || page_truncated;
        for (const auto & limitation : page_limitations) add_limitation(limitations, limitation.get<std::string>());
        std::string selected_id;
        for (const auto * key : {"callers", "callees"}) {
            for (const auto & relation : output[key]) {
                if ((direction == "callers" && std::string(key) == "callees") ||
                        (direction == "callees" && std::string(key) == "callers")) {
                    return invalid_page("semantic call provider returned an unrequested call direction");
                }
                if (!relation.is_object() || !relation.contains("from") || !relation.contains("to")) {
                    complete = false;
                    continue;
                }
                const auto from_id = location_id(relation["from"]);
                const auto to_id = location_id(relation["to"]);
                if (from_id.empty() || to_id.empty()) { complete = false; continue; }
                const bool incoming = std::string(key) == "callers";
                const auto & focus_id = incoming ? to_id : from_id;
                if ((current.provider_location && focus_id != current_id) ||
                        (!selected_id.empty() && selected_id != focus_id)) {
                    return invalid_page("semantic call provider returned inconsistent selected symbols");
                }
                selected_id = focus_id;
                // The provider may resolve an unspecified column to its canonical definition.
                expanded_locations.insert(focus_id);
                const auto edge_key = std::make_pair(from_id, to_id);
                const size_t new_nodes = static_cast<size_t>(discovered_nodes.count(from_id) == 0) +
                    static_cast<size_t>(from_id != to_id && discovered_nodes.count(to_id) == 0);
                if ((edges.count(edge_key) == 0 && edges.size() >= max_edges) ||
                        discovered_nodes.size() + new_nodes > max_nodes) {
                    stop_at_bound("The call-neighborhood node or edge limit was reached.");
                    const auto & next = relation[incoming ? "from" : "to"];
                    if (next.contains("name") && next["name"].is_string())
                        add_continuation({next, next["name"].get<std::string>(), current.depth + 1, true},
                            edges.size() >= max_edges ? "edge_limit" : "node_limit");
                    continue;
                }
                discovered_nodes.insert(from_id);
                discovered_nodes.insert(to_id);
                edges.insert(edge_key);
                // Preserve repeated observations: the shared projection merges their call-site provenance.
                relationships[key].push_back(relation);
                const auto & next = relation[incoming ? "from" : "to"];
                const auto & next_id = incoming ? from_id : to_id;
                if (expanded_locations.count(next_id) || queued_locations.count(next_id)) continue;
                if (!next.contains("name") || !next["name"].is_string() ||
                        next["name"].get_ref<const std::string &>().empty() ||
                        next["name"].get_ref<const std::string &>().size() > 256 ||
                        next["name"].get_ref<const std::string &>().find('\0') != std::string::npos) {
                    complete = false;
                    add_limitation(limitations, "A relationship did not provide a usable symbol name for expansion.");
                    continue;
                }
                queued_locations.insert(next_id);
                pending.push_back({next, next["name"].get<std::string>(), current.depth + 1, true});
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            stop_at_bound("The call-neighborhood elapsed-time budget was reached.");
            for (const auto & remaining : pending) add_continuation(remaining, "time_limit");
            break;
        }
    }
    if (!complete && limitations.empty())
        add_limitation(limitations, "The provider or bounded traversal returned incomplete coverage.");
    relationships["limitations"] = std::move(limitations);
    relationships["provider_requests"] = requests;
    relationships["continuations"] = std::move(continuations);
    relationships["provider_failures"] = std::move(failures);
    return {std::move(relationships), complete, truncated, std::nullopt};
}

bool valid_optional_string(const json & value, const char * key, size_t maximum,
        std::string & error) {
    if (!value.contains(key)) return true;
    if (!value[key].is_string() || value[key].get_ref<const std::string &>().size() > maximum ||
            value[key].get_ref<const std::string &>().find('\0') != std::string::npos) {
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
    const bool impact = id == "builtin.code.impact";
    if (!inspect && !symbols && !references && !calls && !impact) return false;

    if (inspect) {
        installed = common_adapter_register_definition(definition, registry, [bindings](const std::string & input) {
            json args;
            std::string parse_error;
            if (!common_adapter_parse_object(input, args, parse_error) ||
                    !valid_optional_string(args, "path", 512, parse_error) ||
                    (args.contains("path") && args["path"].is_string() && args["path"].get_ref<const std::string &>().empty()) ||
                    args.size() > (args.contains("path") ? 1u : 0u))
                return common_adapter_validation_failure("tool.code.invalid_arguments", "code.inspect accepts an optional repository-relative path");
            std::error_code root_error;
            const bool repository_available = !bindings.repository_root.empty() &&
                std::filesystem::is_directory(bindings.repository_root, root_error) && !root_error;
            const bool symbols_semantic = static_cast<bool>(bindings.code_symbols);
            const bool references_semantic = static_cast<bool>(bindings.code_references);
            const bool calls_semantic = static_cast<bool>(bindings.code_calls);
            const bool semantic_provider = repository_available &&
                (symbols_semantic || references_semantic || calls_semantic);
            std::filesystem::path requested;
            if (args.contains("path") &&
                    !resolve_repository_path(bindings.repository_root, args["path"].get<std::string>(), requested, parse_error))
                return common_adapter_validation_failure("tool.code.invalid_arguments", std::move(parse_error));
            std::string database_status = "unknown";
            json source_in_database = nullptr;
            if (!bindings.code_compile_commands.empty() && bindings.code_compile_commands != "auto") {
                std::error_code db_error;
                auto db = std::filesystem::absolute(bindings.code_compile_commands, db_error);
                if (!db_error && std::filesystem::is_directory(db, db_error)) db /= "compile_commands.json";
                const auto size = !db_error ? std::filesystem::file_size(db, db_error) : 0;
                if (db_error || size > 16 * 1024 * 1024) database_status = "unavailable";
                else {
                    std::ifstream file(db, std::ios::binary);
                    std::string contents(16 * 1024 * 1024 + 1, '\0');
                    if (file) file.read(&contents[0], contents.size());
                    contents.resize(static_cast<size_t>(file.gcount()));
                    const auto entries = !file.bad() && contents.size() <= 16 * 1024 * 1024
                        ? json::parse(contents, nullptr, false) : json();
                    if (!entries.is_array()) database_status = "unavailable";
                    else {
                        database_status = "available";
                        if (!requested.empty()) {
                            bool found = false, invalid = false;
                            for (const auto & entry : entries) {
                                if (!entry.is_object() || !entry.contains("directory") || !entry["directory"].is_string() ||
                                        !entry.contains("file") || !entry["file"].is_string()) { invalid = true; continue; }
                                std::error_code entry_error;
                                const auto candidate = std::filesystem::weakly_canonical(
                                    std::filesystem::path(entry["directory"].get<std::string>()) /
                                    std::filesystem::path(entry["file"].get<std::string>()), entry_error);
                                if (entry_error) { invalid = true; continue; }
                                if (candidate == requested) { found = true; break; }
                            }
                            if (found || !invalid) source_in_database = found;
                        }
                    }
                }
            }
            return common_adapter_success_json({{"schema", "code-capabilities.v1"},
                {"repository_available", repository_available},
                {"backend", semantic_provider ? "semantic_provider" : "bounded_text"},
                {"compile_commands", bindings.code_compile_commands.empty() ? json(nullptr) : json(bindings.code_compile_commands)},
                {"compilation_database_status", database_status}, {"source_in_database", source_in_database},
                {"index_coverage", semantic_provider ? "unknown" : "unavailable"},
                {"capabilities", {{"symbols", repository_available ? (symbols_semantic ? "semantic" : "text_fallback") : "unavailable"},
                    {"references", repository_available ? (references_semantic ? "semantic" : "text_fallback") : "unavailable"},
                    {"calls", repository_available && calls_semantic ? "semantic" : "unavailable"},
                    {"impact", repository_available && calls_semantic ? "bounded_semantic" : "unavailable"}}},
                {"execution", "read_only"},
                {"limitations", json::array({
                    "Text fallback finds text occurrences only; it does not resolve code semantics.",
                    "Semantic provider capability does not establish project-wide index coverage or readiness.",
                    "Code analysis uses the host's current repository state; the current result contract does not pin a commit or content snapshot."})}});
        }, error);
    } else if (symbols || references) {
        const bool legacy = id.rfind("builtin.diagnostics.", 0) == 0;
        installed = common_adapter_register_definition(definition, registry, [bindings, references, legacy](const std::string & input) {
            std::string parse_error;
            json args;
            if (!common_adapter_parse_object(input, args, parse_error) || !args.contains("symbol") ||
                    !args["symbol"].is_string() || args["symbol"].get_ref<const std::string &>().empty() ||
                    args["symbol"].get_ref<const std::string &>().size() > 256 ||
                    args["symbol"].get_ref<const std::string &>().find('\0') != std::string::npos)
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
                ? (legacy ? bindings.diagnostics_references : bindings.code_references)
                : (legacy ? bindings.diagnostics_symbol : bindings.code_symbols);
            if (semantic) {
                auto result = semantic(input);
                if (!result.ok) return canonical_provider_result(std::move(result), legacy);
                auto output = json::parse(result.output, nullptr, false);
                bool provider_complete = true, provider_truncated = false;
                json provider_limitations = json::array();
                if (!output.is_object() || !provider_coverage(output, provider_complete, provider_truncated, provider_limitations))
                    return common_adapter_execution_failure("tool.code.invalid_provider_result",
                        "semantic provider result has invalid coverage metadata", "Code analysis returned an invalid provider response.");
                const bool semantic_empty = legacy && output.is_object() &&
                    output.contains("backend") && output["backend"].is_string() &&
                    output["backend"].get_ref<const std::string &>() == "clangd" &&
                    output.contains("semantic") && output["semantic"].is_boolean() && output["semantic"].get<bool>() &&
                    output.contains("count") && output["count"].is_number_integer() &&
                    output["count"] == 0 && output.value("complete", true) && !output.value("truncated", false);
                if (semantic_empty)
                    return symbols_fallback(bindings, args, references, true);
                if (legacy) return result;
                if (!output.is_object())
                    return common_adapter_execution_failure("tool.code.invalid_provider_result",
                        "semantic provider returned an invalid result", "Code analysis returned an invalid provider response.");
                const auto items_key = references ? (output.contains("results") ? "results" : "references") : "definitions";
                if (!output.contains(items_key)) return common_adapter_execution_failure("tool.code.invalid_provider_result",
                    "semantic provider omitted its result array", "Code analysis returned an invalid provider response.");
                auto items = output[items_key];
                if (!items.is_array() ||
                        (output.contains("backend") && !output["backend"].is_string()) ||
                        (output.contains("truncated") && !output["truncated"].is_boolean()) ||
                        (output.contains("count") && !output["count"].is_number_integer()))
                    return common_adapter_execution_failure("tool.code.invalid_provider_result",
                        "semantic provider result has invalid metadata or omitted its result array", "Code analysis returned an invalid provider response.");
                const auto limit = static_cast<size_t>(args.value("max_results", references ? 128 : 64));
                if (items.size() > limit) {
                    while (items.size() > limit) items.erase(items.end() - 1);
                    provider_complete = false; provider_truncated = true;
                    provider_limitations.push_back("The adapter limited the provider result to the requested result count.");
                }
                const auto provider_item_count = output[items_key].size();
                uint64_t count = provider_item_count;
                if (output.contains("count") && !nonnegative_integer(output["count"], provider_item_count, count))
                    return common_adapter_execution_failure("tool.code.invalid_provider_result",
                        "semantic provider result count did not match its result array", "Code analysis returned an invalid provider response.");
                if (count != provider_item_count)
                    return common_adapter_execution_failure("tool.code.invalid_provider_result",
                        "semantic provider result count did not match its result array", "Code analysis returned an invalid provider response.");
                provider_limitations.push_back("Source locations refer to the host's current repository state; this response does not pin a commit or content snapshot.");
                return common_adapter_success_json({{"schema", "code-analysis.v1"},
                    {"operation", references ? "references" : "symbols"},
                    {"backend", output.value("backend", std::string("semantic_provider"))},
                    {"method", "semantic"}, {"semantic", true}, {"column_encoding", "utf-16"}, {"symbol", args["symbol"]},
                    {"items", items}, {"count", items.size()},
                    {"truncated", provider_truncated}, {"complete", provider_complete},
                    {"limitations", provider_limitations}});
            }
            return symbols_fallback(bindings, args, references, legacy);
        }, error);
    } else if (impact) {
        installed = common_adapter_register_definition(definition, registry,
                [bindings, timeout_ms = definition.timeout_ms](const std::string & input) {
            std::string parse_error;
            json args;
            if (!common_adapter_parse_object(input, args, parse_error) || !args.contains("symbol") ||
                    !args["symbol"].is_string() || args["symbol"].get_ref<const std::string &>().empty() ||
                    args["symbol"].get_ref<const std::string &>().size() > 256 ||
                    args["symbol"].get_ref<const std::string &>().find('\0') != std::string::npos)
                return common_adapter_validation_failure("tool.code.impact.invalid_arguments", "code.impact requires a symbol of at most 256 bytes");
            if (!valid_optional_string(args, "path_hint", 512, parse_error) ||
                    !valid_optional_string(args, "definition_path", 512, parse_error) ||
                    !valid_optional_integer(args, "line", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "column", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "definition_line", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "definition_column", 1, 1000000, parse_error) ||
                    !valid_optional_integer(args, "max_depth", 1, 3, parse_error) ||
                    !valid_optional_integer(args, "max_nodes", 2, 64, parse_error) ||
                    !valid_optional_integer(args, "max_edges", 1, 128, parse_error) ||
                    (args.contains("direction") && !args["direction"].is_string()))
                return common_adapter_validation_failure("tool.code.impact.invalid_arguments",
                    parse_error.empty() ? "direction must be a string" : std::move(parse_error));
            const bool has_hint = args.contains("path_hint") && args.contains("line");
            const bool has_definition = args.contains("definition_path") && args.contains("definition_line");
            const bool any_hint = args.contains("path_hint") || args.contains("line") || args.contains("column");
            const bool any_definition = args.contains("definition_path") || args.contains("definition_line") || args.contains("definition_column");
            if (has_hint == has_definition || (any_hint && !has_hint) ||
                    (any_definition && !has_definition) ||
                    (has_hint && args["path_hint"].get_ref<const std::string &>().empty()) ||
                    (has_definition && args["definition_path"].get_ref<const std::string &>().empty()))
                return common_adapter_validation_failure("tool.code.impact.invalid_arguments", "code.impact requires exactly one explicit source location");
            const auto direction = args.value("direction", std::string("both"));
            if (direction != "callers" && direction != "callees" && direction != "both")
                return common_adapter_validation_failure("tool.code.impact.invalid_arguments", "direction is outside its permitted values");
            if (!bindings.code_calls)
                return common_adapter_execution_failure("tool.code.impact.unavailable",
                    "semantic call analysis is not configured", "Bounded call-neighborhood analysis requires a semantic code provider.");

            auto analysis = call_neighborhood(bindings, args, direction,
                static_cast<size_t>(args.value("max_depth", 2)),
                static_cast<size_t>(args.value("max_nodes", 32)),
                static_cast<size_t>(args.value("max_edges", 64)), timeout_ms);
            if (analysis.failure) return canonical_provider_result(std::move(*analysis.failure), false);
            auto graph_json = call_visualization(analysis.relationships, analysis.complete, analysis.truncated,
                static_cast<size_t>(args.value("max_edges", 64)));
            common_agent_visualization_relationship graph;
            std::string error;
            if (!common_agent_visualization_relationship_from_json(graph_json, graph, error))
                return common_adapter_execution_failure("tool.code.impact.invalid_projection", error,
                    "The bounded code analysis could not be projected as a valid relationship graph.");
            const json response = {{"schema", "code-analysis.v1"},
                {"operation", "bounded_call_neighborhood"}, {"backend", "semantic_provider"},
                {"method", "semantic_call_hierarchy"}, {"column_encoding", "utf-16"},
                {"semantic", true}, {"direction", direction}, {"symbol", args["symbol"]},
                {"max_depth", args.value("max_depth", 2)},
                {"complete", graph.complete}, {"truncated", graph.truncated},
                {"provider_requests", analysis.relationships["provider_requests"]},
                {"continuations", analysis.relationships["continuations"]},
                {"provider_failures", analysis.relationships["provider_failures"]},
                {"relationship_count", graph.edges.size()}, {"visualization", graph_json},
                {"limitations", graph.limitations}};
            if (response.dump().size() > 64 * 1024)
                return common_adapter_limit_failure("tool.code.impact.output_limit",
                    "call neighborhood exceeded the bounded response size",
                    "Code analysis returned too much graph data; reduce max_nodes or max_edges.");
            return common_adapter_success_json(response);
        }, error);
    } else if (calls) {
        const bool legacy = id.rfind("builtin.diagnostics.", 0) == 0;
        installed = common_adapter_register_definition(definition, registry, [bindings, legacy](const std::string & input) {
            std::string parse_error;
            json args;
            if (!common_adapter_parse_object(input, args, parse_error) || !args.contains("symbol") ||
                    !args["symbol"].is_string() || args["symbol"].get_ref<const std::string &>().empty() ||
                    args["symbol"].get_ref<const std::string &>().size() > 256 ||
                    args["symbol"].get_ref<const std::string &>().find('\0') != std::string::npos)
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
            const auto & semantic = legacy ? bindings.diagnostics_call_hierarchy : bindings.code_calls;
            if (!semantic) return common_adapter_execution_failure(legacy ? "tool.diagnostics.call_hierarchy.unavailable" : "tool.code.calls.unavailable",
                legacy ? "diagnostics.call_hierarchy requires a semantic provider" : "semantic call analysis is not configured",
                "Call relationships are unavailable until a semantic code provider is configured.");
            if (!legacy && !((args.contains("definition_path") && args.contains("definition_line")) ||
                    (args.contains("path_hint") && args.contains("line"))))
                return common_adapter_validation_failure("tool.code.calls.invalid_arguments", "code.calls requires an explicit source location");
            args["max_results"] = limit;
            auto result = semantic(args.dump());
            if (!result.ok) return canonical_provider_result(std::move(result), legacy);
            if (legacy) return result;
            auto output = json::parse(result.output, nullptr, false);
            if (!output.is_object() || !output.contains("callers") || !output["callers"].is_array() ||
                    !output.contains("callees") || !output["callees"].is_array() ||
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
            bool provider_complete = true, provider_truncated = false;
            json provider_limitations;
            if (!provider_coverage(output, provider_complete, provider_truncated, provider_limitations))
                return common_adapter_execution_failure("tool.code.invalid_provider_result", "invalid provider coverage", "Code analysis returned invalid coverage metadata.");
            if (!output.contains("truncated")) provider_truncated =
                ((direction == "callers" || direction == "both") && output["callers"].size() >= static_cast<size_t>(limit)) ||
                ((direction == "callees" || direction == "both") && output["callees"].size() >= static_cast<size_t>(limit));
            provider_complete = provider_complete && !provider_truncated;
            output["limitations"] = provider_limitations;
            const auto graph = call_visualization(output, provider_complete, provider_truncated, static_cast<size_t>(limit));
            common_agent_visualization_relationship validated_graph;
            std::string graph_error;
            if (!common_agent_visualization_relationship_from_json(graph, validated_graph, graph_error))
                return common_adapter_execution_failure("tool.code.invalid_provider_result", graph_error,
                    "Code analysis could not produce a valid relationship projection.");
            auto limitations = json::array({
                "The provider returns one-hop call relationships, not a proof of feasible execution."});
            for (const auto & limitation : graph.value("limitations", json::array())) limitations.push_back(limitation);
            auto canonical_output = json{{"schema", "code-analysis.v1"},
                {"operation", "calls"}, {"backend", output.value("backend", std::string("semantic_provider"))},
                {"method", "semantic_call_hierarchy"}, {"semantic", true}, {"column_encoding", "utf-16"}, {"depth", 1},
                {"symbol", args["symbol"]}, {"count", graph["edges"].size()}, {"provider_count", actual_count},
                {"truncated", graph["truncated"]}, {"complete", graph["complete"]},
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
