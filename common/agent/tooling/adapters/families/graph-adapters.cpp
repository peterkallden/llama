#include "agent/tooling/adapters/families/graph-adapters.h"
#include "agent/tooling/adapters/support/adapter-support.h"
#include "agent/visualization/visualization-contract.h"
#include "hash/hash.h"

#include <algorithm>
#include <cctype>
#include <deque>
#include <filesystem>
#include <set>
#include <sstream>

namespace {
using json = common_adapter_json;

json node_json(const common_context_graph_node & n) {
    json observations = json::array();
    const size_t offset = n.observations.size() > 8 ? n.observations.size() - 8 : 0;
    for (size_t i = offset; i < n.observations.size(); ++i) {
        const auto & o = n.observations[i];
        observations.push_back({{"path", o.path}, {"line", o.line},
            {"source_revision", o.source_revision}, {"source_ref", o.source_ref}});
    }
    const auto * latest = n.observations.empty() ? nullptr : &n.observations.back();
    json descriptions = json::array();
    const auto description_offset = n.descriptions.size() > 4 ? n.descriptions.size() - 4 : 0;
    for (size_t i = description_offset; i < n.descriptions.size(); ++i) {
        const auto & d = n.descriptions[i];
        descriptions.push_back({{"text", d.text}, {"source", d.source},
        {"status", "proposed"}, {"source_revision", d.source_revision}, {"evidence_refs", d.evidence_refs}});
    }
    return {{"id", n.id}, {"kind", n.kind}, {"name", n.name}, {"path", latest ? latest->path : n.path},
        {"line", latest ? latest->line : n.line}, {"source_revision", latest ? latest->source_revision : n.source_revision},
        {"source_ref", latest ? latest->source_ref : n.source_ref},
        {"symbol_key", n.symbol_key}, {"observations", observations}, {"observations_truncated", offset != 0},
        {"description", descriptions.empty() ? json(nullptr) : descriptions.back()}, {"descriptions", descriptions},
        {"descriptions_truncated", description_offset != 0}};
}
json node_prompt_json(const common_context_graph_node & node) {
    auto value = node_json(node);
    value.erase("observations");
    value.erase("observations_truncated");
    value.erase("descriptions");
    value.erase("descriptions_truncated");
    value["observation_count"] = node.observations.size();
    return value;
}
json edge_json(const common_context_graph_edge & e) {
    json assertions = json::array();
    size_t bytes = 0;
    for (const auto & a : e.assertions) {
        json item = {{"source", a.source}, {"rationale", a.rationale}, {"evidence_refs", a.evidence_refs}, {"source_revision", a.source_revision}};
        const auto size = item.dump().size();
        if (assertions.size() >= 8 || bytes + size > 12000) break;
        bytes += size; assertions.push_back(std::move(item));
    }
    return {{"id", e.id}, {"from", e.from}, {"to", e.to}, {"relation", e.relation},
        {"status", e.status}, {"source", e.source}, {"rationale", e.rationale},
        {"evidence_refs", e.evidence_refs}, {"source_revision", e.source_revision}, {"version", e.version},
        {"retraction_rationale", e.retraction_rationale}, {"assertions", assertions}, {"assertions_truncated", assertions.size() < e.assertions.size()},
        {"freshness", "unverified"}};
}
json edge_prompt_json(const common_context_graph_edge & edge) {
    auto value = edge_json(edge);
    value.erase("assertions");
    value.erase("assertions_truncated");
    value["assertion_count"] = edge.assertions.size();
    return value;
}
bool text_value(const json & v, const char * key, size_t maximum, bool required = true) {
    if (!v.contains(key)) return !required;
    return v[key].is_string() && (!required || !v[key].get_ref<const std::string &>().empty()) &&
        v[key].get_ref<const std::string &>().size() <= maximum &&
        v[key].get_ref<const std::string &>().find('\0') == std::string::npos;
}
std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
std::string node_search_text(const common_context_graph_node & node) {
    std::string text = node.name + " " + node.path + " " + node.source_ref + " " + node.symbol_key;
    for (const auto & observation : node.observations) text += " " + observation.path + " " + observation.source_ref;
    if (!node.descriptions.empty()) text += " " + node.descriptions.back().text;
    return lower(std::move(text));
}
std::string identity(const std::string & prefix, const json & value) {
    const auto bytes = value.dump();
    return prefix + hash_sha256_hex(bytes.data(), bytes.size());
}
const common_context_graph_node * find_node(const common_context_graph_state & graph, const std::string & id) {
    const auto it = std::find_if(graph.nodes.begin(), graph.nodes.end(), [&](const auto & n) { return n.id == id; });
    return it == graph.nodes.end() ? nullptr : &*it;
}
bool resolve_endpoint(const json & input, const common_agent_scope & scope, common_context_graph_state & graph,
        std::string & id, std::string & error) {
    if (!input.is_object()) { error = "graph endpoint must be an object"; return false; }
    if (input.contains("id")) {
        if (input.size() != 1 || !text_value(input, "id", 256)) { error = "invalid existing graph endpoint"; return false; }
        id = input["id"].get<std::string>();
        if (!find_node(graph, id)) { error = "graph endpoint is unavailable"; return false; }
        return true;
    }
    if (!text_value(input, "kind", 32) || !text_value(input, "name", 512) ||
            !text_value(input, "path", 512, false) || !text_value(input, "source_revision", 256, false) || !text_value(input, "source_ref", 512, false) ||
            !text_value(input, "symbol_key", 512, false) ||
            (input.contains("line") && (!input["line"].is_number_integer() || input["line"] < 1 || input["line"] > 10000000))) {
        error = "invalid graph endpoint"; return false;
    }
    for (auto it = input.begin(); it != input.end(); ++it)
        if (it.key() != "kind" && it.key() != "name" && it.key() != "path" && it.key() != "line" && it.key() != "source_revision" && it.key() != "source_ref" && it.key() != "symbol_key") {
            error = "unknown graph endpoint field"; return false;
        }
    const auto kind = input["kind"].get<std::string>();
    if (kind != "file" && kind != "symbol" && kind != "component" && kind != "memory" && kind != "resource" &&
            kind != "actor" && kind != "claim" && kind != "event" && kind != "plan") {
        error = "unsupported graph node kind"; return false;
    }
    const auto path = input.value("path", std::string{});
    const auto parsed = std::filesystem::path(path);
    if (parsed.is_absolute() || std::any_of(parsed.begin(), parsed.end(), [](const auto & part) { return part == ".."; })) {
        error = "graph path must be repository-relative"; return false;
    }
    if ((kind == "file" || kind == "symbol") && path.empty()) { error = "file and symbol nodes require a path"; return false; }
    if (kind == "symbol" && !input.contains("line")) { error = "symbol nodes require a source location"; return false; }
    const auto name = input["name"].get<std::string>();
    const auto line = input.value("line", int64_t(0));
    const auto revision = input.value("source_revision", std::string{});
    const auto ref = input.value("source_ref", std::string{});
    const auto symbol_key = input.value("symbol_key", std::string{});
    if (kind != "symbol" && !symbol_key.empty()) { error = "symbol_key requires a symbol node"; return false; }
    if ((kind == "actor" || kind == "claim" || kind == "event" || kind == "plan") && ref.empty()) {
        error = "domain nodes require an explicit source reference"; return false;
    }
    auto identity_parts = json::array({scope.namespace_id, scope.project_id, kind, name, path, line, kind == "symbol" ? revision : std::string{}});
    if (!ref.empty()) identity_parts.push_back(ref);
    if (kind == "symbol" && !symbol_key.empty()) identity_parts = json::array({scope.namespace_id, scope.project_id, kind, "symbol_key", symbol_key});
    id = identity("node:", identity_parts);
    auto found = std::find_if(graph.nodes.begin(), graph.nodes.end(), [&](const auto & node) { return node.id == id; });
    if (found == graph.nodes.end()) {
        graph.nodes.push_back({id, kind, name, path, line, revision, ref, symbol_key, {}, {}});
        found = graph.nodes.end() - 1;
    }
    if (kind == "symbol" && !symbol_key.empty()) {
        const auto seen = std::find_if(found->observations.begin(), found->observations.end(), [&](const auto & o) {
            return o.path == path && o.line == line && o.source_revision == revision && o.source_ref == ref;
        });
        if (seen == found->observations.end()) {
            if (found->observations.size() >= 256) { error = "symbol observation limit reached"; return false; }
            found->observations.push_back({path, line, revision, ref});
        }
    }
    return true;
}
}

std::string common_context_graph_prompt_projection(common_agent_data_store & store, const common_agent_scope & scope,
        const std::string & query, size_t budget, bool include_edges) {
    if (scope.namespace_id.empty() || scope.project_id.empty() || query.empty() || budget == 0) return {};
    common_context_graph_state graph; std::string error;
    if (!store.get_context_graph(scope, graph, error)) return {};
    const auto text = lower(query.substr(0, 512));
    std::vector<std::string> terms; std::string token;
    for (const auto c : text) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') token += c;
        else { if (token.size() >= 4) terms.push_back(token); token.clear(); }
    }
    if (token.size() >= 4) terms.push_back(token);
    std::vector<std::pair<size_t, const common_context_graph_node *>> matches;
    for (const auto & node : graph.nodes) {
        const auto haystack = node_search_text(node);
        size_t score = haystack.find(text) != std::string::npos ? 100 : 0;
        for (const auto & term : terms) if (haystack.find(term) != std::string::npos) ++score;
        if (score) matches.push_back({score, &node});
    }
    std::stable_sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    if (matches.empty()) return {};
    json result = {{"revision", graph.revision}, {"nodes", json::array()}, {"edges", json::array()}, {"truncated", matches.size() > 3}};
    std::set<std::string> ids;
    for (size_t i = 0; i < std::min<size_t>(3, matches.size()); ++i) {
        ids.insert(matches[i].second->id); result["nodes"].push_back(node_prompt_json(*matches[i].second));
    }
    std::vector<const common_context_graph_edge *> context_edges;
    for (const auto & edge : graph.edges) context_edges.push_back(&edge);
    std::stable_sort(context_edges.begin(), context_edges.end(), [](const auto * a, const auto * b) {
        const auto priority = [](const std::string & status) {
            return status == "proposed" ? 0 : status == "retraction_proposed" ? 1 : 2;
        };
        return priority(a->status) < priority(b->status);
    });
    if (include_edges) for (const auto * candidate : context_edges) {
        const auto & edge = *candidate;
        if (!ids.count(edge.from) && !ids.count(edge.to)) continue;
        const auto * from = find_node(graph, edge.from); const auto * to = find_node(graph, edge.to);
        if (!from || !to || result["edges"].size() >= 8) { result["truncated"] = true; continue; }
        for (const auto * node : {from, to}) if (!ids.count(node->id)) {
            if (result["nodes"].size() >= 8) { result["truncated"] = true; break; }
            ids.insert(node->id); result["nodes"].push_back(node_prompt_json(*node));
        }
        if (ids.count(edge.from) && ids.count(edge.to)) result["edges"].push_back(edge_prompt_json(edge));
    }
    budget = std::min<size_t>(budget, 1800);
    while (result.dump().size() > budget && !result["edges"].empty()) { result["edges"].erase(result["edges"].end() - 1); result["truncated"] = true; }
    while (result.dump().size() > budget && !result["nodes"].empty()) { result["nodes"].erase(result["nodes"].end() - 1); result["truncated"] = true; }
    return result["nodes"].empty() ? std::string{} : result.dump();
}

bool common_try_register_graph_tool_adapter(const common_tool_definition & definition,
        const common_native_tool_bindings & bindings, common_tool_registry & registry, bool & installed, std::string & error) {
    installed = false;
    if (definition.executor_id.rfind("builtin.graph.", 0) != 0) return false;
    if (!bindings.data_store) return true;
    const auto operation = definition.name;
    const bool write = operation == "graph.link.propose" || operation == "graph.retract.propose" || operation == "graph.describe.propose";
    installed = common_adapter_register_definition(definition, registry, [bindings, operation](const std::string & input) {
        std::string error;
        json args;
        if (!common_adapter_parse_object(input, args, error)) return common_adapter_validation_failure("tool.graph.invalid_arguments", error);
        // The native tool view owns delegated source authorization and dispatch.
        // A bare registry must never bypass that view by invoking source bindings.
        if (operation == "graph.source") return common_adapter_execution_failure("tool.graph.source_unavailable",
            "source graph dispatch requires an authorized native tool view", "Source graph dispatch is unavailable.");
        if (bindings.tool_scope.namespace_id.empty() || bindings.tool_scope.project_id.empty())
            return common_adapter_validation_failure("tool.graph.scope_invalid", "context graph requires a host-bound namespace and project");
        common_context_graph_state graph;
        if (!bindings.data_store->get_context_graph(bindings.tool_scope, graph, error))
            return common_adapter_execution_failure("tool.graph.unavailable", error, "The project graph is unavailable.");
        if (operation == "graph.find") {
            if (!text_value(args, "query", 512)) return common_adapter_validation_failure("tool.graph.invalid_arguments", "query is required");
            const auto query = lower(args["query"].get<std::string>());
            std::vector<std::string> terms;
            std::string term;
            for (const auto c : query) {
                if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') term += c;
                else { if (term.size() >= 4) terms.push_back(term); term.clear(); }
            }
            if (term.size() >= 4) terms.push_back(term);
            std::vector<std::pair<size_t, const common_context_graph_node *>> matches;
            for (const auto & node : graph.nodes) {
                const auto text = node_search_text(node);
                size_t score = text.find(query) != std::string::npos ? 100 : 0;
                for (const auto & word : terms) if (text.find(word) != std::string::npos) ++score;
                if (score) matches.push_back({score, &node});
            }
            std::stable_sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            json nodes = json::array(); const bool truncated = matches.size() > 32;
            for (size_t i = 0; i < std::min<size_t>(32, matches.size()); ++i) nodes.push_back(node_json(*matches[i].second));
            return common_adapter_success_json({{"revision", graph.revision}, {"nodes", nodes}, {"truncated", truncated}});
        }
        if (operation == "graph.get") {
            if (!text_value(args, "id", 256)) return common_adapter_validation_failure("tool.graph.invalid_arguments", "id is required");
            if (args.contains("observation_offset") && (!args["observation_offset"].is_number_unsigned() || args["observation_offset"] > 256))
                return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid observation offset");
            if (args.contains("assertion_offset") && (!args["assertion_offset"].is_number_unsigned() || args["assertion_offset"] > 64))
                return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid assertion offset");
            if (args.contains("description_offset") && (!args["description_offset"].is_number_unsigned() || args["description_offset"] > 64))
                return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid description offset");
            const auto id = args["id"].get<std::string>();
            const auto * node = find_node(graph, id);
            if (node) {
                if (args.contains("assertion_offset")) return common_adapter_validation_failure("tool.graph.invalid_arguments", "assertion pagination requires an edge");
                json result = {{"revision", graph.revision}, {"node", node_json(*node)}};
                if (args.contains("description_offset")) {
                    const auto offset = args["description_offset"].get<size_t>();
                    json page = json::array();
                    for (size_t i = offset; i < node->descriptions.size() && i - offset < 4; ++i) {
                        const auto & d = node->descriptions[i];
                        page.push_back({{"text", d.text}, {"source", d.source}, {"status", "proposed"},
                            {"source_revision", d.source_revision}, {"evidence_refs", d.evidence_refs}});
                    }
                    const auto end = std::min(node->descriptions.size(), offset + page.size());
                    result["description_page"] = {{"offset", offset}, {"items", page}, {"total", node->descriptions.size()},
                        {"next_offset", end < node->descriptions.size() ? json(end) : json(nullptr)}};
                }
                if (args.contains("observation_offset")) {
                    const auto offset = args["observation_offset"].get<size_t>();
                    json page = json::array();
                    for (size_t i = offset; i < node->observations.size() && i - offset < 8; ++i) {
                        const auto & o = node->observations[i];
                        page.push_back({{"path", o.path}, {"line", o.line}, {"source_revision", o.source_revision}, {"source_ref", o.source_ref}});
                    }
                    const auto end = std::min(node->observations.size(), offset + page.size());
                    result["observation_page"] = {{"offset", offset}, {"items", page}, {"total", node->observations.size()},
                        {"next_offset", end < node->observations.size() ? json(end) : json(nullptr)}};
                }
                return common_adapter_success_json(result);
            }
            if (args.contains("observation_offset") || args.contains("description_offset")) return common_adapter_validation_failure("tool.graph.invalid_arguments", "node pagination requires a node");
            const auto edge = std::find_if(graph.edges.begin(), graph.edges.end(), [&](const auto & item) { return item.id == id; });
            if (edge != graph.edges.end()) {
                json result = {{"revision", graph.revision}, {"edge", edge_json(*edge)}};
                if (args.contains("assertion_offset")) {
                    result["edge"] = {{"id", edge->id}, {"from", edge->from}, {"to", edge->to},
                        {"relation", edge->relation}, {"status", edge->status}, {"version", edge->version}};
                    const auto offset = args["assertion_offset"].get<size_t>();
                    json page = json::array();
                    if (offset < edge->assertions.size()) {
                        const auto & a = edge->assertions[offset];
                        page.push_back({{"source", a.source}, {"rationale", a.rationale}, {"evidence_refs", a.evidence_refs}, {"source_revision", a.source_revision}});
                    }
                    const auto end = std::min(edge->assertions.size(), offset + page.size());
                    result["assertion_page"] = {{"offset", offset}, {"items", page}, {"total", edge->assertions.size()},
                        {"next_offset", end < edge->assertions.size() ? json(end) : json(nullptr)}};
                }
                return common_adapter_success_json(result);
            }
            return common_adapter_not_found_failure("tool.graph.not_found", "graph item is unavailable", "Graph item is unavailable.");
        }
        if (operation == "graph.search") {
            for (const char * key : {"from", "to", "relation", "from_kind", "to_kind", "status"})
                if (!text_value(args, key, std::string(key) == "relation" ? 64 :
                        (std::string(key).find("kind") != std::string::npos || std::string(key) == "status" ? 32 : 256), false) ||
                        (args.contains(key) && args[key].get_ref<const std::string &>().empty()))
                    return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid graph search filter");
            const auto from = args.value("from", std::string{}), to = args.value("to", std::string{});
            const auto relation = args.value("relation", std::string{}), from_kind = args.value("from_kind", std::string{});
            const auto to_kind = args.value("to_kind", std::string{}), status = args.value("status", std::string{});
            if (from.empty() && to.empty() && relation.empty() && from_kind.empty() && to_kind.empty() && status.empty())
                return common_adapter_validation_failure("tool.graph.invalid_arguments", "at least one graph search filter is required");
            if (!status.empty() && status != "proposed" && status != "retraction_proposed")
                return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid graph status");
            if (args.contains("limit") && (!args["limit"].is_number_integer() || args["limit"] < 1 || args["limit"] > 100))
                return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid graph search limit");
            const auto limit = args.value("limit", size_t(32));
            std::vector<std::string> ids;
            if (!bindings.data_store->find_context_graph_edges(bindings.tool_scope, graph.revision, from, relation, to, ids, error)) {
                const bool conflict = error == "context graph revision conflict";
                return common_adapter_failure("tool.graph.search_failed", conflict ? common_tool_failure_class::conflict : common_tool_failure_class::execution,
                    conflict, conflict ? "Graph changed during search; retry after reloading." : "Graph search is unavailable.", error);
            }
            std::sort(ids.begin(), ids.end());
            json nodes = json::array(), edges = json::array(); std::set<std::string> seen;
            bool truncated = false; size_t bytes = 0;
            for (const auto & id : ids) {
                const auto edge = std::find_if(graph.edges.begin(), graph.edges.end(), [&](const auto & e) { return e.id == id; });
                if (edge == graph.edges.end()) return common_adapter_execution_failure("tool.graph.search_invalid", "index references an unavailable edge", "Graph search index is inconsistent.");
                const auto * a = find_node(graph, edge->from); const auto * b = find_node(graph, edge->to);
                if (!a || !b) return common_adapter_execution_failure("tool.graph.search_invalid", "edge endpoint is unavailable", "Graph search index is inconsistent.");
                if ((!from_kind.empty() && a->kind != from_kind) || (!to_kind.empty() && b->kind != to_kind) ||
                        (!status.empty() && edge->status != status)) continue;
                const auto item = edge_json(*edge);
                size_t added = item.dump().size();
                if (!seen.count(a->id)) added += node_json(*a).dump().size();
                if (b->id != a->id && !seen.count(b->id)) added += node_json(*b).dump().size();
                if (edges.size() >= limit || bytes + added > 22000) { truncated = true; break; }
                bytes += added; edges.push_back(item);
                for (const auto * node : {a, b}) if (seen.insert(node->id).second) nodes.push_back(node_json(*node));
            }
            return common_adapter_success_json({{"revision", graph.revision}, {"nodes", nodes}, {"edges", edges},
                {"truncated", truncated}, {"coverage", "saved_project_graph_only"}});
        }
        if (operation == "graph.expand") {
            if (!text_value(args, "id", 256)) return common_adapter_validation_failure("tool.graph.invalid_arguments", "id is required");
            const auto center = args["id"].get<std::string>();
            if (!find_node(graph, center)) return common_adapter_not_found_failure("tool.graph.not_found", "graph node is unavailable", "Graph node is unavailable.");
            const int depth = args.value("depth", 1);
            if (depth < 1 || depth > 2) return common_adapter_validation_failure("tool.graph.invalid_arguments", "depth must be one or two");
            std::set<std::string> seen{center}, edge_ids;
            std::deque<std::pair<std::string, int>> queue{{center, 0}};
            json edges = json::array(), nodes = json::array(); bool truncated = false;
            size_t output_bytes = node_json(*find_node(graph, center)).dump().size();
            while (!queue.empty()) {
                const auto current = queue.front(); queue.pop_front();
                if (current.second >= depth) continue;
                for (const auto & edge : graph.edges) {
                    if (edge.from != current.first && edge.to != current.first) continue;
                    const auto next = edge.from == current.first ? edge.to : edge.from;
                    if (!find_node(graph, next)) { truncated = true; continue; }
                    if (!edge_ids.count(edge.id)) {
                        const auto serialized_edge = edge_json(edge);
                        const size_t candidate_bytes = serialized_edge.dump().size() +
                            (seen.count(next) ? 0 : node_json(*find_node(graph, next)).dump().size());
                        if (edges.size() >= 64 || (!seen.count(next) && seen.size() >= 32) || output_bytes + candidate_bytes > 22000) { truncated = true; continue; }
                        output_bytes += candidate_bytes;
                        edge_ids.insert(edge.id); edges.push_back(std::move(serialized_edge));
                    }
                    if (seen.insert(next).second) queue.emplace_back(next, current.second + 1);
                }
            }
            common_agent_visualization_relationship visual;
            visual.title = "Project context graph"; visual.data_revision = std::to_string(graph.revision);
            visual.complete = !truncated; visual.truncated = truncated;
            visual.limitations.push_back("Only saved project graph proposals are included; source providers were not searched.");
            if (truncated) visual.limitations.push_back("Graph expansion reached a bound or omitted an invalid endpoint.");
            for (const auto & node : graph.nodes) if (seen.count(node.id)) {
                nodes.push_back(node_json(node)); visual.nodes.push_back({node.id, node.name, node.kind});
            }
            for (const auto & edge : graph.edges) if (edge_ids.count(edge.id))
                visual.edges.push_back({edge.id, edge.from, edge.to, edge.relation + " (" + edge.status + ")", true, edge.relation, edge.evidence_refs});
            return common_adapter_success_json({{"revision", graph.revision}, {"depth", depth}, {"nodes", nodes}, {"edges", edges},
                {"complete", !truncated}, {"truncated", truncated}, {"coverage", "saved_project_graph_only"},
                {"limitations", visual.limitations}, {"visualization", common_agent_visualization_relationship_to_json(visual)}});
        }
        if (operation == "graph.describe.propose") {
            if (!text_value(args, "id", 256) || !text_value(args, "description", 512) ||
                    !text_value(args, "source_revision", 256, false) || !args.contains("expected_revision") ||
                    !args["expected_revision"].is_number_unsigned() ||
                    (args.contains("evidence_refs") && (!args["evidence_refs"].is_array() || args["evidence_refs"].size() > 16)))
                return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid description proposal");
            if (graph.revision != args["expected_revision"].get<uint64_t>())
                return common_adapter_failure("tool.graph.conflict", common_tool_failure_class::conflict, true, "Graph has changed.", "graph revision conflict");
            auto node = std::find_if(graph.nodes.begin(), graph.nodes.end(), [&](const auto & n) { return n.id == args["id"].get<std::string>(); });
            if (node == graph.nodes.end()) return common_adapter_not_found_failure("tool.graph.not_found", "graph node is unavailable", "Graph node is unavailable.");
            std::vector<std::string> refs;
            if (args.contains("evidence_refs")) for (const auto & ref : args["evidence_refs"]) {
                if (!ref.is_string() || ref.get_ref<const std::string &>().empty() || ref.get_ref<const std::string &>().size() > 512)
                    return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid evidence reference");
                refs.push_back(ref.get<std::string>());
            }
            const auto text = args["description"].get<std::string>();
            const auto source_revision = args.value("source_revision", std::string{});
            if (!node->descriptions.empty()) {
                const auto & last = node->descriptions.back();
                if (last.text == text && last.source == "model" && last.source_revision == source_revision && last.evidence_refs == refs)
                    return common_adapter_success_json({{"revision", graph.revision}, {"node", node_json(*node)}, {"reused", true}});
            }
            if (node->descriptions.size() >= 64) return common_adapter_validation_failure("tool.graph.limit", "description limit reached");
            node->descriptions.push_back({text, "model", source_revision, refs});
            const auto expected = graph.revision++;
            common_context_graph_state published;
            const auto result = bindings.data_store->publish_context_graph(bindings.tool_scope, expected, graph, published, error);
            if (result != common_agent_publication_status::published)
                return common_adapter_failure("tool.graph.publish_failed", result == common_agent_publication_status::conflict ? common_tool_failure_class::conflict : common_tool_failure_class::execution,
                    result == common_agent_publication_status::conflict, "Graph description could not be saved.", error);
            return common_adapter_success_json({{"revision", published.revision}, {"node", node_json(*node)}, {"reused", false}});
        }
        if (operation == "graph.link.propose") {
            if (!args.contains("from") || !args.contains("to") || !text_value(args, "relation", 64) ||
                    !text_value(args, "rationale", 512) || !text_value(args, "source_revision", 256, false) ||
                    (args.contains("evidence_refs") && (!args["evidence_refs"].is_array() || args["evidence_refs"].size() > 16)))
                return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid link proposal");
            const auto relation = args["relation"].get<std::string>();
            std::vector<std::string> refs;
            if (args.contains("evidence_refs")) for (const auto & value : args["evidence_refs"]) {
                if (!value.is_string() || value.get_ref<const std::string &>().empty() || value.get_ref<const std::string &>().size() > 512)
                    return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid evidence reference");
                refs.push_back(value.get<std::string>());
            }
            json old_nodes = json::array();
            for (const auto & node : graph.nodes) old_nodes.push_back(node_json(node));
            std::string from, to;
            if (!resolve_endpoint(args["from"], bindings.tool_scope, graph, from, error) ||
                    !resolve_endpoint(args["to"], bindings.tool_scope, graph, to, error))
                return common_adapter_validation_failure("tool.graph.invalid_arguments", error);
            common_context_graph_edge edge;
            edge.from = from; edge.to = to; edge.relation = relation; edge.rationale = args["rationale"].get<std::string>();
            edge.evidence_refs = refs; edge.source_revision = args.value("source_revision", std::string{});
            edge.id = identity("edge:", json::array({bindings.tool_scope.namespace_id, bindings.tool_scope.project_id, from, relation, to}));
            edge.assertions.push_back({"model", edge.rationale, refs, edge.source_revision});
            auto prior = std::find_if(graph.edges.begin(), graph.edges.end(), [&](const auto & item) { return item.id == edge.id; });
            if (prior != graph.edges.end()) {
                const auto & assertion = edge.assertions.front();
                const bool known = std::any_of(prior->assertions.begin(), prior->assertions.end(), [&](const auto & a) {
                    return a.source == assertion.source && a.rationale == assertion.rationale &&
                        a.evidence_refs == assertion.evidence_refs && a.source_revision == assertion.source_revision;
                });
                if (!known && prior->status != "proposed")
                    return common_adapter_failure("tool.graph.conflict", common_tool_failure_class::conflict, true,
                        "The relation has a pending retraction.", "cannot append an assertion during retraction");
                if (!known) {
                    if (prior->assertions.size() >= 64) return common_adapter_validation_failure("tool.graph.limit", "assertion limit reached");
                    prior->assertions.push_back(assertion);
                    ++prior->version;
                }
                const auto existing = edge_json(*prior);
                json new_nodes = json::array();
                for (const auto & node : graph.nodes) new_nodes.push_back(node_json(node));
                if (new_nodes == old_nodes && known) return common_adapter_success_json({{"revision", graph.revision}, {"edge", existing}, {"reused", true}});
                const auto expected = graph.revision; graph.revision++;
                common_context_graph_state published;
                const auto result = bindings.data_store->publish_context_graph(bindings.tool_scope, expected, graph, published, error);
                if (result != common_agent_publication_status::published)
                    return common_adapter_failure("tool.graph.publish_failed", result == common_agent_publication_status::conflict ? common_tool_failure_class::conflict : common_tool_failure_class::execution,
                        result == common_agent_publication_status::conflict, "Graph observation could not be saved.", error);
                return common_adapter_success_json({{"revision", published.revision}, {"edge", existing}, {"reused", true}});
            }
            const auto expected = graph.revision; graph.revision++; graph.edges.push_back(edge);
            common_context_graph_state published;
            const auto result = bindings.data_store->publish_context_graph(bindings.tool_scope, expected, graph, published, error);
            if (result != common_agent_publication_status::published)
                return common_adapter_failure("tool.graph.publish_failed", result == common_agent_publication_status::conflict ? common_tool_failure_class::conflict : common_tool_failure_class::execution,
                    result == common_agent_publication_status::conflict, "Graph proposal could not be saved.", error);
            return common_adapter_success_json({{"revision", published.revision}, {"edge", edge_json(edge)}, {"reused", false}});
        }
        if (operation == "graph.retract.propose") {
            if (!text_value(args, "id", 256) || !args.contains("expected_version") || !args["expected_version"].is_number_integer() ||
                    !text_value(args, "rationale", 512)) return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid retraction proposal");
            auto edge = std::find_if(graph.edges.begin(), graph.edges.end(), [&](const auto & item) { return item.id == args["id"].get<std::string>(); });
            if (edge == graph.edges.end()) return common_adapter_not_found_failure("tool.graph.not_found", "graph edge is unavailable", "Graph edge is unavailable.");
            if (edge->version != args["expected_version"].get<uint64_t>()) return common_adapter_failure("tool.graph.conflict", common_tool_failure_class::conflict, true, "Graph edge has changed.", "edge version conflict");
            edge->status = "retraction_proposed"; edge->retraction_rationale = args["rationale"].get<std::string>(); ++edge->version;
            const auto expected = graph.revision; graph.revision++;
            common_context_graph_state published;
            const auto result = bindings.data_store->publish_context_graph(bindings.tool_scope, expected, graph, published, error);
            if (result != common_agent_publication_status::published)
                return common_adapter_failure("tool.graph.publish_failed", result == common_agent_publication_status::conflict ? common_tool_failure_class::conflict : common_tool_failure_class::execution,
                    result == common_agent_publication_status::conflict, "Graph retraction could not be saved.", error);
            return common_adapter_success_json({{"revision", published.revision}, {"edge", edge_json(*edge)}});
        }
        return common_adapter_validation_failure("tool.graph.invalid_operation", "unknown graph operation");
    }, error, !write, write);
    return true;
}
