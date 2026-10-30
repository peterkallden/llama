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
    return {{"id", n.id}, {"kind", n.kind}, {"name", n.name}, {"path", n.path},
        {"line", n.line}, {"source_revision", n.source_revision}, {"source_ref", n.source_ref}};
}
json edge_json(const common_context_graph_edge & e) {
    return {{"id", e.id}, {"from", e.from}, {"to", e.to}, {"relation", e.relation},
        {"status", e.status}, {"source", e.source}, {"rationale", e.rationale},
        {"evidence_refs", e.evidence_refs}, {"source_revision", e.source_revision}, {"version", e.version},
        {"freshness", "unverified"}};
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
            (input.contains("line") && (!input["line"].is_number_integer() || input["line"] < 1 || input["line"] > 10000000))) {
        error = "invalid graph endpoint"; return false;
    }
    for (auto it = input.begin(); it != input.end(); ++it)
        if (it.key() != "kind" && it.key() != "name" && it.key() != "path" && it.key() != "line" && it.key() != "source_revision" && it.key() != "source_ref") {
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
    if ((kind == "actor" || kind == "claim" || kind == "event" || kind == "plan") && ref.empty()) {
        error = "domain nodes require an explicit source reference"; return false;
    }
    auto identity_parts = json::array({scope.namespace_id, scope.project_id, kind, name, path, line, kind == "symbol" ? revision : std::string{}});
    if (!ref.empty()) identity_parts.push_back(ref);
    id = identity("node:", identity_parts);
    if (!find_node(graph, id)) graph.nodes.push_back({id, kind, name, path, line, revision, ref});
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
        const auto haystack = lower(node.name + " " + node.path + " " + node.source_ref);
        size_t score = haystack.find(text) != std::string::npos ? 100 : 0;
        for (const auto & term : terms) if (haystack.find(term) != std::string::npos) ++score;
        if (score) matches.push_back({score, &node});
    }
    std::stable_sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    if (matches.empty()) return {};
    json result = {{"revision", graph.revision}, {"nodes", json::array()}, {"edges", json::array()}, {"truncated", matches.size() > 3}};
    std::set<std::string> ids;
    for (size_t i = 0; i < std::min<size_t>(3, matches.size()); ++i) {
        ids.insert(matches[i].second->id); result["nodes"].push_back(node_json(*matches[i].second));
    }
    if (include_edges) for (const auto & edge : graph.edges) {
        if (!ids.count(edge.from) && !ids.count(edge.to)) continue;
        const auto * from = find_node(graph, edge.from); const auto * to = find_node(graph, edge.to);
        if (!from || !to || result["edges"].size() >= 8) { result["truncated"] = true; continue; }
        for (const auto * node : {from, to}) if (!ids.count(node->id)) {
            if (result["nodes"].size() >= 8) { result["truncated"] = true; break; }
            ids.insert(node->id); result["nodes"].push_back(node_json(*node));
        }
        if (ids.count(edge.from) && ids.count(edge.to)) result["edges"].push_back(edge_json(edge));
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
    const bool write = operation == "graph.link.propose" || operation == "graph.retract.propose";
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
                const auto text = lower(node.name + " " + node.path + " " + node.source_ref);
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
            const auto id = args["id"].get<std::string>();
            const auto * node = find_node(graph, id);
            if (node) return common_adapter_success_json({{"revision", graph.revision}, {"node", node_json(*node)}});
            const auto edge = std::find_if(graph.edges.begin(), graph.edges.end(), [&](const auto & item) { return item.id == id; });
            if (edge != graph.edges.end()) return common_adapter_success_json({{"revision", graph.revision}, {"edge", edge_json(*edge)}});
            return common_adapter_not_found_failure("tool.graph.not_found", "graph item is unavailable", "Graph item is unavailable.");
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
        if (operation == "graph.link.propose") {
            if (!args.contains("from") || !args.contains("to") || !text_value(args, "relation", 64) ||
                    !text_value(args, "rationale", 512) || !text_value(args, "source_revision", 256, false) ||
                    (args.contains("evidence_refs") && (!args["evidence_refs"].is_array() || args["evidence_refs"].size() > 16)))
                return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid link proposal");
            const auto relation = args["relation"].get<std::string>();
            if (relation != "calls" && relation != "references" && relation != "contains" && relation != "depends_on" && relation != "implements" && relation != "related_to" &&
                    relation != "supports" && relation != "contradicts" && relation != "associated_with" && relation != "observed_in")
                return common_adapter_validation_failure("tool.graph.invalid_arguments", "unsupported relation");
            std::vector<std::string> refs;
            if (args.contains("evidence_refs")) for (const auto & value : args["evidence_refs"]) {
                if (!value.is_string() || value.get_ref<const std::string &>().empty() || value.get_ref<const std::string &>().size() > 512)
                    return common_adapter_validation_failure("tool.graph.invalid_arguments", "invalid evidence reference");
                refs.push_back(value.get<std::string>());
            }
            std::string from, to;
            if (!resolve_endpoint(args["from"], bindings.tool_scope, graph, from, error) ||
                    !resolve_endpoint(args["to"], bindings.tool_scope, graph, to, error))
                return common_adapter_validation_failure("tool.graph.invalid_arguments", error);
            common_context_graph_edge edge;
            edge.from = from; edge.to = to; edge.relation = relation; edge.rationale = args["rationale"].get<std::string>();
            edge.evidence_refs = refs; edge.source_revision = args.value("source_revision", std::string{});
            edge.id = identity("edge:", json::array({bindings.tool_scope.namespace_id, bindings.tool_scope.project_id, from, relation, to, edge.rationale, refs, edge.source_revision}));
            const auto prior = std::find_if(graph.edges.begin(), graph.edges.end(), [&](const auto & item) { return item.id == edge.id; });
            if (prior != graph.edges.end()) return common_adapter_success_json({{"revision", graph.revision}, {"edge", edge_json(*prior)}, {"reused", true}});
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
            edge->status = "retraction_proposed"; edge->rationale = args["rationale"].get<std::string>(); ++edge->version;
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
