#include "agent/visualization/visualization-contract.h"

#include <set>
#include <stdexcept>

namespace {
using json = nlohmann::json;

bool bounded_text(const std::string & value, size_t maximum, bool optional = false) {
    return (optional || !value.empty()) && value.size() <= maximum &&
        value.find('\0') == std::string::npos;
}

bool unique_values(const std::vector<std::string> & values,
        const std::set<std::string> & available) {
    std::set<std::string> seen;
    for (const auto & value : values) {
        if (available.count(value) == 0 || !seen.insert(value).second) return false;
    }
    return true;
}
}

bool common_agent_visualization_relationship_validate(
        const common_agent_visualization_relationship & value, std::string & error) {
    if (value.schema_version != 1 || value.kind != "relationship") {
        error = "unsupported visualization schema version or kind"; return false;
    }
    if (!bounded_text(value.title, 256) || !bounded_text(value.data_revision, 256, true)) {
        error = "visualization title or data revision is invalid"; return false;
    }
    if (value.nodes.size() > COMMON_AGENT_VISUALIZATION_MAX_NODES ||
            value.edges.size() > COMMON_AGENT_VISUALIZATION_MAX_EDGES) {
        error = "visualization exceeds its node or edge limit"; return false;
    }
    static const std::set<std::string> layouts{"cose", "circle", "grid", "breadthfirst"};
    if (layouts.count(value.view.layout) == 0) { error = "unsupported visualization layout"; return false; }

    std::set<std::string> node_ids, edge_ids, all_ids;
    for (const auto & node : value.nodes) {
        if (!bounded_text(node.id, 256) || !bounded_text(node.label, 512, true) ||
                !bounded_text(node.kind, 64, true) || !all_ids.insert(node.id).second) {
            error = "visualization contains an invalid or duplicate node"; return false;
        }
        node_ids.insert(node.id);
    }
    for (const auto & edge : value.edges) {
        if (!bounded_text(edge.id, 256) || !bounded_text(edge.source, 256) ||
                !bounded_text(edge.target, 256) || !bounded_text(edge.label, 512, true) ||
                !all_ids.insert(edge.id).second || node_ids.count(edge.source) == 0 ||
                node_ids.count(edge.target) == 0) {
            error = "visualization contains an invalid edge or a missing endpoint"; return false;
        }
        edge_ids.insert(edge.id);
    }
    if (!unique_values(value.view.selected_nodes, node_ids) ||
            !unique_values(value.view.selected_edges, edge_ids)) {
        error = "visualization selection contains an unknown or duplicate id"; return false;
    }
    if (value.limitations.size() > 32) { error = "visualization has too many limitations"; return false; }
    for (const auto & limitation : value.limitations) {
        if (!bounded_text(limitation, 512)) { error = "visualization limitation is invalid"; return false; }
    }
    if (value.complete && value.truncated) {
        error = "truncated visualization cannot be complete"; return false;
    }
    if ((!value.complete || value.truncated) && value.limitations.empty()) {
        error = "incomplete visualization must state at least one limitation"; return false;
    }
    try {
        const auto serialized = common_agent_visualization_relationship_to_json(value).dump();
        if (serialized.size() > COMMON_AGENT_VISUALIZATION_MAX_BYTES) {
            error = "visualization exceeds its serialized byte limit"; return false;
        }
    } catch (const std::exception &) {
        error = "visualization contains invalid text encoding"; return false;
    }
    error.clear(); return true;
}

nlohmann::ordered_json common_agent_visualization_relationship_to_json(
        const common_agent_visualization_relationship & value) {
    nlohmann::ordered_json out = {
        {"schema_version", value.schema_version}, {"kind", value.kind},
        {"title", value.title}, {"data_revision", value.data_revision},
        {"nodes", nlohmann::ordered_json::array()}, {"edges", nlohmann::ordered_json::array()},
        {"complete", value.complete}, {"truncated", value.truncated},
        {"limitations", value.limitations},
        {"view", {{"layout", value.view.layout}, {"selected_nodes", value.view.selected_nodes},
                   {"selected_edges", value.view.selected_edges}}},
    };
    for (const auto & node : value.nodes)
        out["nodes"].push_back({{"id", node.id}, {"label", node.label}, {"kind", node.kind}});
    for (const auto & edge : value.edges)
        out["edges"].push_back({{"id", edge.id}, {"source", edge.source}, {"target", edge.target},
            {"label", edge.label}, {"directed", edge.directed}});
    return out;
}

bool common_agent_visualization_relationship_from_json(const nlohmann::json & input,
        common_agent_visualization_relationship & out, std::string & error) {
    out = {};
    if (!input.is_object()) { error = "visualization must be a JSON object"; return false; }
    static const std::set<std::string> keys{"schema_version", "kind", "title", "data_revision",
        "nodes", "edges", "complete", "truncated", "limitations", "view"};
    for (auto it = input.begin(); it != input.end(); ++it) {
        if (keys.count(it.key()) == 0) { error = "visualization contains an unsupported field"; return false; }
    }
    const auto required = [&](const char * key) { return input.contains(key); };
    if (!required("schema_version") || !required("kind") || !required("title") ||
            !required("nodes") || !required("edges") ||
            (!input["schema_version"].is_number_unsigned() && !input["schema_version"].is_number_integer()) ||
            !input["kind"].is_string() || !input["title"].is_string() ||
            !input["nodes"].is_array() || !input["edges"].is_array() ||
            (required("view") && !input["view"].is_object())) {
        error = "visualization is missing required fields or has invalid JSON types"; return false;
    }
    if (input["nodes"].size() > COMMON_AGENT_VISUALIZATION_MAX_NODES ||
            input["edges"].size() > COMMON_AGENT_VISUALIZATION_MAX_EDGES) {
        error = "visualization exceeds its node or edge limit"; return false;
    }
    try {
        if (input.at("schema_version") != 1) throw std::domain_error("unsupported schema version");
        out.schema_version = 1;
        out.kind = input.at("kind").get<std::string>();
        out.title = input.at("title").get<std::string>();
        if (input.contains("data_revision")) {
            if (!input["data_revision"].is_string()) throw std::domain_error("data_revision must be a string");
            out.data_revision = input["data_revision"].get<std::string>();
        }
        if (input.contains("complete")) {
            if (!input["complete"].is_boolean()) throw std::domain_error("complete must be a boolean");
            out.complete = input["complete"].get<bool>();
        }
        if (input.contains("truncated")) {
            if (!input["truncated"].is_boolean()) throw std::domain_error("truncated must be a boolean");
            out.truncated = input["truncated"].get<bool>();
        }
        if (input.contains("limitations")) {
            if (!input["limitations"].is_array()) throw std::domain_error("limitations must be an array");
            for (const auto & item : input["limitations"]) {
                if (!item.is_string()) throw std::domain_error("limitation must be a string");
                out.limitations.push_back(item.get<std::string>());
            }
        }
        for (const auto & item : input["nodes"]) {
            if (!item.is_object() || !item.contains("id") || !item["id"].is_string() ||
                    (item.contains("label") && !item["label"].is_string()) ||
                    (item.contains("kind") && !item["kind"].is_string()))
                throw std::domain_error("node fields have invalid JSON types");
            for (auto it = item.begin(); it != item.end(); ++it)
                if (it.key() != "id" && it.key() != "label" && it.key() != "kind")
                    throw std::domain_error("node contains an unsupported field");
            out.nodes.push_back({item.at("id").get<std::string>(),
                item.value("label", std::string()), item.value("kind", std::string())});
        }
        for (const auto & item : input["edges"]) {
            if (!item.is_object() || !item.contains("id") || !item["id"].is_string() ||
                    !item.contains("source") || !item["source"].is_string() ||
                    !item.contains("target") || !item["target"].is_string() ||
                    (item.contains("label") && !item["label"].is_string()) ||
                    (item.contains("directed") && !item["directed"].is_boolean()))
                throw std::domain_error("edge fields have invalid JSON types");
            for (auto it = item.begin(); it != item.end(); ++it)
                if (it.key() != "id" && it.key() != "source" && it.key() != "target" &&
                        it.key() != "label" && it.key() != "directed")
                    throw std::domain_error("edge contains an unsupported field");
            out.edges.push_back({item.at("id").get<std::string>(), item.at("source").get<std::string>(),
                item.at("target").get<std::string>(), item.value("label", std::string()),
                item.value("directed", true)});
        }
        const auto view = input.value("view", json::object());
        for (auto it = view.begin(); it != view.end(); ++it)
            if (it.key() != "layout" && it.key() != "selected_nodes" && it.key() != "selected_edges")
                throw std::domain_error("view contains an unsupported field");
        if (view.contains("layout")) {
            if (!view["layout"].is_string()) throw std::domain_error("layout must be a string");
            out.view.layout = view["layout"].get<std::string>();
        }
        const auto read_ids = [&](const char * key, std::vector<std::string> & ids) {
            if (!view.contains(key)) return;
            if (!view[key].is_array()) throw std::domain_error(std::string(key) + " must be an array");
            for (const auto & id : view[key]) {
                if (!id.is_string()) throw std::domain_error(std::string(key) + " entries must be strings");
                ids.push_back(id.get<std::string>());
            }
        };
        read_ids("selected_nodes", out.view.selected_nodes);
        read_ids("selected_edges", out.view.selected_edges);
    } catch (const std::exception & e) {
        error = std::string("invalid visualization: ") + e.what(); return false;
    }
    return common_agent_visualization_relationship_validate(out, error);
}
