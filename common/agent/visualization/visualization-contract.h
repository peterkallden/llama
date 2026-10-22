#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <vector>

// Domain-neutral, bounded relationship visualization input. Semantic meaning
// and authorization remain owned by the producer of this projection.
struct common_agent_visualization_node {
    std::string id;
    std::string label;
    std::string kind;
};

struct common_agent_visualization_edge {
    std::string id;
    std::string source;
    std::string target;
    std::string label;
    bool directed = true;
    std::string relation;
    std::vector<std::string> provenance_refs;
};

struct common_agent_visualization_view {
    std::string layout = "cose";
    std::vector<std::string> selected_nodes;
    std::vector<std::string> selected_edges;
};

struct common_agent_visualization_relationship {
    unsigned schema_version = 1;
    std::string kind = "relationship";
    std::string title;
    std::string data_revision;
    std::vector<common_agent_visualization_node> nodes;
    std::vector<common_agent_visualization_edge> edges;
    bool complete = true;
    bool truncated = false;
    std::vector<std::string> limitations;
    common_agent_visualization_view view;
};

inline constexpr size_t COMMON_AGENT_VISUALIZATION_MAX_BYTES = 1024 * 1024;
inline constexpr size_t COMMON_AGENT_VISUALIZATION_MAX_NODES = 500;
inline constexpr size_t COMMON_AGENT_VISUALIZATION_MAX_EDGES = 1000;

bool common_agent_visualization_relationship_validate(
    const common_agent_visualization_relationship &, std::string & error);
bool common_agent_visualization_relationship_from_json(
    const nlohmann::json &, common_agent_visualization_relationship &, std::string & error);
nlohmann::ordered_json common_agent_visualization_relationship_to_json(
    const common_agent_visualization_relationship &);
