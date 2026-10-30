#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Project-owned identities and proposals. Source-specific evidence stays in
// the originating provider; these records retain references to that evidence.
struct common_context_graph_node {
    std::string id;
    std::string kind;
    std::string name;
    std::string path;
    int64_t line = 0;
    std::string source_revision;
    // Opaque source-owned identifier; it conveys no authority to read it.
    std::string source_ref;
};

struct common_context_graph_edge {
    std::string id;
    std::string from;
    std::string to;
    std::string relation;
    std::string status = "proposed";
    std::string source = "model";
    std::string rationale;
    std::vector<std::string> evidence_refs;
    std::string source_revision;
    uint64_t version = 1;
};

struct common_context_graph_state {
    uint64_t revision = 0;
    std::vector<common_context_graph_node> nodes;
    std::vector<common_context_graph_edge> edges;
};
