#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Project-owned identities and proposals. Source-specific evidence stays in
// the originating provider; these records retain references to that evidence.
struct common_context_graph_node_observation {
    std::string path;
    int64_t line = 0;
    std::string source_revision;
    std::string source_ref;
};

struct common_context_graph_description {
    std::string text;
    std::string source = "model";
    std::string source_revision;
    std::vector<std::string> evidence_refs;
};

struct common_context_graph_node {
    std::string id;
    std::string kind;
    std::string name;
    std::string path;
    int64_t line = 0;
    std::string source_revision;
    // Opaque source-owned identifier; it conveys no authority to read it.
    std::string source_ref;
    // An opaque, provider-qualified symbol identifier, not a source location.
    std::string symbol_key;
    std::vector<common_context_graph_node_observation> observations;
    // Append-only proposed descriptions; the last entry is the current proposal.
    std::vector<common_context_graph_description> descriptions;
};

struct common_context_graph_assertion {
    std::string source = "model";
    std::string rationale;
    std::vector<std::string> evidence_refs;
    std::string source_revision;
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
    // A pending withdrawal is separate from the original assertion rationale.
    std::string retraction_rationale;
    std::vector<common_context_graph_assertion> assertions;
};

struct common_context_graph_state {
    uint64_t revision = 0;
    std::vector<common_context_graph_node> nodes;
    std::vector<common_context_graph_edge> edges;
};
