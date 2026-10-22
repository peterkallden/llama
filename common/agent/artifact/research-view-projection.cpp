#include "research-view-projection.h"

#include "hash/hash.h"

#include <algorithm>
#include <map>
#include <set>

namespace {
using json = nlohmann::json;

std::string stable_id(const std::string & prefix, const std::string & value) {
    const auto digest = hash_sha256_hex(value.data(), value.size());
    return prefix + ":" + digest.substr(0, 48);
}

std::string limited_label(const std::string & text) {
    if (text.size() <= 500) return text;
    size_t end = 500;
    while (end > 0 && end < text.size() &&
            (static_cast<unsigned char>(text[end]) & 0xc0u) == 0x80u) --end;
    return text.substr(0, end);
}

struct graph_builder {
    common_agent_visualization_relationship & out;
    std::set<std::string> ids;
    std::set<std::string> edge_ids;
    size_t node_limit = COMMON_AGENT_VISUALIZATION_MAX_NODES;
    size_t edge_limit = COMMON_AGENT_VISUALIZATION_MAX_EDGES;

    bool node(const std::string & id, const std::string & label, const std::string & kind) {
        if (ids.count(id)) return true;
        if (out.nodes.size() >= node_limit) { truncate("research graph reached its node limit"); return false; }
        out.nodes.push_back({id, limited_label(label), kind}); ids.insert(id); return true;
    }
    void edge(const std::string & source, const std::string & target, const std::string & relation,
            const std::string & label, std::vector<std::string> provenance) {
        if (!ids.count(source) || !ids.count(target)) return;
        const std::string key = source + "\n" + target + "\n" + relation + "\n" +
            (provenance.empty() ? std::string{} : provenance.front());
        const auto id = stable_id("relation", key);
        if (edge_ids.count(id)) return;
        if (out.edges.size() >= edge_limit) { truncate("research graph reached its relation limit"); return; }
        common_agent_visualization_edge item{id, source, target, limited_label(label), true};
        item.relation = relation; item.provenance_refs = std::move(provenance);
        out.edges.push_back(std::move(item)); edge_ids.insert(id);
    }
    void truncate(const std::string & reason) {
        out.complete = false; out.truncated = true;
        if (std::find(out.limitations.begin(), out.limitations.end(), reason) == out.limitations.end())
            out.limitations.push_back(reason);
    }
};

const char * evidence_relation(common_agent_research_evidence_relation relation) {
    switch (relation) {
        case common_agent_research_evidence_relation::supports: return "supports";
        case common_agent_research_evidence_relation::contradicts: return "contradicts";
        case common_agent_research_evidence_relation::qualifies: return "qualifies";
        case common_agent_research_evidence_relation::contextualizes: return "contextualizes";
        case common_agent_research_evidence_relation::inconclusive: return "inconclusive";
    }
    return "inconclusive";
}

bool build_graph(const std::string & title, const std::string & revision,
        const std::vector<common_agent_artifact_block> & blocks,
        const std::vector<common_agent_research_source> & sources,
        const std::vector<common_agent_research_evidence> & evidence,
        const std::vector<common_agent_research_gap> & gaps,
        common_agent_visualization_relationship & out, std::string & error) {
    out = {}; out.title = title; out.data_revision = revision;
    graph_builder graph{out};
    std::map<std::string, const common_agent_research_source *> source_by_id;
    std::map<std::string, const common_agent_research_evidence *> evidence_by_id;
    for (const auto & source : sources) source_by_id.emplace(source.source_id, &source);
    for (const auto & item : evidence) evidence_by_id.emplace(item.evidence_id, &item);

    std::set<std::string> relevant_evidence;
    if (blocks.empty()) {
        // Active-research mode projects the current draft directly. In that
        // mode every evidence item belongs to the selected workspace view.
        for (const auto & item : evidence) {
            relevant_evidence.insert(item.evidence_id);
            graph.node(stable_id("evidence", item.evidence_id), item.statement, "evidence");
        }
    }
    for (const auto & block : blocks) {
        const auto block_node = stable_id("block", block.id);
        if (!graph.node(block_node, block.text, "report_block")) continue;
        for (const auto & evidence_id : block.evidence_ids) {
            const auto found = evidence_by_id.find(evidence_id);
            if (found == evidence_by_id.end()) {
                graph.truncate("a report block references evidence absent from its frozen snapshot");
                continue;
            }
            relevant_evidence.insert(evidence_id);
            const auto evidence_node = stable_id("evidence", evidence_id);
            if (graph.node(evidence_node, found->second->statement, "evidence"))
                graph.edge(block_node, evidence_node, "references_evidence", "evidence", {evidence_id});
            if (!block.claim_id.empty()) {
                const auto claim_node = stable_id("claim", block.claim_id);
                if (graph.node(claim_node, block.claim_id, "claim"))
                    graph.edge(block_node, claim_node, "asserts", "claim anchor", {block.id, block.claim_id});
            }
        }
    }
    for (const auto & id : relevant_evidence) {
        const auto & item = *evidence_by_id.at(id);
        const auto ev = stable_id("evidence", id);
        if (!item.claim_id.empty()) {
            const auto claim = stable_id("claim", item.claim_id);
            if (graph.node(claim, item.claim_id, "claim"))
                graph.edge(ev, claim, evidence_relation(item.relation), evidence_relation(item.relation), {id});
        }
        const auto source = source_by_id.find(item.source_id);
        if (source == source_by_id.end()) {
            graph.truncate("evidence has no source record in the frozen snapshot");
            continue;
        }
        const auto src = stable_id("source", item.source_id);
        const std::string label = source->second->title.empty() ? item.source_id : source->second->title;
        if (graph.node(src, label, "source"))
            graph.edge(ev, src, "extracted_from", item.source_location, {id, item.source_id});
    }
    for (const auto & gap : gaps) {
        std::vector<std::string> linked;
        for (const auto & id : gap.evidence_ids) if (relevant_evidence.count(id)) linked.push_back(id);
        if (linked.empty()) continue;
        const auto gap_node = stable_id("gap", gap.gap_id);
        if (!graph.node(gap_node, gap.question, "research_gap")) continue;
        for (const auto & id : linked) graph.edge(gap_node, stable_id("evidence", id), "addresses", "gap evidence", {gap.gap_id, id});
    }
    if (out.truncated && out.limitations.empty()) out.limitations.push_back("research graph is incomplete");
    if (!common_agent_visualization_relationship_validate(out, error)) return false;
    return true;
}
}

bool common_agent_research_report_evidence_graph(
        const common_agent_artifact_document & document,
        const common_agent_artifact_snapshot & snapshot,
        const std::vector<std::string> & selected_block_ids,
        common_agent_visualization_relationship & out,
        std::string & error) {
    if (document.id.empty() || document.revision == 0 || document.snapshot_fingerprint.empty() ||
            document.snapshot_fingerprint != snapshot.fingerprint) {
        error = "report and research snapshot revision do not match"; return false;
    }
    std::set<std::string> requested(selected_block_ids.begin(), selected_block_ids.end());
    if (requested.size() != selected_block_ids.size()) { error = "report selection contains duplicate block ids"; return false; }
    std::vector<common_agent_artifact_block> selected;
    std::set<std::string> found;
    for (const auto & section : document.sections) for (const auto & block : section.blocks) {
        if (requested.empty() || requested.count(block.id)) {
            selected.push_back(block); found.insert(block.id);
        }
    }
    if (selected.empty() || found.size() != requested.size()) { error = "report selection is empty or contains an unknown block"; return false; }
    const std::string report_revision_material = json::array({document.id, document.revision, snapshot.fingerprint}).dump();
    const std::string report_revision = "report:" + hash_sha256_hex(
        report_revision_material.data(), report_revision_material.size()).substr(0, 48);
    return build_graph(document.title, report_revision,
        selected, snapshot.research_sources, snapshot.research_evidence, snapshot.research_gaps, out, error);
}

bool common_agent_research_workspace_evidence_graph(
        const common_agent_research_workspace & workspace,
        common_agent_visualization_relationship & out,
        std::string & error) {
    if (!common_agent_research_workspace_validate(workspace, error)) return false;
    const auto revision = workspace.draft_id.empty()
        ? workspace.research_id + "@" + std::to_string(workspace.revision)
        : workspace.research_id + "@draft:" + workspace.draft_id + ":" + std::to_string(workspace.base_revision);
    const auto revision_hash = hash_sha256_hex(revision.data(), revision.size());
    if (!build_graph(workspace.objective.question, "research:" + revision_hash.substr(0, 48), {}, workspace.sources,
            workspace.evidence, workspace.gaps, out, error)) return false;
    graph_builder graph{out};
    for (const auto & node : out.nodes) graph.ids.insert(node.id);
    for (const auto & edge : out.edges) graph.edge_ids.insert(edge.id);
    const auto objective = stable_id("objective", workspace.objective.objective_id);
    graph.node(objective, workspace.objective.question, "research_objective");
    const auto status_name = [](common_agent_research_gap_status status) {
        switch (status) {
            case common_agent_research_gap_status::open: return "open";
            case common_agent_research_gap_status::investigating: return "in progress";
            case common_agent_research_gap_status::sufficiently_answered: return "resolved";
            case common_agent_research_gap_status::contradicted: return "contradicted";
            case common_agent_research_gap_status::blocked: return "blocked";
            case common_agent_research_gap_status::abandoned: return "abandoned";
        }
        return "unknown";
    };
    const auto assessment_name = [](common_agent_research_gap_assessment assessment) {
        switch (assessment) {
            case common_agent_research_gap_assessment::unknown: return "assessment unknown";
            case common_agent_research_gap_assessment::sufficient: return "evidence sufficient";
            case common_agent_research_gap_assessment::insufficient: return "evidence insufficient";
            case common_agent_research_gap_assessment::inconclusive: return "evidence inconclusive";
            case common_agent_research_gap_assessment::contradicted: return "evidence contradicted";
        }
        return "assessment unknown";
    };
    for (const auto & gap : workspace.gaps) {
        const auto id = stable_id("gap", gap.gap_id);
        graph.node(id, std::string("[") + status_name(gap.status) + "; " + assessment_name(gap.assessment) + "] " + gap.question, "research_gap");
        graph.edge(objective, id, "tracks_gap", status_name(gap.status) + std::string(" · ") + assessment_name(gap.assessment), {gap.gap_id});
    }
    return common_agent_visualization_relationship_validate(out, error);
}
