#include "memory/memory-in-memory.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <set>
#include <sstream>

static std::string lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

static float text_score(const std::string & query, const common_memory_record & record) {
    if (query.empty()) {
        return 0.0f;
    }

    const std::string q = lowercase(query);
    const std::string hay = lowercase(record.content + " " + record.summary);
    std::istringstream iss(q);
    std::string term;
    size_t terms = 0;
    size_t matches = 0;
    while (iss >> term) {
        terms++;
        if (hay.find(term) != std::string::npos) {
            matches++;
        }
    }
    return terms == 0 ? 0.0f : (float) matches / (float) terms;
}

const char * common_memory_kind_name(common_memory_kind kind) {
    switch (kind) {
        case common_memory_kind::episode:     return "episode";
        case common_memory_kind::fact:        return "fact";
        case common_memory_kind::observation: return "observation";
        case common_memory_kind::reflection:  return "reflection";
        case common_memory_kind::procedure:   return "procedure";
        case common_memory_kind::constraint:  return "constraint";
        case common_memory_kind::decision:    return "decision";
        case common_memory_kind::goal:        return "goal";
        case common_memory_kind::preference:  return "preference";
    }
    return "episode";
}

bool common_memory_kind_parse(const std::string & value, common_memory_kind & out) {
    const std::string v = lowercase(value);
    if (v == "episode")     { out = common_memory_kind::episode; return true; }
    if (v == "fact")        { out = common_memory_kind::fact; return true; }
    if (v == "observation") { out = common_memory_kind::observation; return true; }
    if (v == "reflection")  { out = common_memory_kind::reflection; return true; }
    if (v == "procedure")   { out = common_memory_kind::procedure; return true; }
    if (v == "constraint")  { out = common_memory_kind::constraint; return true; }
    if (v == "decision")    { out = common_memory_kind::decision; return true; }
    if (v == "goal")        { out = common_memory_kind::goal; return true; }
    if (v == "preference")  { out = common_memory_kind::preference; return true; }
    return false;
}

const char * common_memory_scope_name(common_memory_scope scope) {
    switch (scope) {
        case common_memory_scope::turn:    return "turn";
        case common_memory_scope::session: return "session";
        case common_memory_scope::project: return "project";
        case common_memory_scope::global:  return "global";
    }
    return "session";
}

bool common_memory_scope_parse(const std::string & value, common_memory_scope & out) {
    const std::string v = lowercase(value);
    if (v == "turn")    { out = common_memory_scope::turn; return true; }
    if (v == "session") { out = common_memory_scope::session; return true; }
    if (v == "project") { out = common_memory_scope::project; return true; }
    if (v == "global")  { out = common_memory_scope::global; return true; }
    return false;
}

bool common_memory_scope_matches(const common_memory_record & record, const common_memory_query & query) {
    if (record.scope != query.scope || record.namespace_id != query.namespace_id) {
        return false;
    }
    switch (query.scope) {
        case common_memory_scope::turn:
            return !query.turn_id.empty() && record.turn_id == query.turn_id;
        case common_memory_scope::session:
            return !query.session_id.empty() && record.session_id == query.session_id;
        case common_memory_scope::project:
            return !query.project_id.empty() && record.project_id == query.project_id;
        case common_memory_scope::global:
            return query.global_opt_in;
    }
    return false;
}

float common_memory_cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0f;
    }

    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return 0.0f;
        }
        dot += (double) a[i] * (double) b[i];
        na += (double) a[i] * (double) a[i];
        nb += (double) b[i] * (double) b[i];
    }
    if (na <= std::numeric_limits<double>::epsilon() || nb <= std::numeric_limits<double>::epsilon()) {
        return 0.0f;
    }
    return (float) (dot / (std::sqrt(na) * std::sqrt(nb)));
}

bool common_memory_in_memory_store::open(const std::string &, std::string & error) {
    error.clear();
    opened = true;
    return true;
}

void common_memory_in_memory_store::close() {
    opened = false;
}

bool common_memory_in_memory_store::put(const common_memory_record & record, std::string & error) {
    if (!opened) {
        error = "memory store is not open";
        return false;
    }
    if (record.id.empty()) {
        error = "memory id must not be empty";
        return false;
    }
    records[record.id] = record;
    error.clear();
    return true;
}

std::optional<common_memory_record> common_memory_in_memory_store::get(const std::string & id, std::string & error) {
    if (!opened) {
        error = "memory store is not open";
        return std::nullopt;
    }
    error.clear();
    auto it = records.find(id);
    if (it == records.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::vector<common_memory_record> common_memory_in_memory_store::list(const common_memory_query & query, std::string & error) {
    if (!opened) { error = "memory store is not open"; return {}; }
    std::vector<common_memory_record> result;
    for (const auto & entry : records) if ((!query.kind || entry.second.kind == *query.kind) && common_memory_scope_matches(entry.second, query)) result.push_back(entry.second);
    error.clear();
    return result;
}

std::vector<common_memory_hit> common_memory_in_memory_store::search(const common_memory_query & query, std::string & error) {
    if (!opened) {
        error = "memory store is not open";
        return {};
    }
    error.clear();

    std::map<std::string, float> graph_scores;
    if (query.graph_depth > 0) {
        for (const auto & edge : edges) {
            if (records.find(edge.from) == records.end() || records.find(edge.to) == records.end()) {
                continue;
            }
            const auto & from = records.at(edge.from);
            const auto & to = records.at(edge.to);
            const float from_score = std::max(text_score(query.text, from), common_memory_cosine_similarity(query.embedding, from.embedding));
            const float to_score = std::max(text_score(query.text, to), common_memory_cosine_similarity(query.embedding, to.embedding));
            if (from_score > 0.0f) {
                graph_scores[edge.to] = std::max(graph_scores[edge.to], std::max(0.0f, edge.weight) * 0.5f);
            }
            if (to_score > 0.0f) {
                graph_scores[edge.from] = std::max(graph_scores[edge.from], std::max(0.0f, edge.weight) * 0.5f);
            }
        }
    }

    std::vector<common_memory_hit> hits;
    for (const auto & kv : records) {
        const common_memory_record & record = kv.second;
        if (query.kind && record.kind != *query.kind) {
            continue;
        }
        if (!common_memory_scope_matches(record, query)) {
            continue;
        }

        const float semantic = !query.embedding.empty()
            ? common_memory_cosine_similarity(query.embedding, record.embedding)
            : text_score(query.text, record);
        const float graph = graph_scores[kv.first];
        const float score = std::max(semantic, graph);
        if (score < query.minimum_score) {
            continue;
        }

        common_memory_hit hit;
        hit.memory = record;
        hit.semantic_score = semantic;
        hit.graph_score = graph;
        hit.final_score = score;
        hit.provenance = graph > semantic ? "in-memory graph expansion" : "in-memory direct search";
        hits.push_back(std::move(hit));
    }

    std::sort(hits.begin(), hits.end(), [](const common_memory_hit & a, const common_memory_hit & b) {
        if (a.final_score != b.final_score) {
            return a.final_score > b.final_score;
        }
        return a.memory.id < b.memory.id;
    });
    if (hits.size() > query.limit) {
        hits.resize(query.limit);
    }
    return hits;
}

bool common_memory_in_memory_store::relate(const std::string & from, const std::string & relation, const std::string & to, float weight, std::string & error) {
    if (!opened) {
        error = "memory store is not open";
        return false;
    }
    if (records.find(from) == records.end() || records.find(to) == records.end()) {
        error = "cannot create relation for unknown memory id";
        return false;
    }
    if (relation.empty()) {
        error = "relation must not be empty";
        return false;
    }
    edges.erase(std::remove_if(edges.begin(), edges.end(), [&](const edge & e) {
        return e.from == from && e.relation == relation && e.to == to;
    }), edges.end());
    edges.push_back(edge{from, relation, to, weight, 0});
    std::sort(edges.begin(), edges.end(), [](const edge & a, const edge & b) {
        return std::tie(a.from, a.relation, a.to) < std::tie(b.from, b.relation, b.to);
    });
    error.clear();
    return true;
}

bool common_memory_in_memory_store::erase(const std::string & id, std::string & error) {
    if (!opened) {
        error = "memory store is not open";
        return false;
    }
    records.erase(id);
    edges.erase(std::remove_if(edges.begin(), edges.end(), [&](const edge & e) {
        return e.from == id || e.to == id;
    }), edges.end());
    error.clear();
    return true;
}
