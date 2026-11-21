#include "memory/memory-retrieval.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <set>

common_memory_retrieval::common_memory_retrieval(common_memory_store & store, common_memory_retrieval_config config) :
    store(store),
    config(config) {
}

std::vector<common_memory_hit> common_memory_retrieval::retrieve(const common_memory_query & query, std::string & error) {
    common_memory_query backend_query = query;
    backend_query.limit = std::min(config.max_results, std::max(query.limit, query.limit * 2));

    auto hits = store.search(backend_query, error);
    if (!error.empty()) {
        return {};
    }

    const int64_t as_of = query.as_of.value_or(static_cast<int64_t>(std::time(nullptr)));
    std::set<std::string> superseded;
    for (const auto & hit : hits) {
        for (const auto & edge : store.list_relations(hit.memory.id, "supersedes", false, query, error)) {
            if (!error.empty()) return {};
            if (edge.created_at <= as_of) superseded.insert(edge.to);
        }
    }

    std::set<std::string> seen;
    std::vector<common_memory_hit> out;
    for (auto & hit : hits) {
        if (!seen.insert(hit.memory.id).second) {
            continue;
        }
        if (!query.include_history && superseded.count(hit.memory.id) != 0) continue;
        common_memory_temporal_fields temporal;
        if (!common_memory_temporal_fields_read(hit.memory, temporal, error)) return {};
        if (temporal.valid_from && *temporal.valid_from > as_of) continue;
        if (!query.include_history && temporal.valid_to && *temporal.valid_to <= as_of) continue;
        const int64_t recency_time = temporal.observed_at.value_or(hit.memory.created_at);
        const int64_t age = std::max<int64_t>(0, as_of - recency_time);
        hit.recency_score = static_cast<float>(std::exp(-static_cast<double>(age) / (30.0 * 24.0 * 60.0 * 60.0)));
        if (query.include_history && superseded.count(hit.memory.id) != 0) hit.provenance += "; historical: superseded";
        if (!temporal.valid_from && !temporal.valid_to) hit.provenance += "; validity: unknown";
        hit.final_score =
            config.weights.semantic_weight   * hit.semantic_score +
            config.weights.graph_weight      * hit.graph_score +
            config.weights.recency_weight    * hit.recency_score +
            config.weights.importance_weight * hit.memory.importance +
            config.weights.confidence_weight * hit.memory.confidence;
        if (hit.final_score >= query.minimum_score) {
            out.push_back(std::move(hit));
        }
    }

    std::sort(out.begin(), out.end(), [](const common_memory_hit & a, const common_memory_hit & b) {
        if (a.final_score != b.final_score) {
            return a.final_score > b.final_score;
        }
        if (a.semantic_score != b.semantic_score) {
            return a.semantic_score > b.semantic_score;
        }
        return a.memory.id < b.memory.id;
    });
    if (out.size() > query.limit) {
        out.resize(query.limit);
    }
    return out;
}
