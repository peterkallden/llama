#include "memory/memory-retrieval.h"

#include <algorithm>
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

    std::set<std::string> seen;
    std::vector<common_memory_hit> out;
    for (auto & hit : hits) {
        if (!seen.insert(hit.memory.id).second) {
            continue;
        }
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
