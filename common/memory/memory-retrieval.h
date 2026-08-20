#pragma once

#include "memory/memory-store.h"

struct common_memory_retrieval_weights {
    float semantic_weight   = 0.45f;
    float graph_weight      = 0.20f;
    float recency_weight    = 0.15f;
    float importance_weight = 0.15f;
    float confidence_weight = 0.05f;
};

struct common_memory_retrieval_config {
    common_memory_retrieval_weights weights;
    size_t max_results = 20;
};

class common_memory_retrieval {
public:
    explicit common_memory_retrieval(common_memory_store & store, common_memory_retrieval_config config = {});

    std::vector<common_memory_hit> retrieve(const common_memory_query & query, std::string & error);

private:
    common_memory_store & store;
    common_memory_retrieval_config config;
};
