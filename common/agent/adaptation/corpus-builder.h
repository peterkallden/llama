#pragma once

#include "agent/adaptation/training-candidate.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct common_learning_corpus_policy {
    std::string builder_version = "1";
    uint64_t seed = 42;
    size_t max_candidates = 1024;
    size_t max_bytes = 4 * 1024 * 1024;
};

struct common_learning_corpus_revision {
    int schema_version = 1;
    std::string id;
    std::string builder_version;
    uint64_t seed = 0;
    std::vector<std::string> candidate_ids;
    std::string jsonl;
    std::string manifest_json;
    std::string bundle_hash;
};

bool common_learning_build_corpus(
        const std::vector<common_training_candidate> & candidates,
        const common_learning_corpus_policy & policy,
        common_learning_corpus_revision & revision,
        std::string & error);
