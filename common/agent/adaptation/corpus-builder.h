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
    size_t validation_percent = 10;
    size_t test_percent = 10;
    size_t max_replay_candidates = 0;
    std::string redaction_policy_id = "stub:caller-asserted-v1";
    std::vector<std::string> held_out_candidate_ids;
    std::vector<std::string> revoked_candidate_ids;
    // Selected by a host/ledger query.  The builder only bounds and records
    // this selection; it does not invent semantic similarity by itself.
    std::vector<std::string> replay_candidate_ids;
};

struct common_learning_corpus_revision {
    int schema_version = 1;
    std::string id;
    std::string builder_version;
    uint64_t seed = 0;
    std::vector<std::string> candidate_ids;
    std::vector<std::string> replay_candidate_ids;
    std::vector<std::string> held_out_candidate_ids;
    std::string jsonl;
    std::string manifest_json;
    std::string bundle_hash;
};

bool common_learning_build_corpus(
        const std::vector<common_training_candidate> & candidates,
        const common_learning_corpus_policy & policy,
        common_learning_corpus_revision & revision,
        std::string & error);
