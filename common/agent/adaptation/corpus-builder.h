#pragma once

#include "agent/adaptation/training-candidate.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

struct common_learning_corpus_view {
    // Empty fields mean "all". A view is a deterministic projection of the
    // shared corpus; it does not create a second corpus or alter qualification.
    std::string learning_domain;
    std::string tool_family;
    std::set<std::string> provider_kinds;
};

bool common_learning_corpus_view_matches(
        const common_training_candidate & candidate,
        const common_learning_corpus_view & view);

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
    common_learning_corpus_view view;
};

struct common_learning_corpus_revision {
    int schema_version = 1;
    std::string id;
    std::string builder_version;
    uint64_t seed = 0;
    std::vector<std::string> candidate_ids;
    std::vector<std::string> replay_candidate_ids;
    std::vector<std::string> held_out_candidate_ids;
    common_learning_corpus_view view;
    std::string jsonl;
    std::string manifest_json;
    std::string bundle_hash;
};

struct common_learning_corpus_inspection {
    size_t row_count = 0;
    size_t byte_count = 0;
    bool truncated = false;
    std::map<std::string, size_t> domains;
    std::map<std::string, size_t> families;
    std::map<std::string, size_t> providers;
    std::vector<std::string> candidate_ids;
};

bool common_learning_build_corpus(
        const std::vector<common_training_candidate> & candidates,
        const common_learning_corpus_policy & policy,
        common_learning_corpus_revision & revision,
        std::string & error);

bool common_learning_inspect_corpus(
        const common_learning_corpus_revision & revision,
        size_t max_rows,
        common_learning_corpus_inspection & inspection,
        std::string & error);

bool common_learning_export_corpus_jsonl(
        const common_learning_corpus_revision & revision,
        const std::filesystem::path & path,
        size_t max_bytes,
        std::string & error);
