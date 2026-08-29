#include "agent/adaptation/corpus-builder.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::abort();
    }
}

static common_training_candidate candidate(const std::string & id, const std::string & prompt) {
    common_training_candidate value;
    value.id = id;
    value.transaction_ids = {"learning://transaction/" + prompt + "/1",
                             "learning://transaction/" + prompt + "/2",
                             "learning://transaction/" + prompt + "/3"};
    value.cause = common_learning_cause::model_behavior;
    value.hypothesis = "A stable tool contract is bound incorrectly.";
    value.approved_prompt = prompt;
    value.approved_target = "Emit the valid tool binding.";
    value.observed_occurrences = 3;
    value.verified_recoveries = 2;
    value.confidence = 0.9f;
    value.redaction_policy_id = "stub:caller-asserted-v1";
    value.redaction_method = "caller_asserted";
    value.redaction_status = common_learning_redaction_status::caller_asserted;
    value.status = common_training_candidate_status::approved;
    return value;
}

int main() {
    std::vector<common_training_candidate> candidates = {
        candidate("learning://candidate/b", "second"),
        candidate("learning://candidate/a", "first"),
    };
    common_learning_corpus_revision first;
    std::string error;
    require(common_learning_build_corpus(candidates, {}, first, error), "initial corpus build failed");
    common_learning_corpus_revision second;
    require(common_learning_build_corpus(candidates, {}, second, error), "second corpus build failed");
    require(first.jsonl == second.jsonl, "corpus JSONL is not deterministic");
    require(first.manifest_json == second.manifest_json, "corpus manifest is not deterministic");
    require(first.bundle_hash == second.bundle_hash, "corpus hash is not deterministic");
    require(first.candidate_ids.size() == 2 && first.candidate_ids.front() == "learning://candidate/a",
            "corpus ordering is not stable");
    require(first.jsonl.find("\"input\":\"first\"") != std::string::npos, "corpus input is missing");
    require(first.jsonl.find("\"redaction\"") != std::string::npos, "corpus redaction metadata is missing");
    const auto manifest = nlohmann::json::parse(first.manifest_json);
    require(manifest["transaction_ids"].size() == 6, "corpus provenance is incomplete");
    require(manifest["split_counts"].is_object(), "corpus split counts are missing");
    auto reordered = candidates;
    for (auto & value : reordered) std::reverse(value.transaction_ids.begin(), value.transaction_ids.end());
    common_learning_corpus_revision reordered_revision;
    require(common_learning_build_corpus(reordered, {}, reordered_revision, error), "reordered corpus build failed");
    require(reordered_revision.bundle_hash == first.bundle_hash, "transaction order changed corpus identity");

    candidates.front().status = common_training_candidate_status::observed;
    require(!common_learning_build_corpus(candidates, {}, first, error), "unapproved candidate was admitted");
    candidates.front().status = common_training_candidate_status::approved;
    candidates.front().transaction_ids = candidates.back().transaction_ids;
    require(!common_learning_build_corpus(candidates, {}, first, error), "overlapping transactions were admitted");
    candidates.front().id = candidates.back().id;
    candidates.front().transaction_ids = {"learning://transaction/4", "learning://transaction/5", "learning://transaction/6"};
    require(!common_learning_build_corpus(candidates, {}, first, error), "conflicting duplicate was admitted");

    candidates.front().id = "learning://candidate/c";
    candidates.front().transaction_ids = {"learning://transaction/4", "learning://transaction/5", "learning://transaction/6"};
    auto duplicate = candidates;
    duplicate.push_back(candidates.front());
    require(common_learning_build_corpus(duplicate, {}, first, error), "identical duplicate was not deduplicated");
    require(first.candidate_ids.size() == 2, "identical duplicate changed corpus cardinality");

    auto held_out_policy = common_learning_corpus_policy{};
    held_out_policy.held_out_candidate_ids = {"learning://candidate/a"};
    require(common_learning_build_corpus(candidates, held_out_policy, first, error), "held-out corpus build failed");
    require(first.candidate_ids.size() == 1 && first.held_out_candidate_ids.size() == 1,
            "held-out candidate was not excluded or recorded");
    require(first.jsonl.find("learning://candidate/a") == std::string::npos, "held-out candidate leaked into JSONL");

    auto replay_policy = common_learning_corpus_policy{};
    replay_policy.max_replay_candidates = 1;
    replay_policy.replay_candidate_ids = {"learning://candidate/c"};
    require(common_learning_build_corpus(candidates, replay_policy, first, error), "bounded replay build failed");
    require(first.replay_candidate_ids == replay_policy.replay_candidate_ids, "replay selection was not recorded");
    require(first.jsonl.find("\"replay\":true") != std::string::npos, "replay row was not marked");

    auto rejected_redaction = candidates;
    rejected_redaction.front().redaction_status = common_learning_redaction_status::not_evaluated;
    require(!common_learning_build_corpus(rejected_redaction, {}, first, error),
            "unredacted candidate was admitted");
    auto revoked = candidates;
    revoked.front().status = common_training_candidate_status::revoked;
    require(!common_learning_build_corpus(revoked, {}, first, error), "revoked candidate was admitted");
    auto replay_held_out = replay_policy;
    replay_held_out.held_out_candidate_ids = {"learning://candidate/c"};
    require(!common_learning_build_corpus(candidates, replay_held_out, first, error),
            "held-out candidate was selected for replay");
    auto invalid_split = common_learning_corpus_policy{};
    invalid_split.validation_percent = 75;
    invalid_split.test_percent = 26;
    require(!common_learning_build_corpus(candidates, invalid_split, first, error), "invalid split was accepted");
    auto too_large_replay = replay_policy;
    too_large_replay.replay_candidate_ids = {"learning://candidate/a", "learning://candidate/c"};
    require(!common_learning_build_corpus(candidates, too_large_replay, first, error), "replay bound was ignored");
    return 0;
}
