#include "agent/adaptation/corpus-builder.h"

#include <cassert>

static common_training_candidate candidate(const std::string & id, const std::string & prompt) {
    common_training_candidate value;
    value.id = id;
    value.transaction_ids = {"learning://transaction/1", "learning://transaction/2", "learning://transaction/3"};
    value.cause = common_learning_cause::model_behavior;
    value.hypothesis = "A stable tool contract is bound incorrectly.";
    value.approved_prompt = prompt;
    value.approved_target = "Emit the valid tool binding.";
    value.observed_occurrences = 3;
    value.verified_recoveries = 2;
    value.confidence = 0.9f;
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
    assert(common_learning_build_corpus(candidates, {}, first, error));
    common_learning_corpus_revision second;
    assert(common_learning_build_corpus(candidates, {}, second, error));
    assert(first.jsonl == second.jsonl);
    assert(first.manifest_json == second.manifest_json);
    assert(first.bundle_hash == second.bundle_hash);
    assert(first.candidate_ids.size() == 2 && first.candidate_ids.front() == "learning://candidate/a");
    assert(first.jsonl.find("\"input\":\"first\"") != std::string::npos);

    candidates.front().status = common_training_candidate_status::observed;
    assert(!common_learning_build_corpus(candidates, {}, first, error));
    candidates.front().status = common_training_candidate_status::approved;
    candidates.front().transaction_ids = candidates.back().transaction_ids;
    assert(!common_learning_build_corpus(candidates, {}, first, error));
    candidates.front().id = candidates.back().id;
    candidates.front().transaction_ids = {"learning://transaction/4", "learning://transaction/5", "learning://transaction/6"};
    assert(!common_learning_build_corpus(candidates, {}, first, error));
    return 0;
}
