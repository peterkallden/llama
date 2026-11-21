#include "agent/adaptation/flydelta/flydelta-training.h"

#include <string>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_training_example example(
        const std::string & id,
        const std::string & evidence,
        common_flydelta_training_split split) {
    common_flydelta_training_example value;
    value.id = id;
    value.behavior_key = "tool_use/diagnostics/missing-argument";
    value.context_fingerprint = "sha256:" + id;
    value.basis_revision = "flydelta://basis/corpus";
    value.evidence_ref = evidence;
    value.split = split;
    value.context.expansion_dim = 8;
    value.context.indices = {1};
    value.context.values = {1.0f};
    value.target_coefficients = {0.25f, -0.1f};
    value.outcome = common_flydelta_counterfactual_outcome::helped;
    value.confidence = 1.0f;
    return value;
}

int main() {
    const common_flydelta_memory_config memory{8, 2, 1.0f};
    common_flydelta_training_corpus corpus;
    corpus.id = "flydelta://corpus/1";
    corpus.basis_revision = "flydelta://basis/corpus";
    corpus.train.push_back(example("train", "evidence://train", common_flydelta_training_split::train));
    corpus.validation.push_back(example("validation", "evidence://validation", common_flydelta_training_split::validation));
    corpus.holdout.push_back(example("holdout", "evidence://holdout", common_flydelta_training_split::holdout));

    std::string error;
    CHECK(common_flydelta_training_corpus_validate(corpus, memory, 3, error));
    CHECK(!common_flydelta_training_corpus_validate(corpus, memory, 2, error));

    corpus.validation[0].split = common_flydelta_training_split::train;
    CHECK(!common_flydelta_training_corpus_validate(corpus, memory, 3, error));
    corpus.validation[0].split = common_flydelta_training_split::validation;
    corpus.holdout[0].id = corpus.train[0].id;
    CHECK(!common_flydelta_training_corpus_validate(corpus, memory, 3, error));
    corpus.holdout[0].id = "holdout";
    corpus.holdout[0].evidence_ref = corpus.train[0].evidence_ref;
    CHECK(!common_flydelta_training_corpus_validate(corpus, memory, 3, error));
    return 0;
}
