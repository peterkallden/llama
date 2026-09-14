#pragma once

#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// The split is assigned by the host using a behavior/contract family, not by
// randomly shuffling individual prompts. This keeps near-duplicate tool
// schemas out of validation and holdout by accident.
enum class common_flydelta_training_split {
    train,
    validation,
    holdout,
};

const char * common_flydelta_training_split_name(common_flydelta_training_split split);
bool common_flydelta_training_split_from_name(
        const std::string & value,
        common_flydelta_training_split & split);

struct common_flydelta_alpha_search_config {
    std::vector<float> candidates;
    float magnitude_penalty = 0.0f;
    size_t max_candidates = 32;
};

struct common_flydelta_alpha_trial {
    float alpha = 0.0f;
    common_flydelta_counterfactual_outcome outcome = common_flydelta_counterfactual_outcome::unknown;
    float quality_delta = 0.0f;
    bool executed = false;
    bool verifier_known = false;
    std::string evidence_ref;
};

struct common_flydelta_alpha_selection {
    bool selected = false;
    float alpha = 0.0f;
    float score = 0.0f;
    size_t trial_index = 0;
};

struct common_flydelta_training_example;

bool common_flydelta_alpha_search_config_validate(
        const common_flydelta_alpha_search_config & config,
        std::string & error);
bool common_flydelta_alpha_trial_validate(
        const common_flydelta_alpha_trial & trial,
        std::string & error);

// Chooses the smallest intervention with the best penalized verified lift.
// HARMED and UNKNOWN trials can never be selected. No selection is a valid
// result when no candidate produces a host-verified improvement.
bool common_flydelta_select_alpha(
        const common_flydelta_alpha_search_config & config,
        const std::vector<common_flydelta_alpha_trial> & trials,
        common_flydelta_alpha_selection & selection,
        std::string & error);

// Runs a baseline plus each configured candidate on the same immutable
// fixture. The host callback owns inference and verification; this helper
// only classifies the resulting candidate against the baseline and applies
// the existing conservative alpha selector.
using common_flydelta_alpha_runner = std::function<bool(
        const common_flydelta_experiment_fixture & fixture,
        float alpha,
        bool apply_overlay,
        common_flydelta_counterfactual_trial & trial,
        std::string & error)>;

bool common_flydelta_run_alpha_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_alpha_search_config & config,
        const common_flydelta_alpha_runner & runner,
        std::vector<common_flydelta_alpha_trial> & trials,
        common_flydelta_alpha_selection & selection,
        std::string & error);

// Materializes a selected, host-verified HELPED alpha trial into the typed
// DeltaMemory input contract. The target coefficients are supplied by the
// host basis/evaluator and are validated before the example is returned.
bool common_flydelta_training_example_from_alpha_selection(
        const std::string & id,
        const std::string & behavior_key,
        const std::string & context_fingerprint,
        const std::string & basis_revision,
        const common_flydelta_sparse_code & context,
        const std::vector<float> & target_coefficients,
        const std::string & evidence_ref,
        common_flydelta_training_split split,
        float confidence,
        const common_flydelta_memory_config & memory,
        const std::vector<common_flydelta_alpha_trial> & trials,
        const common_flydelta_alpha_selection & selection,
        common_flydelta_training_example & example,
        std::string & error);

struct common_flydelta_training_example {
    int schema_version = 1;
    std::string id;
    std::string behavior_key;
    std::string context_fingerprint;
    std::string basis_revision;
    std::string evidence_ref;
    common_flydelta_training_split split = common_flydelta_training_split::train;
    common_flydelta_sparse_code context;
    std::vector<float> target_coefficients;
    common_flydelta_counterfactual_outcome outcome = common_flydelta_counterfactual_outcome::unknown;
    float confidence = 0.0f;
};

// A host-assigned corpus keeps train, validation and holdout examples
// together without allowing an example to silently cross a split boundary.
// The split assignment is part of the evidence contract; it is not inferred
// from model output and it is not changed by the trainer.
struct common_flydelta_training_corpus {
    int schema_version = 1;
    std::string id;
    std::string basis_revision;
    std::vector<common_flydelta_training_example> train;
    std::vector<common_flydelta_training_example> validation;
    std::vector<common_flydelta_training_example> holdout;
};

bool common_flydelta_training_corpus_validate(
        const common_flydelta_training_corpus & corpus,
        const common_flydelta_memory_config & memory,
        size_t max_examples,
        std::string & error);

bool common_flydelta_training_example_validate(
        const common_flydelta_training_example & example,
        const common_flydelta_memory_config & memory,
        std::string & error);
std::string common_flydelta_training_example_to_json(
        const common_flydelta_training_example & example);
bool common_flydelta_training_example_from_json(
        const std::string & text,
        common_flydelta_training_example & example,
        std::string & error);

// The batch is deliberately train-only. Validation/holdout examples must be
// passed to evaluation, never silently consumed as learning input.
bool common_flydelta_train_delta_memory(
        common_flydelta_delta_memory & memory,
        const std::vector<common_flydelta_training_example> & examples,
        float learning_rate,
        float decay,
        size_t & trained_examples,
        std::string & error);
