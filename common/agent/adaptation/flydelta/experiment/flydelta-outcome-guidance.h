#pragma once

#include "agent/adaptation/flydelta/flydelta-experiment.h"
#include "agent/adaptation/flydelta/flydelta-candidate-lifecycle.h"

#include <cstdint>
#include <string>
#include <vector>

enum class common_flydelta_outcome_guidance_mode { off, shadow, experimental };
// Offline evaluation can compare expected-utility ordering with Thompson
// sampling without adding another production runtime mode.
enum class common_flydelta_outcome_guidance_policy { posterior_sampling, posterior_mean };

const char * common_flydelta_outcome_guidance_mode_name(
        common_flydelta_outcome_guidance_mode mode);
bool common_flydelta_parse_outcome_guidance_mode(
        const std::string & value, common_flydelta_outcome_guidance_mode & mode,
        std::string & error);

struct common_flydelta_outcome_guidance_config {
    int schema_version = 1;
    common_flydelta_outcome_guidance_mode mode =
        common_flydelta_outcome_guidance_mode::shadow;
    uint64_t seed = 0x464c5944454c5441ULL;
    double harm_penalty = 5.0;
    size_t min_backoff_groups = 8;
    double max_prior_strength = 4.0;
};

bool common_flydelta_outcome_guidance_config_validate(
        const common_flydelta_outcome_guidance_config & config,
        std::string & error);

// Compatibility is intentionally an exact hard partition in V1. No history
// crosses scope, model, baseline deployment, verifier, RuleSet or action
// contract revisions.
struct common_flydelta_outcome_guidance_context {
    std::string scope_fingerprint;
    std::string model_profile_fingerprint;
    std::string tokenizer_fingerprint;
    std::string template_fingerprint;
    std::string baseline_deployment_fingerprint;
    std::string verifier_revision;
    std::string ruleset_revision;
    std::string action_contract_fingerprint;
    std::string semantic_domain;
    std::string strategy;
    std::string estimator;
};

struct common_flydelta_outcome_guidance_observation {
    std::string stable_observation_id;
    std::string fixture_group_id;
    std::string basis_identity;
    common_flydelta_outcome_guidance_context context;
    size_t rank = 0;
    double relative_layer = 0.0;
    double relative_dose = 0.0;
    bool mixed_coefficients = false;
    int diagnostic_margin_sign = 0;
    bool comparison_completed = false;
    bool overlay_applied = false;
    // Whether the comparison produced one of HELPED/NEUTRAL/HARMED. This is
    // independent of whether each individual verifier call had a predicate:
    // a completed comparison can be inconclusive even with known verifiers.
    bool outcome_known = false;
    bool verifier_known = false;
    common_flydelta_counterfactual_outcome outcome =
        common_flydelta_counterfactual_outcome::unknown;
};

struct common_flydelta_outcome_guidance_candidate {
    std::string arm_id;
    size_t rank = 0;
    double relative_layer = 0.0;
    double relative_dose = 0.0;
    bool mixed_coefficients = false;
    int diagnostic_margin_sign = 0;
    std::string basis_identity;
};

struct common_flydelta_outcome_guidance_prediction {
    std::string arm_id;
    std::string level;
    double p_known = 0.5;
    double p_helped_given_known = 1.0 / 3.0;
    double p_neutral_given_known = 1.0 / 3.0;
    double p_harmed_given_known = 1.0 / 3.0;
    double p_known_low = 0.0;
    double p_known_high = 1.0;
    double p_helped_low = 0.0;
    double p_helped_high = 1.0;
    double p_neutral_low = 0.0;
    double p_neutral_high = 1.0;
    double p_harmed_low = 0.0;
    double p_harmed_high = 1.0;
    double expected_utility = 0.0;
    double known_weight = 0.0;
    double helped_weight = 0.0;
    double neutral_weight = 0.0;
    double harmed_weight = 0.0;
    double effective_known_observations = 0.0;
    double effective_unknown_observations = 0.0;
    size_t backoff_fixture_groups = 0;
    double backoff_prior_strength = 0.0;
    double backoff_categorical_prior_strength = 0.0;
    size_t raw_observations = 0;
    size_t fixture_groups = 0;
};

struct common_flydelta_outcome_guidance_replay_metrics {
    size_t eligible_predictions = 0;
    size_t known_outcomes = 0;
    size_t unknown_outcomes = 0;
    double model_verifiability_brier = 0.0;
    double fixed_prior_verifiability_brier = 0.0;
    double historical_mean_verifiability_brier = 0.0;
    double model_verifiability_ece = 0.0;
    double model_outcome_brier = 0.0;
    double fixed_prior_outcome_brier = 0.0;
    double historical_mean_outcome_brier = 0.0;
};

struct common_flydelta_outcome_guidance_decision {
    std::vector<size_t> original_order;
    // Posterior-only rank, hypothetical mixed-policy order, and actual order
    // remain distinct in both shadow and experimental modes.
    std::vector<size_t> shadow_order;
    std::vector<size_t> mixed_policy_order;
    std::vector<size_t> selected_order;
    std::vector<std::string> selection_reasons;
    std::vector<std::string> mixed_policy_reasons;
    std::vector<common_flydelta_outcome_guidance_prediction> predictions;
    std::string history_fingerprint;
    size_t read_history_rows = 0;
    size_t compatible_history_rows = 0;
    size_t deduplicated_history_rows = 0;
    size_t rejected_history_rows = 0;
    std::string fallback_reason;
};

bool common_flydelta_outcome_guidance_build_decision(
        const common_flydelta_outcome_guidance_config & config,
        const common_flydelta_outcome_guidance_context & context,
        const std::vector<common_flydelta_outcome_guidance_observation> & history,
        const std::vector<common_flydelta_outcome_guidance_candidate> & candidates,
        const size_t selection_limit,
        const uint64_t job_seed,
        common_flydelta_outcome_guidance_decision & decision,
        std::string & error,
        common_flydelta_outcome_guidance_policy policy =
            common_flydelta_outcome_guidance_policy::posterior_sampling,
        size_t policy_sequence_start = 0);

// Replays observations in their registered order. Holdout fixture groups are
// never added to the training projection; non-holdout rows become available
// only after their position in the log. Scores are mean Brier losses.
bool common_flydelta_outcome_guidance_replay(
        const common_flydelta_outcome_guidance_config & config,
        const std::vector<common_flydelta_outcome_guidance_observation> & chronological,
        const std::vector<std::string> & holdout_fixture_groups,
        common_flydelta_outcome_guidance_replay_metrics & metrics,
        std::string & error);

bool common_flydelta_append_outcome_guidance_decision_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & event_context,
        const std::string & job_id,
        common_flydelta_outcome_guidance_mode mode,
        uint64_t seed,
        int phase_id,
        size_t decision_sequence,
        size_t selection_limit,
        size_t policy_sequence_start,
        size_t policy_cursor_before,
        size_t policy_slots_consumed,
        const common_flydelta_outcome_guidance_context & context,
        const std::vector<common_flydelta_outcome_guidance_candidate> & candidates,
        const common_flydelta_outcome_guidance_decision & decision,
        std::string & error);
