#include "agent/adaptation/flydelta/experiment/flydelta-outcome-guidance.h"

#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_outcome_guidance_context context() {
    common_flydelta_outcome_guidance_context c;
    c.scope_fingerprint = "scope-A";
    c.model_profile_fingerprint = "model-A";
    c.tokenizer_fingerprint = "tok-A";
    c.template_fingerprint = "template-A";
    c.baseline_deployment_fingerprint = "base-A";
    c.verifier_revision = "oracle-A";
    c.semantic_domain = "dataset.aggregate";
    c.strategy = "coordinate";
    c.estimator = "margin-geometry-v1";
    return c;
}

static common_flydelta_outcome_guidance_observation observation(
        const std::string & id, const std::string & group,
        common_flydelta_counterfactual_outcome outcome) {
    common_flydelta_outcome_guidance_observation o;
    o.stable_observation_id = id;
    o.fixture_group_id = group;
    o.context = context();
    o.rank = 2;
    o.relative_layer = .5;
    o.relative_dose = .5;
    o.comparison_completed = true;
    o.overlay_applied = true;
    o.verifier_known = outcome != common_flydelta_counterfactual_outcome::unknown;
    o.outcome = outcome;
    return o;
}

int main() {
    std::string error;
    common_flydelta_outcome_guidance_config config;
    CHECK(config.mode == common_flydelta_outcome_guidance_mode::shadow);
    CHECK(common_flydelta_outcome_guidance_config_validate(config, error));

    const auto c = context();
    std::vector<common_flydelta_outcome_guidance_candidate> candidates(4);
    candidates[0].arm_id = "arm-A"; candidates[0].rank = 1;
    candidates[0].relative_layer = .5; candidates[0].relative_dose = .5;
    candidates[1].arm_id = "arm-B"; candidates[1].rank = 2;
    candidates[1].relative_layer = .5; candidates[1].relative_dose = .5;
    candidates[1].mixed_coefficients = true; candidates[1].diagnostic_margin_sign = 1;
    candidates[2].arm_id = "arm-C"; candidates[2].rank = 1;
    candidates[2].relative_layer = .5; candidates[2].relative_dose = .9;
    candidates[2].mixed_coefficients = true; candidates[2].diagnostic_margin_sign = -1;
    candidates[3].arm_id = "arm-D"; candidates[3].rank = 1;
    candidates[3].relative_layer = .5; candidates[3].relative_dose = .2;
    std::vector<common_flydelta_outcome_guidance_observation> history;
    for (int i=0;i<8;++i) history.push_back(observation(
        "obs-"+std::to_string(i), "fixture-"+std::to_string(i),
        common_flydelta_counterfactual_outcome::helped));
    // Repeated measurements from one fixture share total weight one.
    history.push_back(observation("same-fixture-help", "fixture-shared",
        common_flydelta_counterfactual_outcome::helped));
    history.push_back(observation("same-fixture-harm", "fixture-shared",
        common_flydelta_counterfactual_outcome::harmed));
    // Duplicate retries, incomplete comparisons, and unapplied overlays are
    // rejected from the projection.
    history.push_back(history.front());
    auto diagnostic = observation("diagnostic", "fixture-diagnostic",
        common_flydelta_counterfactual_outcome::helped);
    diagnostic.comparison_completed = false;
    history.push_back(diagnostic);
    auto no_overlay = observation("no-overlay", "fixture-no-overlay",
        common_flydelta_counterfactual_outcome::helped);
    no_overlay.overlay_applied = false;
    history.push_back(no_overlay);
    auto unknown = observation("unknown", "fixture-unknown",
        common_flydelta_counterfactual_outcome::unknown);
    history.push_back(unknown);
    auto wrong_model = observation("wrong-model", "fixture-wrong-model",
        common_flydelta_counterfactual_outcome::harmed);
    wrong_model.context.model_profile_fingerprint = "model-B";
    history.push_back(wrong_model);

    common_flydelta_outcome_guidance_decision shadow;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 55, shadow, error));
    CHECK(shadow.original_order == std::vector<size_t>({0,1,2,3}));
    CHECK(shadow.selected_order == shadow.original_order);
    CHECK(shadow.selection_reasons.size() == candidates.size());
    CHECK(shadow.selection_reasons[0] == "shadow_preserved_existing_order");
    CHECK(shadow.predictions.size() == candidates.size());
    CHECK(shadow.predictions[0].level == "backoff");
    CHECK(shadow.predictions[0].fixture_groups == 0);
    CHECK(shadow.predictions[0].p_helped_given_known > .60);
    CHECK(shadow.predictions[0].p_harmed_high > shadow.predictions[0].p_harmed_low);
    CHECK(shadow.predictions[0].backoff_fixture_groups >= 8);
    CHECK(shadow.predictions[0].backoff_prior_strength == 4.0);

    common_flydelta_outcome_guidance_decision direct;
    common_flydelta_outcome_guidance_candidate direct_candidate;
    direct_candidate.arm_id = "direct";
    direct_candidate.rank = 2;
    direct_candidate.relative_layer = .5;
    direct_candidate.relative_dose = .5;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, {direct_candidate}, 1, 55, direct, error));
    CHECK(direct.predictions[0].fixture_groups == 10);
    CHECK(direct.predictions[0].raw_observations == 11);
    CHECK(direct.predictions[0].effective_known_observations == 9.0);
    CHECK(direct.predictions[0].effective_unknown_observations == 1.0);

    config.mode = common_flydelta_outcome_guidance_mode::experimental;
    common_flydelta_outcome_guidance_decision experimental_a, experimental_b;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 55, experimental_a, error));
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 55, experimental_b, error));
    CHECK(experimental_a.selected_order == experimental_b.selected_order);
    CHECK(experimental_a.selected_order.size() == candidates.size());
    CHECK(experimental_a.history_fingerprint == experimental_b.history_fingerprint);
    CHECK(experimental_a.selection_reasons.size() == candidates.size());
    CHECK(experimental_a.selection_reasons[0] == "existing_order");
    CHECK(experimental_a.selection_reasons[1] == "existing_order");
    CHECK(experimental_a.selection_reasons[2] == "posterior_sample");
    CHECK(experimental_a.selection_reasons[3] == "uniform_exploration");

    common_learning_in_memory_lifecycle_store guidance_store;
    common_flydelta_lifecycle_event_context guidance_event;
    guidance_event.event_id = "guidance:test:1";
    guidance_event.idempotency_key = guidance_event.event_id;
    guidance_event.source_id = "test-outcome-guidance";
    guidance_event.scope.namespace_id = "local";
    guidance_event.scope.project_id = "agent-tests";
    guidance_event.scope.session_id = "guidance-session";
    guidance_event.content_hash = "sha256:guidance-test";
    guidance_event.created_at = "2026-10-18T00:00:00Z";
    CHECK(common_flydelta_append_outcome_guidance_decision_lifecycle(
        guidance_store, guidance_event, "job-guidance", config.mode, config.seed,
        c, candidates, experimental_a, error));
    const auto guidance_records = guidance_store.list(error);
    CHECK(error.empty() && guidance_records.size() == 1);
    const auto guidance_payload = nlohmann::ordered_json::parse(
        guidance_records.front().payload_json);
    CHECK(guidance_payload["record_type"] == "outcome_guidance_decision");
    CHECK(guidance_payload["selection_reasons"].size() == candidates.size());
    CHECK(guidance_payload["predictions"][0].contains("p_harmed_interval_95_approx"));

    // The OFF contract is identity and does not return an empty permutation.
    config.mode = common_flydelta_outcome_guidance_mode::off;
    common_flydelta_outcome_guidance_decision disabled;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 55, disabled, error));
    CHECK(disabled.selected_order == disabled.original_order);
    CHECK(disabled.selection_reasons.size() == candidates.size());
    CHECK(disabled.predictions.empty());

    // An UNKNOWN host comparison only lowers P(known); it adds no outcome
    // mass to the helped/neutral/harmed Dirichlet.
    const auto before = shadow.predictions[0];
    std::vector<common_flydelta_outcome_guidance_observation> unknown_only = {unknown};
    unknown_only[0].rank = candidates[0].rank;
    config.mode = common_flydelta_outcome_guidance_mode::shadow;
    common_flydelta_outcome_guidance_decision unknown_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, unknown_only, {candidates[0]}, 1, 2, unknown_decision, error));
    CHECK(unknown_decision.predictions[0].p_known < .5);
    CHECK(std::abs(unknown_decision.predictions[0].p_helped_given_known - 1.0/3.0) < 1e-9);

    // A changed observation under the same stable id changes the projection
    // fingerprint; a duplicate retry contributes only once.
    auto changed = history.front();
    changed.outcome = common_flydelta_counterfactual_outcome::harmed;
    std::vector<common_flydelta_outcome_guidance_observation> changed_history = {changed};
    auto original_history = history;
    original_history.resize(1);
    common_flydelta_outcome_guidance_decision changed_decision, original_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, changed_history, {candidates[0]}, 1, 2, changed_decision, error));
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, original_history, {candidates[0]}, 1, 2, original_decision, error));
    CHECK(changed_decision.history_fingerprint != original_decision.history_fingerprint);

    // Chronological replay holds out whole fixture groups and compares the
    // posterior with fixed and simple historical-share baselines.
    config.mode = common_flydelta_outcome_guidance_mode::shadow;
    std::vector<common_flydelta_outcome_guidance_observation> chronological;
    for (int i = 0; i < 8; ++i) chronological.push_back(observation(
        "train-" + std::to_string(i), "train-fixture-" + std::to_string(i),
        i < 6 ? common_flydelta_counterfactual_outcome::helped :
            common_flydelta_counterfactual_outcome::neutral));
    chronological.push_back(observation("holdout-a", "holdout-fixture-a",
        common_flydelta_counterfactual_outcome::helped));
    chronological.push_back(observation("holdout-b", "holdout-fixture-b",
        common_flydelta_counterfactual_outcome::unknown));
    common_flydelta_outcome_guidance_replay_metrics replay;
    CHECK(common_flydelta_outcome_guidance_replay(config, chronological,
        {"holdout-fixture-a", "holdout-fixture-b"}, replay, error));
    CHECK(replay.eligible_predictions == 2 && replay.known_outcomes == 1 &&
        replay.unknown_outcomes == 1);
    CHECK(replay.model_verifiability_brier >= 0 && replay.model_verifiability_brier <= 1);
    CHECK(replay.model_verifiability_ece >= 0 && replay.model_verifiability_ece <= 1);
    CHECK(replay.model_outcome_brier >= 0 && replay.model_outcome_brier <= 2);
    CHECK(replay.fixed_prior_verifiability_brier >= 0 &&
        replay.historical_mean_verifiability_brier >= 0);

    common_flydelta_outcome_guidance_context incompatible = c;
    incompatible.baseline_deployment_fingerprint = "base-B";
    common_flydelta_outcome_guidance_decision incompatible_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, incompatible, history, {candidates[0]}, 1, 2, incompatible_decision, error));
    CHECK(incompatible_decision.predictions[0].p_helped_given_known < before.p_helped_given_known);
    common_flydelta_outcome_guidance_context wrong_ruleset = c;
    wrong_ruleset.ruleset_revision = "rules-v2";
    common_flydelta_outcome_guidance_decision wrong_ruleset_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, wrong_ruleset, history, {candidates[0]}, 1, 2, wrong_ruleset_decision, error));
    CHECK(wrong_ruleset_decision.predictions[0].fixture_groups == 0);

    config.harm_penalty = -1;
    CHECK(!common_flydelta_outcome_guidance_config_validate(config, error));
    std::cout << "FlyDelta outcome guidance checks passed\n";
    return 0;
}
