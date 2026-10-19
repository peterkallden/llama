#include "agent/adaptation/flydelta/experiment/flydelta-outcome-guidance.h"
#include "agent/prediction/outcome-guidance.h"
#include "agent/tooling/adapters/tool-adapters.h"
#include "agent/tooling/catalog/tool-catalog.h"
#include "agent/tooling/contracts/schema-contract.h"
#include "agent/tooling/registry/tool-registry.h"

#include <algorithm>
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
    o.outcome_known = outcome != common_flydelta_counterfactual_outcome::unknown;
    return o;
}

int main() {
    std::string error;
    common_outcome_guidance_posterior shared_posterior;
    CHECK(common_outcome_guidance_make_posterior(
        3.0, 1.0, .75, {2.0, 1.0, 0.0}, {2.0/3.0, 1.0/3.0, 0.0},
        2.0, 1.0, shared_posterior));
    CHECK(shared_posterior.p_known > .5 && shared_posterior.p_known < 1.0);
    CHECK(shared_posterior.category_probabilities.size() == 3);
    CHECK(std::abs(shared_posterior.category_probabilities[0] +
        shared_posterior.category_probabilities[1] +
        shared_posterior.category_probabilities[2] - 1.0) < 1e-12);
    CHECK(!common_outcome_guidance_make_posterior(
        -1.0, 0.0, .5, {1.0}, {1.0}, 0.0, 0.0, shared_posterior));
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    CHECK(catalog.bootstrap("analysis", bootstrap, error));
    const auto * predict_definition = catalog.find_definition("outcome.predict");
    CHECK(predict_definition != nullptr && predict_definition->risk_class == common_tool_risk_class::local_read);
    const auto all_configured = catalog.load_profile("all-configured", error);
    CHECK(error.empty() && std::any_of(all_configured.begin(), all_configured.end(),
        [](const auto & definition) { return definition.name == "outcome.predict"; }));
    const auto result_schema = nlohmann::json::parse(predict_definition->result_schema_json, nullptr, false);
    CHECK(result_schema.is_object() && result_schema.contains("properties") &&
        result_schema["properties"].contains("predictions") &&
        result_schema["properties"].contains("unresolved_refs") &&
        result_schema["properties"].contains("scope_fingerprint"));
    const std::string sample_result = nlohmann::ordered_json{
        {"status", "available"}, {"scope_fingerprint", "sha256:scope"},
        {"history_watermark", 12}, {"history_fingerprint", "sha256:history"},
        {"unresolved_refs", nlohmann::ordered_json::array()},
        {"predictions", nlohmann::ordered_json::array({nlohmann::ordered_json{
            {"observation_ref", "obs-1"},
            {"prediction_kind", "chronological_leave_fixture_group_out"},
            {"p_known", 0.8}, {"p_known_interval", {0.5, 1.0}},
            {"p_helped_given_known", 0.7}, {"p_neutral_given_known", 0.2},
            {"p_harmed_given_known", 0.1}, {"expected_utility", 0.2},
            {"evidence_groups", 8}, {"backoff_level", "backoff"},
            {"history_fingerprint", "sha256:prior"}, {"domain", "dataset"},
            {"strategy", "deep"}, {"estimator", "coefficient-search"},
            {"observed_outcome", "helped"},
        }})},
    }.dump();
    std::string normalized_result;
    CHECK(common_schema_normalize_and_validate_object(
        sample_result, predict_definition->result_schema_json, normalized_result, error));
    common_native_tool_bindings bindings;
    bool prediction_called = false;
    bindings.outcome_predict = [&prediction_called](const std::string & input) {
        prediction_called = input == R"({"candidate_refs":["obs-1"]})";
        return common_tool_execution_result::success(
            R"({"status":"insufficient_history","scope":"local/p/s","history_watermark":1,"history_fingerprint":"sha256:test","predictions":[]})");
    };
    common_tool_registry registry;
    common_tool_adapter_result adapter_result;
    CHECK(common_register_native_tool_adapters(catalog, "analysis", bindings,
        registry, adapter_result, error));
    CHECK(registry.contains("outcome.predict") && registry.is_read_only("outcome.predict"));
    CHECK(registry.execute({"outcome.predict", R"({"candidate_refs":["obs-1"]})"}).ok);
    CHECK(prediction_called);
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

    // Layer search can borrow only the deliberately weak same-domain prior
    // from Deep/TFO strategies; the exact deployment/verifier partition stays.
    auto whirlpool_context = c;
    whirlpool_context.strategy = "whirlpool";
    whirlpool_context.estimator = "layer-region-prior-v2";
    common_flydelta_outcome_guidance_candidate layer_candidate;
    layer_candidate.arm_id = "layer:12";
    layer_candidate.rank = 1;
    layer_candidate.relative_layer = .5;
    layer_candidate.relative_dose = .2;
    common_flydelta_outcome_guidance_decision cross_stage;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, whirlpool_context, history, {layer_candidate}, 1, 56, cross_stage, error));
    CHECK(cross_stage.predictions.size() == 1);
    CHECK(cross_stage.predictions.front().level == "backoff");
    CHECK(cross_stage.predictions.front().backoff_fixture_groups >= 8);
    CHECK(cross_stage.predictions.front().p_helped_given_known > .60);

    // Transfer keeps WHERE: equal-sized early/harmed and late/helped histories
    // must produce different priors for the two proposed layer regions.
    std::vector<common_flydelta_outcome_guidance_observation> regional_history;
    for (int i = 0; i < 8; ++i) {
        auto early = observation("early-" + std::to_string(i), "early-group-" + std::to_string(i),
            common_flydelta_counterfactual_outcome::harmed);
        early.relative_layer = .125;
        auto late = observation("late-" + std::to_string(i), "late-group-" + std::to_string(i),
            common_flydelta_counterfactual_outcome::helped);
        late.relative_layer = .875;
        regional_history.push_back(early);
        regional_history.push_back(late);
    }
    const std::vector<common_flydelta_outcome_guidance_candidate> regional_candidates = {
        common_flydelta_outcome_guidance_layer_candidate(2, 17, .2),
        common_flydelta_outcome_guidance_layer_candidate(14, 17, .2),
    };
    common_flydelta_outcome_guidance_decision regional;
    CHECK(common_flydelta_outcome_guidance_build_decision(config, whirlpool_context,
        regional_history, regional_candidates, 2, 56, regional, error));
    CHECK(regional.predictions[1].p_helped_given_known > regional.predictions[0].p_helped_given_known);
    auto direct_layer = regional_history.back();
    direct_layer.context = whirlpool_context;
    direct_layer.rank = regional_candidates.back().rank;
    direct_layer.basis_identity = regional_candidates.back().basis_identity;
    direct_layer.relative_layer = regional_candidates.back().relative_layer;
    direct_layer.relative_dose = regional_candidates.back().relative_dose;
    direct_layer.diagnostic_margin_sign = regional_candidates.back().diagnostic_margin_sign;
    CHECK(common_flydelta_outcome_guidance_build_decision(config, whirlpool_context,
        {direct_layer}, {regional_candidates.back()}, 1, 56, regional, error));
    CHECK(regional.predictions.front().level == "arm_category");

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

    auto mixed_group_history = std::vector{observation("mixed-help", "mixed-group",
        common_flydelta_counterfactual_outcome::helped)};
    common_flydelta_outcome_guidance_decision known_only_group;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, mixed_group_history, {direct_candidate}, 1, 9, known_only_group, error));
    mixed_group_history.push_back(observation("mixed-unknown", "mixed-group",
        common_flydelta_counterfactual_outcome::unknown));
    common_flydelta_outcome_guidance_decision mixed_known_unknown_group;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, mixed_group_history, {direct_candidate}, 1, 9,
        mixed_known_unknown_group, error));
    CHECK(mixed_known_unknown_group.predictions[0].p_known <
        known_only_group.predictions[0].p_known);
    CHECK(std::fabs(mixed_known_unknown_group.predictions[0].p_helped_given_known -
        known_only_group.predictions[0].p_helped_given_known) < 1e-12);

    config.mode = common_flydelta_outcome_guidance_mode::experimental;
    common_flydelta_outcome_guidance_decision experimental_a, experimental_b;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 0, experimental_a, error));
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 0, experimental_b, error));
    CHECK(experimental_a.selected_order == experimental_b.selected_order);
    CHECK(experimental_a.selected_order.size() == candidates.size());
    CHECK(experimental_a.history_fingerprint == experimental_b.history_fingerprint);
    CHECK(experimental_a.selection_reasons.size() == candidates.size());
    CHECK(experimental_a.selection_reasons[0] == "existing_order");
    CHECK(experimental_a.selection_reasons[1] == "existing_order");
    CHECK(experimental_a.selection_reasons[2] == "posterior_sample");
    CHECK(experimental_a.selection_reasons[3] == "uniform_exploration");
    CHECK(experimental_a.mixed_policy_reasons == experimental_a.selection_reasons);
    CHECK(experimental_a.mixed_policy_order.size() == candidates.size());
    {
        auto sorted_order = experimental_a.mixed_policy_order;
        std::sort(sorted_order.begin(), sorted_order.end());
        CHECK(sorted_order == experimental_a.original_order);
    }
    common_flydelta_outcome_guidance_decision continued_policy;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 0, continued_policy, error,
        common_flydelta_outcome_guidance_policy::posterior_sampling, 1));
    CHECK(continued_policy.selection_reasons[0] == "existing_order");
    CHECK(continued_policy.selection_reasons[1] == "posterior_sample");
    CHECK(continued_policy.selection_reasons[2] == "uniform_exploration");
    CHECK(continued_policy.selection_reasons[3] == "existing_order");
    common_flydelta_outcome_guidance_decision posterior_mean;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, history, candidates, candidates.size(), 0, posterior_mean, error,
        common_flydelta_outcome_guidance_policy::posterior_mean));
    CHECK(posterior_mean.selection_reasons[2] == "posterior_mean");
    CHECK(posterior_mean.selection_reasons[3] == "uniform_exploration");
    CHECK(posterior_mean.selected_order == posterior_mean.mixed_policy_order);
    {
        auto sorted_order = posterior_mean.selected_order;
        std::sort(sorted_order.begin(), sorted_order.end());
        CHECK(sorted_order == posterior_mean.original_order);
    }

    // Exact categories are basis-local; a same-rank arm on a different basis
    // must not inherit leaf evidence (it may still receive coarse backoff).
    auto other_basis = candidates[0];
    other_basis.arm_id = "other-basis";
    other_basis.rank = 2;
    other_basis.basis_identity = "basis-C";
    auto basis_a = observation("basis-a", "basis-fixture-a",
        common_flydelta_counterfactual_outcome::helped);
    basis_a.basis_identity = "basis-A";
    auto basis_b = observation("basis-b", "basis-fixture-b",
        common_flydelta_counterfactual_outcome::harmed);
    basis_b.basis_identity = "basis-B";
    common_flydelta_outcome_guidance_decision basis_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, {basis_a, basis_b}, {other_basis}, 1, 0, basis_decision, error));
    CHECK(basis_decision.predictions[0].fixture_groups == 0);
    CHECK(std::abs(basis_decision.predictions[0].p_helped_given_known - 1.0/3.0) < 1e-9);

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
    experimental_a.stage = "tfo_population";
    experimental_a.phase_id = 2;
    experimental_a.round = 1;
    experimental_a.decision_id = "tfo-population-1";
    CHECK(common_flydelta_append_outcome_guidance_decision_lifecycle(
        guidance_store, guidance_event, "job-guidance", config.mode, config.seed, 0, 0,
        candidates.size(), 0, 0, candidates.size(), c, candidates, experimental_a, error));
    const auto guidance_records = guidance_store.list(error);
    CHECK(error.empty() && guidance_records.size() == 1);
    const auto guidance_payload = nlohmann::ordered_json::parse(
        guidance_records.front().payload_json);
    CHECK(guidance_payload["record_type"] == "outcome_guidance_decision");
    CHECK(guidance_payload["schema_version"] == 4);
    CHECK(guidance_payload["stage"] == "tfo_population");
    CHECK(guidance_payload["round"] == 1);
    CHECK(guidance_payload["decision_id"] == "tfo-population-1");
    CHECK(guidance_payload["decision_sequence"] == 0);
    CHECK(guidance_payload["policy_sequence_start"] == 0);
    CHECK(guidance_payload["policy_slots_consumed"] == candidates.size());
    CHECK(guidance_payload["candidate_pool_fingerprint"].get<std::string>().rfind(
        "sha256:", 0) == 0);
    CHECK(guidance_payload["candidate_pool_fingerprint"] ==
        common_flydelta_outcome_guidance_pool_fingerprint(candidates));
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

    auto conflict = original_history.front();
    conflict.outcome = common_flydelta_counterfactual_outcome::harmed;
    common_flydelta_outcome_guidance_decision conflict_decision;
    CHECK(common_flydelta_outcome_guidance_build_decision(
        config, c, {original_history.front(), conflict}, {candidates[0]}, 1, 2,
        conflict_decision, error));
    CHECK(conflict_decision.rejected_history_rows == 2);
    CHECK(conflict_decision.predictions[0].fixture_groups == 0);

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
    auto malformed_unknown = observation("holdout-malformed", "holdout-fixture-malformed",
        common_flydelta_counterfactual_outcome::unknown);
    malformed_unknown.outcome_known = true;
    chronological.push_back(malformed_unknown);
    common_flydelta_outcome_guidance_replay_metrics replay;
    CHECK(common_flydelta_outcome_guidance_replay(config, chronological,
        {"holdout-fixture-a", "holdout-fixture-b", "holdout-fixture-malformed"}, replay, error));
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
