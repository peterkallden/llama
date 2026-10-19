#include "agent/adaptation/flydelta/flydelta-whirlpool-search.h"

#include <algorithm>
#include <cmath>
#include <vector>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_fixture fixture() {
    common_flydelta_experiment_fixture value;
    value.id = "fixture:whirlpool";
    value.task_fingerprint = "task";
    value.model_profile_fingerprint = "model";
    value.tokenizer_fingerprint = "tokenizer";
    value.template_fingerprint = "template";
    value.execution_context_fingerprint = "execution-context";
    value.verifier_revision = "verifier";
    return value;
}

int main() {
    common_flydelta_whirlpool_search_config config;
    config.available_layers = {1, 2, 3, 4, 5, 6, 7};
    config.seed_layers = {2, 5};
    config.layer_diagnostics = {
        {1, 0.2f, 0.01f, 0.1f, 0.0f},
        {2, 0.4f, 0.05f, 0.1f, 0.0f},
        {5, 0.7f, 0.20f, 0.1f, 0.0f},
    };
    config.total_scale = 0.02f;
    config.max_rounds = 2;
    config.probes_per_round = 4;
    config.max_trials = 8;
    config.initial_radius = 2;
    std::string error;
    CHECK(common_flydelta_whirlpool_search_config_validate(config, error));

    std::vector<common_flydelta_intervention_region_trial> trials;
    common_flydelta_intervention_region_selection selection;
    common_flydelta_whirlpool_trace trace;
    CHECK(common_flydelta_run_whirlpool_search(
        fixture(), config,
        [](const auto &, const auto * candidate, auto & trial, auto & margin,
                auto & geometry, auto & geometry_available, std::string &) {
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.evidence_ref = candidate == nullptr ? "evidence:baseline" : "evidence:arm";
            if (candidate == nullptr) {
                trial.passed = false;
                trial.quality = 0.0f;
                return true;
            }
            const uint32_t layer = candidate->anchor_layer_index;
            trial.passed = layer == 5;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            geometry_available = true;
            geometry.layer_index = layer;
            geometry.cosine = layer == 5 ? 0.9f : 0.4f;
            geometry.progress = layer == 5 ? 0.8f : 0.1f;
            geometry.leakage = 0.05f;
            geometry.shift_norm = candidate->total_scale;
            margin = {};
            return true;
        }, trials, selection, trace, error));
    CHECK(error.empty());
    CHECK(!trials.empty());
    CHECK(selection.selected);
    CHECK(trials[selection.trial_index].candidate.anchor_layer_index == 5);
    CHECK(trials[selection.trial_index].outcome ==
        common_flydelta_counterfactual_outcome::helped);
    CHECK(std::any_of(trials.begin(), trials.end(), [](const auto & trial) {
        return trial.candidate.anchor_layer_index == 5;
    }));
    CHECK(trace.model_evaluations == trials.size() + 1);
    CHECK(trace.best_trial_index < trials.size());
    CHECK(trace.rounds.size() == 2);
    CHECK(trace.rounds.front().centre_before == 5);
    CHECK(trace.rounds.front().centre_after == 5);
    CHECK(trace.final_centre == 5);
    CHECK(common_flydelta_whirlpool_trace_validate(trace, config, trials.size(), error));

    common_flydelta_whirlpool_search_config batch_config = config;
    batch_config.max_rounds = 1;
    batch_config.max_trials = 4;
    batch_config.use_dose_controller = false;
    size_t batch_calls = 0;
    std::vector<size_t> batch_sizes;
    trials.clear();
    selection = {};
    trace = {};
    const common_flydelta_whirlpool_search_runner batch_baseline_runner =
        [](const auto &, const auto * candidate, auto & trial, auto & margin,
                auto &, auto & geometry_available, std::string &) {
            if (candidate != nullptr) return false;
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.evidence_ref = "evidence:batch-baseline";
            trial.passed = false;
            trial.quality = 0.0f;
            margin = {};
            geometry_available = false;
            return true;
        };
    const common_flydelta_whirlpool_search_batch_runner batch_runner =
        [&](const auto &, const auto & candidates, auto & batch_trials, auto & batch_margins,
                auto & batch_geometries, auto & batch_geometry_available, std::string &) {
            ++batch_calls;
            batch_sizes.push_back(candidates.size());
            batch_trials.clear();
            batch_margins.clear();
            batch_geometries.clear();
            batch_geometry_available.clear();
            for (const auto & candidate : candidates) {
                common_flydelta_counterfactual_trial trial;
                trial.executed = true;
                trial.verifier_known = true;
                trial.evidence_ref = "evidence:batch-arm";
                trial.quality = candidate.anchor_layer_index == 5 ? 1.0f : 0.0f;
                common_flydelta_decision_margin margin;
                common_flydelta_representation_diagnostics geometry;
                geometry.layer_index = candidate.anchor_layer_index;
                geometry.cosine = candidate.anchor_layer_index == 5 ? 0.9f : 0.4f;
                geometry.progress = candidate.anchor_layer_index == 5 ? 0.8f : 0.1f;
                geometry.leakage = 0.05f;
                geometry.shift_norm = candidate.total_scale;
                batch_trials.push_back(std::move(trial));
                batch_margins.push_back(std::move(margin));
                batch_geometries.push_back(std::move(geometry));
                batch_geometry_available.push_back(true);
            }
            return true;
        };
    CHECK(common_flydelta_run_whirlpool_search_batched(
        fixture(), batch_config, batch_baseline_runner, batch_runner,
        trials, selection, trace, error));
    CHECK(error.empty());
    CHECK(batch_calls == 1);
    CHECK(batch_sizes.size() == 1 && batch_sizes.front() >= 2);
    CHECK(trace.model_evaluations == trials.size() + 1);
    CHECK(common_flydelta_whirlpool_trace_validate(
        trace, batch_config, trials.size(), error));

    common_flydelta_whirlpool_search_config guided_config = batch_config;
    guided_config.layer_diagnostics.clear();
    size_t guidance_requests = 0;
    uint32_t first_guided_layer = 0;
    const common_flydelta_outcome_guidance_adviser guided_adviser =
        [&guidance_requests, &first_guided_layer](const auto & request,
                auto & decision, std::string &) {
            ++guidance_requests;
            decision.shadow_mode = false;
            decision.original_order.resize(request.candidates.size());
            decision.shadow_order.resize(request.candidates.size());
            decision.mixed_policy_order.resize(request.candidates.size());
            decision.selected_order.resize(request.candidates.size());
            decision.selection_reasons.assign(request.candidates.size(), "guided");
            decision.mixed_policy_reasons.assign(request.candidates.size(), "guided");
            for (size_t i = 0; i < request.candidates.size(); ++i) {
                decision.original_order[i] = i;
                decision.shadow_order[i] = i;
                decision.mixed_policy_order[i] = i;
                decision.selected_order[i] = i;
            }
            if (request.stage == "whirlpool_initial_centre") {
                decision.shadow_mode = false;
                decision.predictions.push_back({});
                decision.predictions.back().level = "backoff";
                decision.predictions.back().arm_id = request.candidates[5].arm_id;
                std::rotate(decision.selected_order.begin(),
                    decision.selected_order.begin() + 5, decision.selected_order.end());
                decision.mixed_policy_order = decision.selected_order;
            } else if (request.stage == "whirlpool_probe") {
                std::rotate(decision.selected_order.begin(),
                    decision.selected_order.end() - 1, decision.selected_order.end());
                decision.mixed_policy_order = decision.selected_order;
                const std::string & id = request.candidates[decision.selected_order[0]].arm_id;
                first_guided_layer = static_cast<uint32_t>(std::stoul(id.substr(6)));
            }
            decision.history_fingerprint = "sha256:test-history";
            return request.selection_limit <= request.candidates.size();
        };
    trials.clear();
    selection = {};
    trace = {};
    CHECK(common_flydelta_run_whirlpool_search_batched(
        fixture(), guided_config, batch_baseline_runner, batch_runner,
        trials, selection, trace, error, guided_adviser));
    CHECK(guidance_requests == 2);
    CHECK(trace.initial_centre_source == "bayesian_posterior");
    CHECK(trace.rounds.front().centre_before == 6);
    CHECK(trace.rounds.front().probed_layers.front() == first_guided_layer);
    CHECK(trace.initial_centre_guidance_decision_id == "whirlpool:initial-centre");
    CHECK(trace.rounds.front().guidance_decision_id == "whirlpool:probe:0");
    CHECK(trace.rounds.front().guidance_history_fingerprint == "sha256:test-history");
    CHECK(common_flydelta_whirlpool_trace_validate(trace, guided_config, trials.size(), error));

    auto diagnostic_wins_config = guided_config;
    diagnostic_wins_config.layer_diagnostics = config.layer_diagnostics;
    trials.clear();
    selection = {};
    trace = {};
    guidance_requests = 0;
    CHECK(common_flydelta_run_whirlpool_search_batched(
        fixture(), diagnostic_wins_config, batch_baseline_runner, batch_runner,
        trials, selection, trace, error, guided_adviser));
    CHECK(trace.initial_centre_source == "current_diagnostic");
    CHECK(trace.rounds.front().centre_before == 5);

    // Truncation precedes visited-layer filtering on the legacy path. A
    // shadow/fallback adviser must not refill the resulting shorter batch.
    auto legacy_config = batch_config;
    legacy_config.probes_per_round = 2;
    legacy_config.max_rounds = 2;
    legacy_config.max_trials = 8;
    CHECK(common_flydelta_run_whirlpool_search_batched(
        fixture(), legacy_config, batch_baseline_runner, batch_runner,
        trials, selection, trace, error));
    std::vector<uint32_t> legacy_layers;
    for (const auto & trial : trials) legacy_layers.push_back(trial.candidate.anchor_layer_index);
    CHECK(legacy_layers == std::vector<uint32_t>({5, 3, 4}));
    for (const bool fallback : {false, true}) {
        CHECK(common_flydelta_run_whirlpool_search_batched(
            fixture(), legacy_config, batch_baseline_runner, batch_runner,
            trials, selection, trace, error,
            [&](const auto & request, auto & decision, auto & adviser_error) {
                if (!guided_adviser(request, decision, adviser_error)) return false;
                decision.shadow_mode = !fallback;
                if (fallback) decision.fallback_reason = "scripted_storage_failure";
                return true;
            }));
        std::vector<uint32_t> actual_layers;
        for (const auto & trial : trials) actual_layers.push_back(trial.candidate.anchor_layer_index);
        CHECK(actual_layers == legacy_layers);
    }

    common_flydelta_whirlpool_search_config invalid = config;
    invalid.probes_per_round = 1;
    CHECK(!common_flydelta_whirlpool_search_config_validate(invalid, error));

    common_flydelta_whirlpool_search_config retry_config = config;
    retry_config.max_rounds = 1;
    retry_config.max_trials = 2;
    retry_config.dose_policy.max_shift_norm = 1.0f;
    retry_config.dose_policy.max_leakage = 1.0f;
    size_t retry_calls = 0;
    trials.clear();
    selection = {};
    trace = {};
    CHECK(common_flydelta_run_whirlpool_search(
        fixture(), retry_config,
        [&](const auto &, const auto * candidate, auto & trial, auto & margin,
                auto & geometry, auto & geometry_available, std::string &) {
            ++retry_calls;
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.evidence_ref = candidate == nullptr
                ? "evidence:retry-baseline" : "evidence:retry-arm";
            if (candidate == nullptr) return true;
            trial.quality = 0.0f;
            geometry_available = true;
            geometry.layer_index = candidate->anchor_layer_index;
            geometry.cosine = 0.8f;
            geometry.progress = 0.4f;
            geometry.leakage = 0.1f;
            geometry.shift_norm = candidate->total_scale >= 0.02f ? 4.0f : 0.5f;
            margin = {};
            return true;
        }, trials, selection, trace, error));
    CHECK(error.empty());
    CHECK(retry_calls > trials.size() + 1);
    CHECK(!trials.empty());
    CHECK(std::any_of(trials.begin(), trials.end(), [](const auto & trial) {
        return trial.dose_evaluated &&
            trial.dose_action == common_flydelta_dose_action::accept &&
            trial.requested_total_scale > trial.executed_total_scale &&
            trial.dose_safety_limited;
    }));
    CHECK(trace.model_evaluations == retry_calls);
    CHECK(common_flydelta_whirlpool_trace_validate(trace, retry_config, trials.size(), error));

    common_flydelta_whirlpool_search_config missing_geometry_config = config;
    missing_geometry_config.max_rounds = 1;
    missing_geometry_config.max_trials = 2;
    trials.clear();
    selection = {};
    trace = {};
    CHECK(common_flydelta_run_whirlpool_search(
        fixture(), missing_geometry_config,
        [](const auto &, const auto * candidate, auto & trial, auto & margin,
                auto &, auto & geometry_available, std::string &) {
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.quality = candidate == nullptr ? 0.0f : 0.1f;
            trial.evidence_ref = candidate == nullptr
                ? "evidence:missing-geometry-baseline" : "evidence:missing-geometry-arm";
            margin = {};
            geometry_available = false;
            return true;
        }, trials, selection, trace, error));
    CHECK(!trials.empty());
    CHECK(std::all_of(trials.begin(), trials.end(), [](const auto & trial) {
        return !trial.geometry_available && !trial.safe_to_continue;
    }));
    CHECK(!selection.selected);
    CHECK(common_flydelta_whirlpool_trace_validate(
        trace, missing_geometry_config, trials.size(), error));
    return 0;
}
