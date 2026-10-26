#include "agent/adaptation/flydelta/flydelta-deep-search.h"
#include "agent/adaptation/flydelta/flydelta-intervention-region-search.h"

#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

static common_flydelta_experiment_fixture fixture() {
    common_flydelta_experiment_fixture value;
    value.schema_version = 1;
    value.id = "fixture:deep-search";
    value.task_fingerprint = "task";
    value.model_profile_fingerprint = "model";
    value.tokenizer_fingerprint = "tokenizer";
    value.template_fingerprint = "template";
    value.execution_context_fingerprint = "execution";
    value.verifier_revision = "verifier";
    return value;
}

static common_flydelta_direction_candidate direction(
        common_flydelta_direction_kind kind, float x, float y, float alignment) {
    common_flydelta_direction_candidate value;
    value.kind = kind;
    value.layer_index = 2;
    value.values = {x, y};
    value.source_samples = 6;
    value.retained_samples = 6;
    value.median_alignment = alignment;
    value.experimental_only = true;
    return value;
}

int main() {
    std::string error;
    common_flydelta_deep_search_config config;
    config.max_rank = 2;
    config.max_directions = 3;
    config.full_generation_top_k = 2;
    config.coefficients.max_candidates = 8;
    config.coefficients.step = 0.1f;
    config.coefficients.max_l2_norm = 0.32f;

    std::vector<common_flydelta_deep_search_direction> input = {
        {direction(common_flydelta_direction_kind::raw_repair, 1.0f, 0.0f, 0.4f), true, 0.2f},
        {direction(common_flydelta_direction_kind::diagonal_whitened_mean, 0.0f, 1.0f, 0.9f), true, 0.8f},
        {direction(common_flydelta_direction_kind::token_margin_direction, 1.0f, 1.0f, 0.1f), true, 0.1f},
    };
    std::vector<common_flydelta_deep_search_direction> selected;
    CHECK(common_flydelta_select_deep_search_directions(config, input, selected, error));
    CHECK(selected.size() == 2);
    CHECK(selected.front().direction.kind ==
        common_flydelta_direction_kind::diagonal_whitened_mean);

    size_t calls = 0;
    common_flydelta_deep_search_result result;
    CHECK(common_flydelta_run_deep_search(
        fixture(), config, input,
        [&](const common_flydelta_experiment_fixture &,
                const common_flydelta_low_rank_basis & basis,
                const std::vector<float> & coefficients, bool apply_overlay,
                common_flydelta_counterfactual_trial & trial,
                common_flydelta_decision_margin & margin,
                common_flydelta_representation_diagnostics & geometry,
                bool & geometry_available, std::string &) {
            ++calls;
            trial = {};
            trial.executed = true;
            // Diagnostic execution has no generated answer to verify.
            trial.verifier_known = false;
            trial.overlay_applied = apply_overlay;
            trial.evidence_ref = "evidence:deep-search";
            trial.passed = false;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            margin = {};
            margin.available = apply_overlay;
            margin.positive_total_logprob = coefficients.empty() ? 0.0f : coefficients[0];
            margin.negative_total_logprob = 0.0f;
            margin.positive_token_count = 1;
            margin.negative_token_count = 1;
            geometry = {};
            geometry_available = apply_overlay;
            geometry.layer_index = static_cast<uint32_t>(basis.layer_index);
            geometry.cosine = 0.8f;
            geometry.progress = coefficients.empty() ? 0.0f : coefficients[0];
            geometry.leakage = 0.05f;
            geometry.shift_norm = std::fabs(geometry.progress);
            return true;
        },
        [&](const common_flydelta_experiment_fixture &,
                const common_flydelta_low_rank_basis &,
                const std::vector<float> & coefficients, bool apply_overlay,
                common_flydelta_counterfactual_trial & trial,
                common_flydelta_decision_margin & margin,
                common_flydelta_representation_diagnostics & geometry,
                bool & geometry_available, std::string &) {
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.overlay_applied = apply_overlay;
            trial.evidence_ref = "evidence:deep-search-full";
            trial.host_evaluated = true;
            trial.passed = apply_overlay && coefficients.size() == 2 && coefficients[0] > 0.0f;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            margin = {};
            geometry = {};
            geometry_available = false;
            return true;
        }, result, error));
    CHECK(result.layer_index == 2);
    CHECK(result.basis.vectors.size() == 2);
    CHECK(result.coefficient_trials.size() == 6); // four diagnostic + two full top arms
    for (size_t index = 0; index < 4; ++index) {
        CHECK(!result.coefficient_trials[index].verifier_known);
        CHECK(result.coefficient_trials[index].outcome ==
            common_flydelta_counterfactual_outcome::unknown);
    }
    CHECK(result.coefficient_selection.selected);
    CHECK(calls == 5); // diagnostic baseline plus four bounded coefficient arms
    CHECK(result.coefficient_trials[result.coefficient_selection.trial_index].outcome ==
        common_flydelta_counterfactual_outcome::helped);
    const auto & finalist = result.coefficient_trials[result.coefficient_selection.trial_index];
    CHECK(finalist.host_evaluated && finalist.verifier_known);
    CHECK(finalist.requested_strength > 0 && finalist.executed_strength > 0);
    CHECK(finalist.requested_coefficients.size() == finalist.coefficients.size());
    common_flydelta_intervention_region_trial region;
    CHECK(common_flydelta_coefficient_trial_to_region(finalist, result.basis, region, error));
    CHECK(region.host_evaluated && region.verifier_known && region.promising);
    CHECK(region.candidate.total_scale == finalist.executed_strength);
    region.host_evaluated = false;
    CHECK(!common_flydelta_intervention_region_trial_validate(region, error));

    common_flydelta_deep_search_result partial_result;
    const common_flydelta_coefficient_search_runner diagnostic_partial =
        [](const common_flydelta_experiment_fixture &, const common_flydelta_low_rank_basis &,
           const std::vector<float> & coefficients, bool overlay,
           common_flydelta_counterfactual_trial & trial, common_flydelta_decision_margin & margin,
           common_flydelta_representation_diagnostics & geometry, bool & has_geometry, std::string &) {
            trial = {}; trial.executed = true; trial.overlay_applied = overlay;
            margin = {}; margin.available = overlay;
            margin.positive_total_logprob = coefficients.empty() ? 0.0f : coefficients[0];
            margin.positive_token_count = 1; margin.negative_token_count = 1;
            geometry = {}; has_geometry = false; return true;
        };
    const common_flydelta_coefficient_search_runner full_partial =
        [](const common_flydelta_experiment_fixture &, const common_flydelta_low_rank_basis &,
           const std::vector<float> & coefficients, bool overlay,
           common_flydelta_counterfactual_trial & trial, common_flydelta_decision_margin & margin,
           common_flydelta_representation_diagnostics & geometry, bool & has_geometry, std::string &) {
            trial = {}; trial.executed = true; trial.host_evaluated = true;
            trial.verifier_known = true; trial.passed = false; trial.overlay_applied = overlay;
            trial.evidence_ref = "evidence:partial";
            auto & progress = trial.semantic_progress_observation;
            progress.available = true;
            progress.regression_coverage_complete = true;
            progress.operation_state = common_flydelta_semantic_progress_field_state::matched;
            progress.dataset_state = common_flydelta_semantic_progress_field_state::matched;
            progress.grouping_state = common_flydelta_semantic_progress_field_state::matched;
            progress.contract_state = common_flydelta_semantic_progress_field_state::matched;
            const bool improved = overlay && !coefficients.empty() && coefficients[0] > 0;
            progress.measure_state = improved ? common_flydelta_semantic_progress_field_state::matched
                : common_flydelta_semantic_progress_field_state::mismatch;
            progress.score = improved ? 1 : 0;
            margin = {}; geometry = {}; has_geometry = false; return true;
        };
    CHECK(common_flydelta_run_deep_search(
        fixture(), config, input, diagnostic_partial, full_partial, partial_result, error));
    CHECK(!partial_result.coefficient_selection.selected);
    CHECK(partial_result.provisional_selection.selected);
    const auto & partial = partial_result.coefficient_trials[partial_result.provisional_selection.trial_index];
    CHECK(partial.outcome == common_flydelta_counterfactual_outcome::unknown);
    CHECK(partial.semantic_progress.outcome == common_flydelta_semantic_progress_outcome::improved);
    CHECK(partial.semantic_progress.regression_known);
    CHECK(!partial.coefficients.empty());
    const common_flydelta_coefficient_search_runner unknown_regression =
        [full_partial](const common_flydelta_experiment_fixture & f, const common_flydelta_low_rank_basis & b,
           const std::vector<float> & c, bool overlay, common_flydelta_counterfactual_trial & trial,
           common_flydelta_decision_margin & margin, common_flydelta_representation_diagnostics & geometry,
           bool & has_geometry, std::string & e) {
            if (!full_partial(f, b, c, overlay, trial, margin, geometry, has_geometry, e)) return false;
            if (overlay) trial.semantic_progress_observation.grouping_state =
                common_flydelta_semantic_progress_field_state::unknown;
            return true;
        };
    common_flydelta_deep_search_result unknown_result;
    CHECK(common_flydelta_run_deep_search(
        fixture(), config, input, diagnostic_partial, unknown_regression, unknown_result, error));
    CHECK(!unknown_result.provisional_selection.selected);
    const common_flydelta_coefficient_search_runner incomplete_coverage =
        [full_partial](const common_flydelta_experiment_fixture & f, const common_flydelta_low_rank_basis & b,
           const std::vector<float> & c, bool overlay, common_flydelta_counterfactual_trial & trial,
           common_flydelta_decision_margin & margin, common_flydelta_representation_diagnostics & geometry,
           bool & has_geometry, std::string & e) {
            if (!full_partial(f, b, c, overlay, trial, margin, geometry, has_geometry, e)) return false;
            trial.semantic_progress_observation.regression_coverage_complete = false;
            return true;
        };
    common_flydelta_deep_search_result incomplete_result;
    CHECK(common_flydelta_run_deep_search(
        fixture(), config, input, diagnostic_partial, incomplete_coverage, incomplete_result, error));
    CHECK(!incomplete_result.provisional_selection.selected);

    size_t diagnostic_batch_calls = 0;
    size_t full_batch_calls = 0;
    const common_flydelta_coefficient_search_runner batch_probe =
        [](const common_flydelta_experiment_fixture &,
                const common_flydelta_low_rank_basis & basis,
                const std::vector<float> & coefficients, bool apply_overlay,
                common_flydelta_counterfactual_trial & trial,
                common_flydelta_decision_margin & margin,
                common_flydelta_representation_diagnostics & geometry,
                bool & geometry_available, std::string &) {
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.overlay_applied = apply_overlay;
            trial.evidence_ref = "evidence:deep-search-batch";
            trial.host_evaluated = true;
            trial.quality = apply_overlay && !coefficients.empty() && coefficients[0] > 0.0f
                ? 1.0f : 0.0f;
            margin = {};
            margin.available = apply_overlay;
            margin.positive_token_count = 1;
            margin.negative_token_count = 1;
            geometry = {};
            geometry_available = apply_overlay;
            geometry.layer_index = static_cast<uint32_t>(basis.layer_index);
            geometry.cosine = 0.7f;
            geometry.progress = apply_overlay ? coefficients[0] : 0.0f;
            geometry.leakage = 0.02f;
            geometry.shift_norm = std::fabs(geometry.progress);
            return true;
        };
    const auto batch_probe_runner =
        [](size_t * calls, const common_flydelta_coefficient_search_runner & scalar) {
            return [calls, scalar](
                    const common_flydelta_experiment_fixture & current_fixture,
                    const common_flydelta_low_rank_basis & basis,
                    const std::vector<std::vector<float>> & coefficients,
                    std::vector<common_flydelta_counterfactual_trial> & trials,
                    std::vector<common_flydelta_decision_margin> & margins,
                    std::vector<common_flydelta_representation_diagnostics> & geometries,
                    std::vector<bool> & geometry_available, std::string & batch_error) {
                ++(*calls);
                trials.clear();
                margins.clear();
                geometries.clear();
                geometry_available.clear();
                for (const auto & values : coefficients) {
                    common_flydelta_counterfactual_trial trial;
                    common_flydelta_decision_margin margin;
                    common_flydelta_representation_diagnostics geometry;
                    bool has_geometry = false;
                    if (!scalar(current_fixture, basis, values, true, trial, margin,
                            geometry, has_geometry, batch_error)) return false;
                    trials.push_back(std::move(trial));
                    margins.push_back(std::move(margin));
                    geometries.push_back(std::move(geometry));
                    geometry_available.push_back(has_geometry);
                }
                return true;
            };
    };
    common_flydelta_deep_search_result batched_result;
    if (!common_flydelta_run_deep_search_batched(
            fixture(), config, input, batch_probe,
            batch_probe_runner(&diagnostic_batch_calls, batch_probe), batch_probe,
            batch_probe_runner(&full_batch_calls, batch_probe), batched_result, error)) {
        std::cerr << "deep batch search failed: " << error << '\n';
        return __LINE__;
    }
    CHECK(diagnostic_batch_calls == 1);
    CHECK(full_batch_calls == 1);
    CHECK(batched_result.coefficient_trials.size() == result.coefficient_trials.size());

    auto matched_fixture = fixture();
    matched_fixture.guidance_fixture_group_id = "fixture-group:deep-matched";
    auto matched_config = config;
    matched_config.matched_evaluation = true;
    matched_config.matched_max_arms = 9;
    matched_config.coefficients.guidance_basis_identity = "stale-before-deep-basis-selection";
    const common_flydelta_outcome_guidance_adviser matched_adviser =
        [](const common_flydelta_outcome_guidance_request & request,
                common_flydelta_outcome_guidance_decision & decision, std::string &) {
            if (request.stage != "deep_full_generation_frontier") return false;
            decision = {};
            decision.history_fingerprint = "history:frozen-deep-fixture";
            decision.original_order.resize(request.candidates.size());
            std::iota(decision.original_order.begin(), decision.original_order.end(), 0);
            decision.selected_order = decision.original_order; // shadow actually runs ordinary K
            decision.mixed_policy_order = decision.original_order;
            std::reverse(decision.mixed_policy_order.begin(), decision.mixed_policy_order.end());
            return true;
        };
    common_flydelta_deep_search_result matched_result;
    CHECK(common_flydelta_run_deep_search_batched(matched_fixture, matched_config, input,
        batch_probe, batch_probe_runner(&diagnostic_batch_calls, batch_probe),
        batch_probe, batch_probe_runner(&full_batch_calls, batch_probe),
        matched_result, error, matched_adviser));
    CHECK(matched_result.matched_evaluation_complete);
    CHECK(full_batch_calls == 2); // one additional union batch, not three separate policy runs
    CHECK(matched_result.matched_evaluation.union_arm_results.size() == 4);
    CHECK(matched_result.coefficient_trials.size() == batched_result.coefficient_trials.size());
    CHECK(matched_result.matched_evaluation.baseline_selection.size() == config.full_generation_top_k);
    for (size_t i = 4; i < matched_result.coefficient_trials.size(); ++i) {
        const auto & trial = matched_result.coefficient_trials[i];
        CHECK(trial.coefficients == batched_result.coefficient_trials[i].coefficients);
        CHECK(trial.mutation_kind == "full_generation_top_arm");
        CHECK(trial.guidance_basis_id.find("stale-before") == std::string::npos);
    }
    matched_config.matched_max_arms = 2;
    CHECK(common_flydelta_run_deep_search_batched(matched_fixture, matched_config, input,
        batch_probe, batch_probe_runner(&diagnostic_batch_calls, batch_probe),
        batch_probe, batch_probe_runner(&full_batch_calls, batch_probe),
        matched_result, error, matched_adviser));
    CHECK(!matched_result.matched_evaluation_complete);
    CHECK(matched_result.matched_evaluation_reason == "matched_union_arm_budget_exceeded");
    CHECK(matched_result.coefficient_trials.size() == batched_result.coefficient_trials.size());

    common_learning_in_memory_lifecycle_store lifecycle;
    common_flydelta_lifecycle_event_context context;
    context.event_id = "event:deep-search";
    context.idempotency_key = "idempotency:deep-search";
    context.source_id = "source:deep-search";
    context.scope.namespace_id = "local";
    context.scope.project_id = "agent-tests";
    context.scope.session_id = "session-deep-search";
    context.content_hash = "sha256:deep-search";
    context.created_at = "2026-09-16T00:00:00Z";
    CHECK(common_flydelta_append_deep_search_lifecycle(
        lifecycle, context, fixture(), config, result,
        "flydelta://sideband/deep-search", error));
    auto records = lifecycle.list(error);
    CHECK(error.empty() && records.size() == 2); // only the two full-generation finalists
    CHECK(records.back().payload_json.find("full_generation_top_arm") != std::string::npos);
    return 0;
}
