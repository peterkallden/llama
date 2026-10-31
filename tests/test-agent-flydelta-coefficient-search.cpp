#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d\\n", __FILE__, __LINE__); \
        return __LINE__; \
    } \
} while (false)

static common_flydelta_direction_candidate direction(int layer, float first, float second) {
    common_flydelta_direction_candidate value;
    value.kind = common_flydelta_direction_kind::raw_repair;
    value.layer_index = layer;
    value.values = {first, second};
    value.source_samples = 1;
    value.retained_samples = 1;
    value.median_alignment = 1.0f;
    return value;
}

static common_flydelta_synthesis_candidate_descriptor descriptor(
        const char * candidate_ref, bool host_verified = true) {
    common_flydelta_synthesis_candidate_descriptor value;
    value.candidate_ref = candidate_ref;
    value.concept_key = "dataset.grouped_sum";
    value.behavior_key = "dataset.grouped_sum";
    value.source_ref = "teaching://relation/grouped-sum";
    value.oracle_ref = "oracle://dataset-operation";
    value.oracle_revision = "oracle:v1";
    value.model_profile_fingerprint = "sha256:model";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.template_fingerprint = "sha256:template";
    value.capture_layout_revision = "layer-input:v1";
    value.scope_fingerprint = "sha256:scope";
    value.host_verified = host_verified;
    value.experimental_only = false;
    return value;
}

static common_flydelta_experiment_fixture fixture() {
    common_flydelta_experiment_fixture value;
    value.id = "flydelta://fixture/coefficient";
    value.task_fingerprint = "sha256:task";
    value.model_profile_fingerprint = "sha256:model";
    value.tokenizer_fingerprint = "sha256:tokenizer";
    value.template_fingerprint = "sha256:template";
    value.execution_context_fingerprint = "sha256:context";
    value.verifier_revision = "verifier:v1";
    return value;
}

int main() {
    std::string error;
    common_flydelta_search_execution_budget budget;
    budget.limit = 10;
    budget.used = 4;
    CHECK(common_flydelta_reserve_search_evaluations(budget, 6, error));
    CHECK(budget.used == 10);
    CHECK(!common_flydelta_reserve_search_evaluations(budget, 1, error));
    CHECK(budget.used == 10);
    // Partial host evidence must survive the coefficient adapter without
    // turning an incomplete result into a verified success.
    common_flydelta_counterfactual_trial baseline_progress, candidate_progress;
    baseline_progress.executed = candidate_progress.executed = true;
    baseline_progress.host_evaluated = candidate_progress.host_evaluated = true;
    baseline_progress.verifier_known = candidate_progress.verifier_known = true;
    baseline_progress.semantic_progress_observation.available = true;
    candidate_progress.semantic_progress_observation.available = true;
    baseline_progress.semantic_progress_observation.measure_state =
        common_flydelta_semantic_progress_field_state::mismatch;
    candidate_progress.semantic_progress_observation.measure_state =
        common_flydelta_semantic_progress_field_state::matched;
    candidate_progress.semantic_progress_observation.score = 1;
    common_flydelta_coefficient_trial partial_trial;
    common_flydelta_coefficient_trial_set_observation(
        partial_trial, baseline_progress, candidate_progress);
    CHECK(partial_trial.outcome == common_flydelta_counterfactual_outcome::unknown);
    CHECK(partial_trial.semantic_progress.comparable);
    CHECK(partial_trial.semantic_progress.outcome == common_flydelta_semantic_progress_outcome::improved);
    CHECK(!partial_trial.semantic_progress.improved_dimensions.empty());
    candidate_progress.host_evaluated = false;
    common_flydelta_coefficient_trial_set_observation(
        partial_trial, baseline_progress, candidate_progress);
    CHECK(!partial_trial.semantic_progress.comparable);
    candidate_progress.host_evaluated = true;
    baseline_progress.semantic_progress_observation.grouping_state =
        common_flydelta_semantic_progress_field_state::matched;
    candidate_progress.semantic_progress_observation.grouping_state =
        common_flydelta_semantic_progress_field_state::missing;
    common_flydelta_coefficient_trial_set_observation(
        partial_trial, baseline_progress, candidate_progress);
    CHECK(!partial_trial.semantic_progress.improved_dimensions.empty());
    CHECK(!partial_trial.semantic_progress.regressed_dimensions.empty());
    common_flydelta_low_rank_basis basis;
    CHECK(common_flydelta_build_low_rank_basis(
        2, 2, {direction(2, 1.0f, 0.0f), direction(2, 0.0f, 1.0f)}, basis, error));
    CHECK(basis.vectors.size() == 2);
    CHECK(std::fabs(basis.vectors[0][0] * basis.vectors[1][0] +
        basis.vectors[0][1] * basis.vectors[1][1]) < 0.0001f);
    const auto basis_id = common_flydelta_coefficient_basis_identity(basis);
    std::vector<float> reconstructed;
    CHECK(common_flydelta_reconstruct_intervention(
        basis, basis_id, {0.12f, 0.03f}, reconstructed, error));
    CHECK(reconstructed.size() == 2);
    CHECK(std::fabs(reconstructed[0] - 0.12f) < 0.00001f);
    CHECK(std::fabs(reconstructed[1] - 0.03f) < 0.00001f);
    CHECK(!common_flydelta_reconstruct_intervention(
        basis, "different-basis", {0.12f, 0.03f}, reconstructed, error));
    CHECK(reconstructed.empty());

    // Experimental captures can span a basis without earning learning credit;
    // repeated/collinear captures must not inflate the measured rank.
    common_flydelta_low_rank_basis measured;
    auto repeated = direction(2, 1.0f, 0.0f);
    repeated.experimental_only = true;
    CHECK(common_flydelta_build_low_rank_basis(2, 4,
        {repeated, direction(2, 2.0f, 0.0f)}, measured, error));
    CHECK(measured.vectors.size() == 1);
    std::vector<common_flydelta_direction_candidate> independent;
    for (size_t component = 0; component < 4; ++component) {
        auto value = repeated;
        value.values.assign(4, 0.0f);
        value.values[component] = 1.0f;
        independent.push_back(std::move(value));
    }
    CHECK(common_flydelta_build_low_rank_basis(4, 4, independent, measured, error));
    CHECK(measured.vectors.size() == 4);

    common_flydelta_semantic_basis_query query;
    query.concept_key = "dataset.grouped_sum";
    query.behavior_key = "dataset.grouped_sum";
    query.model_profile_fingerprint = "sha256:model";
    query.tokenizer_fingerprint = "sha256:tokenizer";
    query.template_fingerprint = "sha256:template";
    query.capture_layout_revision = "layer-input:v1";
    query.scope_fingerprint = "sha256:scope";
    query.oracle_ref = "oracle://dataset-operation";
    query.oracle_revision = "oracle:v1";
    query.max_rank = 2;
    std::vector<common_flydelta_semantic_direction_candidate> semantic_candidates;
    auto first = direction(2, 1.0f, 0.0f);
    auto second = direction(2, 0.0f, 1.0f);
    auto incompatible = direction(2, 1.0f, 1.0f);
    semantic_candidates.push_back({first, descriptor("direction://grouped")});
    semantic_candidates.push_back({second, descriptor("direction://aggregate")});
    auto wrong = descriptor("direction://wrong");
    wrong.oracle_revision = "oracle:old";
    semantic_candidates.push_back({incompatible, wrong});
    common_flydelta_low_rank_basis semantic_basis;
    std::vector<size_t> selected_indices;
    CHECK(common_flydelta_resolve_semantic_basis(
        query, semantic_candidates, semantic_basis, selected_indices, error));
    CHECK(selected_indices.size() == 2 && selected_indices[0] == 0 && selected_indices[1] == 1);
    CHECK(semantic_basis.vectors.size() == 2);
    auto no_verified = semantic_candidates;
    for (auto & candidate : no_verified) candidate.descriptor.host_verified = false;
    CHECK(!common_flydelta_resolve_semantic_basis(
        query, no_verified, semantic_basis, selected_indices, error));

    common_flydelta_decision_margin margin;
    margin.available = true;
    margin.positive_total_logprob = -1.0f;
    margin.negative_total_logprob = -4.0f;
    margin.positive_token_count = 2;
    margin.negative_token_count = 4;
    CHECK(common_flydelta_decision_margin_validate(margin, error));
    CHECK(std::fabs(margin.total_delta() - 3.0f) < 0.0001f);
    CHECK(std::fabs(margin.normalized_delta() - 0.5f) < 0.0001f);

    common_flydelta_coefficient_search_config config;
    config.step = 0.05f;
    config.max_candidates = 5;
    std::vector<std::vector<float>> proposals;
    CHECK(common_flydelta_propose_low_rank_coefficients(config, 2, proposals, error));
    CHECK(proposals.size() == 5 && proposals[0][0] == 0.0f);
    CHECK(common_flydelta_propose_shallow_rank_two_controls(config, true, proposals, error));
    CHECK(proposals.size() == 4);
    CHECK(std::fabs(proposals[0][0] - config.step) < 0.0001f && proposals[0][1] == 0.0f);
    CHECK(proposals[1][0] == 0.0f && std::fabs(proposals[1][1] - config.step) < 0.0001f);
    CHECK(std::fabs(proposals[2][0] - proposals[2][1]) < 0.0001f);
    CHECK(std::fabs(proposals[3][0] + proposals[3][1]) < 0.0001f);
    CHECK(std::string(common_flydelta_coefficient_search_strategy_name(
        common_flydelta_coefficient_search_strategy::coordinate)) == "coordinate");
    config.population_size = 0;
    config.iterations = 0;
    CHECK(common_flydelta_coefficient_search_config_validate(config, 2, error));
    config.max_batch_arms = 65;
    CHECK(!common_flydelta_coefficient_search_config_validate(config, 2, error));
    config.max_batch_arms = 0;
    CHECK(common_flydelta_coefficient_search_config_validate(config, 2, error));
    config.strategy = common_flydelta_coefficient_search_strategy::tfo_lite;
    config.population_size = 3;
    config.iterations = 2;
    config.seed = 17;
    CHECK(common_flydelta_coefficient_search_config_validate(config, 2, error));

    // Anchored proposals are complete interventions and keep an independent
    // search arm. The unsteered baseline remains a separate invocation.
    common_flydelta_coefficient_search_config anchored;
    anchored.max_candidates = 5;
    anchored.use_dose_controller = false;
    anchored.anchor_coefficients = {0.12f, 0.03f};
    anchored.anchor_basis_identity = common_flydelta_coefficient_basis_identity(basis);
    std::vector<std::vector<float>> executed_arms;
    bool saw_unsteered = false;
    const common_flydelta_coefficient_search_runner anchor_runner =
        [&](const auto &, const auto &, const auto & values, bool overlay,
                auto & trial, auto & result, auto &, auto &, std::string &) {
            trial.executed = true; trial.host_evaluated = true;
            trial.verifier_known = true; trial.overlay_applied = overlay;
            trial.evidence_ref = "evidence:anchor";
            if (overlay) executed_arms.push_back(values); else saw_unsteered = true;
            result = {}; return true;
        };
    std::vector<common_flydelta_coefficient_trial> anchored_trials;
    common_flydelta_coefficient_selection anchored_selection;
    CHECK(common_flydelta_run_low_rank_coefficient_search(
        fixture(), basis, anchored, anchor_runner, anchored_trials, anchored_selection, error));
    CHECK(saw_unsteered && executed_arms.size() >= 3);
    CHECK(executed_arms.front() == anchored.anchor_coefficients);
    CHECK(std::find(executed_arms.begin(), executed_arms.end(), std::vector<float>{0.0f, 0.05f}) !=
        executed_arms.end());
    anchored.evaluated_coefficients = {{0.0f, 0.05f}, {0.0f, -0.05f}};
    executed_arms.clear();
    CHECK(common_flydelta_run_low_rank_coefficient_search(
        fixture(), basis, anchored, anchor_runner, anchored_trials, anchored_selection, error));
    CHECK(executed_arms.front() == anchored.anchor_coefficients);
    CHECK(std::find(executed_arms.begin(), executed_arms.end(), std::vector<float>{0.0f, 0.05f}) == executed_arms.end());
    anchored.anchor_basis_identity = "different-basis";
    CHECK(!common_flydelta_run_low_rank_coefficient_search(
        fixture(), basis, anchored, anchor_runner, anchored_trials, anchored_selection, error));

    std::vector<common_flydelta_coefficient_trial> trials;
    common_flydelta_coefficient_selection selection;
    CHECK(common_flydelta_run_low_rank_coefficient_search(
        fixture(), basis, config,
        [](const auto &, const auto &, const auto & coefficients, bool apply,
                auto & trial, auto & result, auto &, auto &, std::string &) {
            trial.executed = true;
            trial.verifier_known = true;
            trial.host_evaluated = true;
            trial.evidence_ref = "evidence://coefficient-trial";
            trial.passed = apply && coefficients[0] > 0.0f;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            result.available = true;
            result.positive_total_logprob = apply ? -1.0f : -2.0f;
            result.negative_total_logprob = -2.0f;
            result.positive_token_count = 1;
            result.negative_token_count = 1;
            return true;
        }, trials, selection, error));
    CHECK(selection.selected);
    CHECK(trials[selection.trial_index].outcome == common_flydelta_counterfactual_outcome::helped);

    common_flydelta_coefficient_search_config tfo_config;
    tfo_config.strategy = common_flydelta_coefficient_search_strategy::tfo_lite;
    tfo_config.step = 0.05f;
    tfo_config.max_candidates = 6;
    tfo_config.population_size = 3;
    tfo_config.iterations = 2;
    tfo_config.seed = 23;
    std::vector<common_flydelta_coefficient_trial> tfo_trials;
    common_flydelta_coefficient_selection tfo_selection;
    CHECK(common_flydelta_run_low_rank_coefficient_search(
        fixture(), basis, tfo_config,
        [](const auto &, const auto &, const auto & coefficients, bool apply,
                auto & trial, auto & result, auto & geometry, auto & geometry_available,
                std::string &) {
            trial.executed = true;
            trial.verifier_known = true;
            trial.host_evaluated = true;
            trial.evidence_ref = "evidence://coefficient-tfo-trial";
            trial.passed = apply && coefficients[0] > 0.0f;
            trial.quality = trial.passed ? 1.0f : 0.0f;
            geometry_available = apply;
            geometry.layer_index = 2;
            geometry.cosine = 0.8f;
            geometry.progress = 0.3f;
            geometry.leakage = std::fabs(coefficients[1]);
            geometry.shift_norm = std::fabs(coefficients[0]);
            result.available = true;
            result.positive_total_logprob = coefficients[0] + coefficients[1];
            result.negative_total_logprob = 0.0f;
            result.positive_token_count = 1;
            result.negative_token_count = 1;
            return true;
        }, tfo_trials, tfo_selection, error));
    CHECK(tfo_trials.size() >= 3 && tfo_trials.size() <= 6);
    CHECK(tfo_selection.selected);
    bool found_mixed_arm = false;
    for (const auto & trial : tfo_trials) {
        CHECK(trial.dose_evaluated);
        CHECK(trial.dose_action == common_flydelta_dose_action::accept);
        found_mixed_arm = found_mixed_arm ||
            (std::fabs(trial.coefficients[0]) > 0.0001f &&
             std::fabs(trial.coefficients[1]) > 0.0001f);
        CHECK(trial.mutation_kind == "initial_coordinate" ||
              trial.mutation_kind == "initial_mixed" ||
              trial.mutation_kind == "forage_anchor" ||
              trial.mutation_kind == "forage_perturbation");
    }
    CHECK(found_mixed_arm);
    CHECK(std::any_of(tfo_trials.begin(), tfo_trials.end(), [](const auto & trial) {
        return trial.geometry_available && trial.geometry.leakage >= 0.0f;
    }));
    for (size_t i = 0; i < tfo_trials.size(); ++i) {
        if (tfo_trials[i].arm_id.empty()) tfo_trials[i].arm_id = "tfo-guidance-arm-" +
            std::to_string(i);
        tfo_trials[i].overlay_applied = true;
    }

    common_flydelta_coefficient_search_config batch_config = tfo_config;
    batch_config.strategy = common_flydelta_coefficient_search_strategy::coordinate;
    batch_config.max_candidates = 5;
    batch_config.max_batch_arms = 2;
    batch_config.use_dose_controller = false;
    size_t coefficient_batch_calls = 0;
    size_t largest_coefficient_batch = 0;
    std::vector<common_flydelta_coefficient_trial> batch_trials_result;
    common_flydelta_coefficient_selection batch_selection;
    const common_flydelta_coefficient_search_runner batch_baseline_runner =
        [](const auto &, const auto &, const auto &, bool apply,
                auto & trial, auto & margin, auto &, auto & geometry_available, std::string &) {
            if (apply) return false;
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.host_evaluated = true;
            trial.evidence_ref = "evidence://coefficient-batch-baseline";
            margin = {};
            geometry_available = false;
            return true;
        };
    const common_flydelta_coefficient_search_batch_runner coefficient_batch_runner =
        [&](const auto &, const auto &, const auto & coefficients,
                auto & batch_counterfactuals, auto & batch_margins,
                auto & batch_geometries, auto & batch_geometry_available, std::string &) {
            ++coefficient_batch_calls;
            largest_coefficient_batch = std::max(
                largest_coefficient_batch, coefficients.size());
            batch_counterfactuals.clear();
            batch_margins.clear();
            batch_geometries.clear();
            batch_geometry_available.clear();
            for (const auto & values : coefficients) {
                common_flydelta_counterfactual_trial trial;
                trial.executed = true;
                trial.verifier_known = true;
                trial.host_evaluated = true;
                trial.evidence_ref = "evidence://coefficient-batch";
                trial.passed = values[0] > 0.0f;
                trial.quality = trial.passed ? 1.0f : 0.0f;
                common_flydelta_decision_margin margin;
                common_flydelta_representation_diagnostics geometry;
                batch_counterfactuals.push_back(std::move(trial));
                batch_margins.push_back(std::move(margin));
                batch_geometries.push_back(std::move(geometry));
                batch_geometry_available.push_back(false);
            }
            return true;
        };
    CHECK(common_flydelta_run_low_rank_coefficient_search_batched(
        fixture(), basis, batch_config, batch_baseline_runner,
        coefficient_batch_runner, batch_trials_result, batch_selection, error));
    CHECK(coefficient_batch_calls == 2);
    CHECK(largest_coefficient_batch <= batch_config.max_batch_arms);
    CHECK(batch_trials_result.size() == 4);
    CHECK(batch_selection.selected);
    for (const auto & trial : batch_trials_result) {
        CHECK(!trial.intervention_norms_available);
        CHECK(trial.requested_intervention_norm == 0);
    }

    auto calibrated_config = batch_config;
    calibrated_config.experimental_dose_mode = common_flydelta_coefficient_dose_mode::direction_dose;
    calibrated_config.max_candidates = 12;
    calibrated_config.max_dose_retries = 0;
    auto & calibration = calibrated_config.residual_calibration;
    calibration.schema_version = 2;
    calibration.reference = "calibration:coefficient-contract";
    calibration.model_profile_fingerprint = fixture().model_profile_fingerprint;
    calibration.tokenizer_fingerprint = fixture().tokenizer_fingerprint;
    calibration.template_fingerprint = fixture().template_fingerprint;
    calibration.capture_layout_revision = "layer-input:v1";
    calibration.token_selection = "generation_boundary";
    calibration.layer_index = basis.layer_index;
    calibration.sample_count = 1;
    calibration.dimension = basis.dimension;
    calibration.sample_norms = {5.0f};
    calibration.median_norm = 5.0f;
    calibration.intervention_site = "layer-input:additive-cvec:v1";
    calibration.measurement_stage = "final_prompt_row";
    calibration.runtime_configuration_fingerprint = "runtime:test";
    calibration.runtime_layout_revision = "layout:test";
    calibration.dataset_revision = "dataset:test:v1";
    calibration.aggregation = "median_of_sequence_medians:v1";
    calibration.sequence_ids = {"sequence:test"};
    calibrated_config.injection_intervention_site = calibration.intervention_site;
    calibrated_config.injection_runtime_configuration_fingerprint = calibration.runtime_configuration_fingerprint;
    calibrated_config.injection_runtime_layout_revision = calibration.runtime_layout_revision;
    calibrated_config.injection_capture_layout_revision = calibration.capture_layout_revision;
    calibrated_config.injection_token_selection = calibration.token_selection;
    CHECK(common_flydelta_run_low_rank_coefficient_search_batched(
        fixture(), basis, calibrated_config, batch_baseline_runner,
        coefficient_batch_runner, batch_trials_result, batch_selection, error));
    const auto coarse_trial_count = batch_trials_result.size();
    for (const auto & trial : batch_trials_result) {
        CHECK(trial.intervention_norms_available);
        CHECK(!trial.overlay_applied);
        CHECK(trial.applied_intervention_norm == 0);
        CHECK(trial.applied_residual_fraction == 0);
    }
    const common_flydelta_coefficient_search_batch_runner calibrated_batch_runner =
        [&](const auto & current_fixture, const auto & current_basis, const auto & coefficients,
                auto & results, auto & margins, auto & geometries, auto & available, std::string & current_error) {
            if (!coefficient_batch_runner(current_fixture, current_basis, coefficients,
                    results, margins, geometries, available, current_error)) return false;
            for (size_t i = 0; i < margins.size(); ++i) {
                results[i].overlay_applied = true;
                margins[i].available = true;
                margins[i].positive_total_logprob = coefficients[i][0] - 1.0f;
                margins[i].negative_total_logprob = -2.0f;
                margins[i].positive_token_count = margins[i].negative_token_count = 1;
            }
            return true;
        };
    CHECK(common_flydelta_run_low_rank_coefficient_search_batched(
        fixture(), basis, calibrated_config, batch_baseline_runner,
        calibrated_batch_runner, batch_trials_result, batch_selection, error));
    CHECK(batch_trials_result.size() >= coarse_trial_count);
    CHECK(std::any_of(batch_trials_result.begin(), batch_trials_result.end(), [](const auto & trial) {
        return trial.mutation_kind == "dose_refinement";
    }));
    CHECK(batch_trials_result.size() < calibrated_config.max_candidates);
    CHECK(batch_selection.selected);
    for (const auto & trial : batch_trials_result) {
        CHECK(trial.intervention_norms_available);
        CHECK(trial.residual_calibration_ref == calibration.reference);
        CHECK(trial.requested_residual_fraction > 0);
        CHECK(trial.applied_residual_fraction > 0);
        CHECK(std::fabs(trial.applied_intervention_norm /
            calibration.median_norm - trial.applied_residual_fraction) < 1e-6f);
    }
    auto capped_config = calibrated_config;
    capped_config.max_l2_norm = 0.01f;
    capped_config.residual_fraction_candidates = {0.1f};
    CHECK(common_flydelta_run_low_rank_coefficient_search_batched(
        fixture(), basis, capped_config, batch_baseline_runner,
        calibrated_batch_runner, batch_trials_result, batch_selection, error));
    CHECK(!batch_trials_result.empty());
    bool saw_clipped_proposal = false;
    for (const auto & trial : batch_trials_result) {
        CHECK(trial.proposal_norm_limit_known);
        if (trial.proposal_norm_limited) {
            saw_clipped_proposal = true;
            CHECK(trial.proposed_intervention_norm < trial.requested_intervention_norm);
        }
        CHECK(std::fabs(trial.proposed_intervention_norm - trial.applied_intervention_norm) < 1e-6f);
        CHECK(!trial.dose_safety_limited);
    }
    CHECK(saw_clipped_proposal);
    const common_flydelta_coefficient_search_batch_runner calibrated_tfo_runner =
        [&](const auto & current_fixture, const auto & current_basis, const auto & coefficients,
                auto & results, auto & margins, auto & geometries, auto & available, std::string & current_error) {
            if (!calibrated_batch_runner(current_fixture, current_basis, coefficients,
                    results, margins, geometries, available, current_error)) return false;
            for (size_t i = 0; i < coefficients.size(); ++i) {
                float magnitude = 0;
                if (!common_flydelta_intervention_norm(current_basis, coefficients[i], magnitude, current_error)) return false;
                geometries[i] = {1, static_cast<uint32_t>(current_basis.layer_index), 0.8f, 0.1f, 0.0f, magnitude * 2};
                available[i] = true;
            }
            return true;
        };
    auto calibrated_tfo = calibrated_config;
    calibrated_tfo.strategy = common_flydelta_coefficient_search_strategy::tfo_lite;
    calibrated_tfo.population_size = 4;
    calibrated_tfo.iterations = 2;
    calibrated_tfo.max_candidates = 8;
    calibrated_tfo.use_dose_controller = true;
    calibrated_tfo.max_dose_retries = 1;
    calibrated_tfo.dose_policy.max_shift_norm = 0.1f;
    CHECK(common_flydelta_run_low_rank_coefficient_search_batched(
        fixture(), basis, calibrated_tfo, batch_baseline_runner,
        calibrated_tfo_runner, batch_trials_result, batch_selection, error));
    CHECK(!batch_trials_result.empty());
    CHECK(std::any_of(batch_trials_result.begin(), batch_trials_result.end(), [](const auto & trial) {
        return trial.iteration == 1;
    }));
    for (const auto & trial : batch_trials_result) {
        CHECK(trial.intervention_norms_available);
        CHECK(trial.residual_calibration_ref == calibration.reference);
        CHECK(trial.requested_residual_fraction > 0);
        CHECK(std::fabs(trial.applied_intervention_norm / calibration.median_norm -
            trial.applied_residual_fraction) < 1e-6f);
    }
    CHECK(std::any_of(batch_trials_result.begin(), batch_trials_result.end(), [](const auto & trial) {
        return trial.dose_safety_limited && trial.dose_action == common_flydelta_dose_action::accept &&
            trial.applied_residual_fraction < trial.requested_residual_fraction;
    }));
    bool saw_safe_parent = false;
    for (const auto & trial : batch_trials_result) {
        if (trial.iteration == 0 || trial.parent_trial_index >= batch_trials_result.size()) continue;
        const auto & parent = batch_trials_result[trial.parent_trial_index];
        CHECK(parent.executed && parent.overlay_applied);
        CHECK(!parent.dose_evaluated || parent.dose_action == common_flydelta_dose_action::accept);
        saw_safe_parent = true;
    }
    CHECK(saw_safe_parent);
    const auto anchor = batch_trials_result.front().coefficients;
    calibrated_tfo.anchor_coefficients = anchor;
    calibrated_tfo.anchor_basis_identity = common_flydelta_coefficient_basis_identity(basis);
    calibrated_tfo.evaluated_coefficients = {anchor};
    CHECK(common_flydelta_run_low_rank_coefficient_search_batched(
        fixture(), basis, calibrated_tfo, batch_baseline_runner,
        calibrated_tfo_runner, batch_trials_result, batch_selection, error));
    CHECK(!batch_trials_result.empty());
    CHECK(batch_trials_result.front().requested_coefficients == anchor);
    calibrated_tfo.anchor_basis_identity = "invalid-basis";
    CHECK(!common_flydelta_run_low_rank_coefficient_search_batched(
        fixture(), basis, calibrated_tfo, batch_baseline_runner,
        calibrated_tfo_runner, batch_trials_result, batch_selection, error));
    calibrated_tfo.anchor_coefficients.clear();
    calibrated_tfo.anchor_basis_identity.clear();
    calibrated_tfo.residual_calibration.layer_index = basis.layer_index + 1;
    CHECK(!common_flydelta_run_low_rank_coefficient_search_batched(
        fixture(), basis, calibrated_tfo, batch_baseline_runner,
        calibrated_tfo_runner, batch_trials_result, batch_selection, error));

    calibrated_config.residual_calibration.layer_index = basis.layer_index + 1;
    const size_t before_invalid = coefficient_batch_calls;
    CHECK(!common_flydelta_run_low_rank_coefficient_search_batched(
        fixture(), basis, calibrated_config, batch_baseline_runner,
        coefficient_batch_runner, batch_trials_result, batch_selection, error));
    CHECK(coefficient_batch_calls == before_invalid);

    common_flydelta_coefficient_search_config batched_tfo_config = tfo_config;
    batched_tfo_config.max_candidates = 5;
    batched_tfo_config.max_batch_arms = 2;
    size_t tfo_guidance_calls = 0;
    const common_flydelta_outcome_guidance_adviser identity_adviser =
        [&tfo_guidance_calls](const auto & request, auto & decision, std::string &) {
            ++tfo_guidance_calls;
            const auto & pool = request.candidates;
            const size_t limit = request.selection_limit;
            decision.original_order.resize(pool.size());
            decision.shadow_order.resize(pool.size());
            decision.selected_order.resize(pool.size());
            decision.selection_reasons.assign(pool.size(), "test_identity");
            for (size_t i = 0; i < pool.size(); ++i) {
                decision.original_order[i] = i;
                decision.shadow_order[i] = i;
                decision.selected_order[i] = i;
            }
            return limit <= pool.size();
        };
    size_t tfo_batch_calls = 0;
    const common_flydelta_coefficient_search_batch_runner tfo_batch_runner =
        [&tfo_batch_calls](const auto &, const auto &, const auto & coefficients,
                auto & batch_counterfactuals, auto & batch_margins,
                auto & batch_geometries, auto & batch_geometry_available, std::string &) {
            ++tfo_batch_calls;
            batch_counterfactuals.clear();
            batch_margins.clear();
            batch_geometries.clear();
            batch_geometry_available.clear();
            for (const auto & values : coefficients) {
                common_flydelta_counterfactual_trial trial;
                trial.executed = true;
                trial.host_evaluated = true;
                trial.verifier_known = true;
                trial.overlay_applied = true;
                trial.evidence_ref = "evidence://coefficient-batched-tfo";
                trial.passed = values[0] > 0.0f;
                batch_counterfactuals.push_back(std::move(trial));
                batch_margins.emplace_back();
                batch_geometries.emplace_back();
                batch_geometry_available.push_back(false);
            }
            return true;
        };
    std::vector<common_flydelta_coefficient_trial> batched_tfo_trials;
    common_flydelta_coefficient_selection batched_tfo_selection;
    const bool batched_tfo_ok = common_flydelta_run_low_rank_coefficient_search_batched(
        fixture(), basis, batched_tfo_config, batch_baseline_runner,
        tfo_batch_runner, batched_tfo_trials, batched_tfo_selection, error,
        identity_adviser);
    if (!batched_tfo_ok) std::fprintf(stderr, "batched TFO failed: %s\\n", error.c_str());
    CHECK(batched_tfo_ok);
    CHECK(tfo_guidance_calls > 0 && tfo_batch_calls > 0);
    CHECK(batched_tfo_trials.size() <= batched_tfo_config.max_candidates);

    common_flydelta_coefficient_search_config staged_config = batch_config;
    size_t staged_diagnostic_batch_calls = 0;
    size_t staged_full_batch_calls = 0;
    size_t staged_full_arm_count = 0;
    size_t staged_guidance_calls = 0;
    const common_flydelta_coefficient_search_runner staged_diagnostic_runner =
        [](const auto &, const auto &, const auto &, bool apply,
                auto & trial, auto & margin, auto &, auto & geometry_available, std::string &) {
            trial = {};
            trial.executed = true;
            trial.verifier_known = false;
            trial.passed = false;
            trial.quality = 0.0f;
            margin = {};
            geometry_available = false;
            if (!apply) return true;
            return true;
        };
    const common_flydelta_coefficient_search_batch_runner staged_diagnostic_batch_runner =
        [&](const auto &, const auto &, const auto & coefficients,
                auto & batch_counterfactuals, auto & batch_margins,
                auto & batch_geometries, auto & batch_geometry_available, std::string &) {
            ++staged_diagnostic_batch_calls;
            batch_counterfactuals.clear();
            batch_margins.clear();
            batch_geometries.clear();
            batch_geometry_available.clear();
            for (const auto & values : coefficients) {
                common_flydelta_counterfactual_trial trial;
                trial.executed = true;
                trial.verifier_known = false;
                trial.passed = false;
                trial.quality = 0.0f;
                common_flydelta_decision_margin margin;
                margin.available = true;
                margin.positive_total_logprob = values[0];
                margin.negative_total_logprob = 0.0f;
                margin.positive_token_count = 1;
                margin.negative_token_count = 1;
                batch_counterfactuals.push_back(std::move(trial));
                batch_margins.push_back(std::move(margin));
                batch_geometries.emplace_back();
                batch_geometry_available.push_back(false);
            }
            return true;
        };
    const common_flydelta_coefficient_search_runner staged_full_runner =
        [](const auto &, const auto &, const auto &, bool apply,
                auto & trial, auto & margin, auto &, auto & geometry_available, std::string &) {
            (void) apply;
            trial = {};
            trial.executed = true;
            trial.verifier_known = true;
            trial.host_evaluated = true;
            trial.evidence_ref = "evidence://coefficient-staged-baseline";
            trial.passed = false;
            trial.quality = 0.0f;
            margin = {};
            geometry_available = false;
            return true;
        };
    const common_flydelta_coefficient_search_batch_runner staged_full_batch_runner =
        [&](const auto &, const auto &, const auto & coefficients,
                auto & batch_counterfactuals, auto & batch_margins,
                auto & batch_geometries, auto & batch_geometry_available, std::string &) {
            ++staged_full_batch_calls;
            staged_full_arm_count += coefficients.size();
            batch_counterfactuals.clear();
            batch_margins.clear();
            batch_geometries.clear();
            batch_geometry_available.clear();
            for (const auto & values : coefficients) {
                common_flydelta_counterfactual_trial trial;
                trial.executed = true;
                trial.verifier_known = true;
                trial.host_evaluated = true;
                trial.evidence_ref = "evidence://coefficient-staged-full";
                trial.passed = true;
                trial.quality = values[0] > 0.0f ? 1.0f : 0.5f;
                batch_counterfactuals.push_back(std::move(trial));
                batch_margins.emplace_back();
                batch_geometries.emplace_back();
                batch_geometry_available.push_back(false);
            }
            return true;
        };
    std::vector<common_flydelta_coefficient_trial> staged_trials;
    common_flydelta_coefficient_selection staged_selection;
    const common_flydelta_outcome_guidance_adviser staged_identity_adviser =
        [&staged_guidance_calls](const auto & request, auto & decision, std::string &) {
            ++staged_guidance_calls;
            const auto & pool = request.candidates;
            const size_t limit = request.selection_limit;
            decision.original_order.resize(pool.size());
            decision.shadow_order.resize(pool.size());
            decision.selected_order.resize(pool.size());
            decision.selection_reasons.assign(pool.size(), "shadow_preserved_existing_order");
            for (size_t i = 0; i < pool.size(); ++i) {
                decision.original_order[i] = i;
                decision.shadow_order[i] = i;
                decision.selected_order[i] = i;
            }
            return limit <= pool.size();
        };
    const bool staged_ok = common_flydelta_run_low_rank_coefficient_search_batched_staged(
        fixture(), basis, staged_config,
        staged_diagnostic_runner, staged_diagnostic_batch_runner,
        staged_full_runner, staged_full_batch_runner, 2,
        staged_trials, staged_selection, error, staged_identity_adviser);
    if (!staged_ok) {
        std::fprintf(stderr, "staged search failed: %s\\n", error.c_str());
    }
    CHECK(staged_ok);
    CHECK(staged_diagnostic_batch_calls > 0);
    CHECK(staged_guidance_calls == 1);
    CHECK(staged_full_batch_calls == 1 && staged_full_arm_count == 2);
    CHECK(staged_trials.size() > staged_full_arm_count);
    CHECK(staged_selection.selected);
    CHECK(staged_trials.back().mutation_kind == "full_generation_top_arm");
    CHECK(staged_trials.back().verifier_known);

    common_learning_in_memory_lifecycle_store lifecycle;
    common_flydelta_lifecycle_event_context lifecycle_context;
    lifecycle_context.event_id = "event:coefficient-search";
    lifecycle_context.idempotency_key = "idempotency:coefficient-search";
    lifecycle_context.source_id = "source:coefficient-search";
    lifecycle_context.scope.namespace_id = "local";
    lifecycle_context.scope.project_id = "agent-tests";
    lifecycle_context.scope.session_id = "session-1";
    lifecycle_context.content_hash = "sha256:coefficient-search";
    lifecycle_context.created_at = "2026-09-15T00:00:00Z";
    auto guidance_fixture = fixture();
    guidance_fixture.scope_fingerprint = "sha256:scope-guidance";
    guidance_fixture.baseline_deployment_fingerprint = "sha256:baseline-guidance";
    guidance_fixture.semantic_domain = "dataset.grouped_sum";
    guidance_fixture.guidance_fixture_group_id = "fixture-group:coefficient-search";
    guidance_fixture.outcome_guidance_verifier_revision = "verifier-contract:revision-2";
    guidance_fixture.model_n_layers = 32;
    guidance_fixture.ruleset_revision = "ruleset:revision-2";
    guidance_fixture.action_contract_fingerprint = "sha256:action-contract";
    CHECK(common_flydelta_append_coefficient_search_lifecycle(
        lifecycle, lifecycle_context, guidance_fixture, basis, tfo_config, tfo_trials,
        "flydelta://sideband/coefficient-search", error));
    const auto records = lifecycle.list(error);
    CHECK(error.empty() && records.size() == tfo_trials.size());
    CHECK(records.front().payload_json.find("coefficient-tfo") != std::string::npos);
    CHECK(records.front().payload_json.find("coefficients") != std::string::npos);
    CHECK(records.front().payload_json.find("diagnostics") != std::string::npos);
    CHECK(records.front().payload_json.find("leakage") != std::string::npos);
    CHECK(records.front().payload_json.find("outcome_guidance") != std::string::npos);
    CHECK(records.front().payload_json.find("ruleset:revision-2") != std::string::npos);
    CHECK(records.front().payload_json.find("sha256:action-contract") != std::string::npos);

    common_flydelta_coefficient_search_config coordinate_config = tfo_config;
    coordinate_config.strategy = common_flydelta_coefficient_search_strategy::coordinate;
    coordinate_config.max_candidates = tfo_trials.size();
    std::vector<std::vector<float>> coordinate_proposals;
    CHECK(common_flydelta_propose_low_rank_coefficients(
        coordinate_config, basis.vectors.size(), coordinate_proposals, error));
    CHECK(!coordinate_proposals.empty() &&
          coordinate_proposals.size() <= coordinate_config.max_candidates);
    // The strategies share the same bounded arm contract, but TFO-lite may
    // use mixed coefficients where the coordinate stencil cannot.
    for (const auto & trial : tfo_trials) {
        CHECK(std::sqrt(trial.coefficients[0] * trial.coefficients[0] +
            trial.coefficients[1] * trial.coefficients[1]) <=
            tfo_config.max_l2_norm + 0.0001f);
    }
    return 0;
}
