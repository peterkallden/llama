#include "agent/adaptation/flydelta/flydelta-deep-search.h"
#include "agent/adaptation/flydelta/experiment/flydelta-coefficient-execution-internal.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <random>
#include <set>

namespace {

bool finite(float value) {
    return std::isfinite(value);
}

float direction_score(const common_flydelta_deep_search_direction & value) {
    return value.decision_score_available
        ? value.decision_score : value.direction.median_alignment;
}

common_flydelta_coefficient_search_batch_runner make_scalar_batch_runner(
        const common_flydelta_coefficient_search_runner & runner) {
    return [runner](
            const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & basis,
            const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & trials,
            std::vector<common_flydelta_decision_margin> & margins,
            std::vector<common_flydelta_representation_diagnostics> & geometries,
            std::vector<bool> & geometry_available, std::string & batch_error) {
        trials.clear();
        margins.clear();
        geometries.clear();
        geometry_available.clear();
        for (const auto & values : coefficients) {
            common_flydelta_counterfactual_trial trial;
            common_flydelta_decision_margin margin;
            common_flydelta_representation_diagnostics geometry;
            bool has_geometry = false;
            if (!runner(current_fixture, basis, values, true, trial, margin,
                    geometry, has_geometry, batch_error)) return false;
            trials.push_back(std::move(trial));
            margins.push_back(std::move(margin));
            geometries.push_back(std::move(geometry));
            geometry_available.push_back(has_geometry);
        }
        return true;
    };
}

} // namespace

bool common_flydelta_deep_search_config_validate(
        const common_flydelta_deep_search_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.max_rank == 0 || config.max_rank > 4 ||
            config.max_directions == 0 || config.max_directions > 16 ||
            config.full_generation_top_k == 0 || config.full_generation_top_k > 16 ||
            config.matched_max_arms == 0 || config.matched_max_arms > 48 ||
            config.max_rank > config.max_directions ||
            !common_flydelta_coefficient_search_config_validate(
                config.coefficients, config.max_rank, error)) {
        if (error.empty()) error = "FlyDelta deep search configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_select_deep_search_directions(
        const common_flydelta_deep_search_config & config,
        const std::vector<common_flydelta_deep_search_direction> & directions,
        std::vector<common_flydelta_deep_search_direction> & selected,
        std::string & error) {
    error.clear();
    selected.clear();
    if (!common_flydelta_deep_search_config_validate(config, error) ||
            directions.empty() || directions.size() > config.max_directions) {
        if (error.empty()) error = "FlyDelta deep search direction input is invalid";
        return false;
    }
    const int32_t layer = directions.front().direction.layer_index;
    const size_t dimension = directions.front().direction.values.size();
    for (const auto & value : directions) {
        if (!common_flydelta_direction_candidate_validate(
                value.direction, dimension, error) ||
                value.direction.layer_index != layer ||
                (value.decision_score_available && !finite(value.decision_score))) {
            if (error.empty()) error = "FlyDelta deep search directions are incompatible";
            return false;
        }
    }
    selected = directions;
    std::stable_sort(selected.begin(), selected.end(),
        [](const auto & left, const auto & right) {
            const float left_score = direction_score(left);
            const float right_score = direction_score(right);
            if (left_score != right_score) return left_score > right_score;
            return left.direction.kind < right.direction.kind;
        });
    if (selected.size() > config.max_rank) selected.resize(config.max_rank);
    return true;
}

bool common_flydelta_run_deep_search_batched(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_deep_search_config & config,
        const std::vector<common_flydelta_deep_search_direction> & directions,
        const common_flydelta_coefficient_search_runner & diagnostic_runner,
        const common_flydelta_coefficient_search_batch_runner & diagnostic_batch_runner,
        const common_flydelta_coefficient_search_runner & full_generation_runner,
        const common_flydelta_coefficient_search_batch_runner & full_generation_batch_runner,
        common_flydelta_deep_search_result & result,
        std::string & error,
        const common_flydelta_outcome_guidance_adviser & adviser) {
    error.clear();
    result = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_deep_search_config_validate(config, error) ||
            !diagnostic_runner || !diagnostic_batch_runner ||
            !full_generation_runner || !full_generation_batch_runner) {
        if (error.empty()) error = "FlyDelta deep search input is invalid";
        return false;
    }
    if (!common_flydelta_select_deep_search_directions(
            config, directions, result.selected_directions, error)) return false;
    std::vector<common_flydelta_direction_candidate> candidates;
    candidates.reserve(result.selected_directions.size());
    for (const auto & value : result.selected_directions) candidates.push_back(value.direction);
    if (!common_flydelta_build_low_rank_basis(
            candidates.front().values.size(), config.max_rank, candidates,
            result.basis, error)) return false;
    result.layer_index = result.basis.layer_index;
    // Deep chooses/rebuilds its own basis. Do not reuse the caller's earlier
    // basis fingerprint for the forecast, full execution or history.
    auto coefficient_config = config.coefficients;
    coefficient_config.guidance_basis_identity =
        common_flydelta_coefficient_detail::guidance_basis_identity(result.basis);
    coefficient_config.guidance_layer_index = result.basis.layer_index;
    std::vector<common_flydelta_coefficient_trial> diagnostic_trials;
    common_flydelta_coefficient_selection diagnostic_selection;
    if (!common_flydelta_run_low_rank_coefficient_search_batched(
            fixture, result.basis, coefficient_config, diagnostic_runner,
            diagnostic_batch_runner,
            diagnostic_trials, diagnostic_selection, error, adviser)) return false;
    result.coefficient_trials = diagnostic_trials;

    common_flydelta_counterfactual_trial baseline;
    common_flydelta_decision_margin baseline_margin;
    common_flydelta_representation_diagnostics baseline_geometry;
    bool baseline_geometry_available = false;
    if (!full_generation_runner(
            fixture, result.basis, std::vector<float>(result.basis.vectors.size(), 0.0f),
            false, baseline, baseline_margin, baseline_geometry,
            baseline_geometry_available, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error)) return false;

    std::vector<size_t> ranking;
    for (size_t index = 0; index < diagnostic_trials.size(); ++index) {
        const auto & trial = diagnostic_trials[index];
        if (coefficient_config.experimental_dose_mode != common_flydelta_coefficient_dose_mode::raw_coefficients &&
                coefficient_config.use_dose_controller && trial.dose_evaluated &&
                trial.dose_action != common_flydelta_dose_action::accept) continue;
        if (trial.executed && trial.overlay_applied &&
                common_flydelta_coefficient_detail::norm(trial.coefficients) > 0.0f) ranking.push_back(index);
    }
    std::stable_sort(ranking.begin(), ranking.end(), [&](size_t left, size_t right) {
        if (diagnostic_trials[left].search_fitness != diagnostic_trials[right].search_fitness) {
            return diagnostic_trials[left].search_fitness > diagnostic_trials[right].search_fitness;
        }
        return diagnostic_trials[left].coefficients.size() <
            diagnostic_trials[right].coefficients.size();
    });
    std::set<std::vector<float>> unique_coefficients;
    ranking.erase(std::remove_if(ranking.begin(), ranking.end(), [&](size_t index) {
        return !unique_coefficients.insert(diagnostic_trials[index].coefficients).second;
    }), ranking.end());
    const size_t top_k = std::min(config.full_generation_top_k, ranking.size());
    const auto baseline_ranking = ranking;
    std::vector<common_flydelta_outcome_guidance_candidate> guidance_candidates;
    common_flydelta_outcome_guidance_decision guidance;
    bool valid_guidance = false;
    if (adviser && top_k > 0) {
        guidance_candidates.reserve(ranking.size());
        for (const size_t index : ranking) {
            const auto & trial = diagnostic_trials[index];
            guidance_candidates.push_back(common_flydelta_coefficient_detail::guidance_candidate(
                trial.coefficients, result.basis, coefficient_config,
                &trial.margin_comparison));
        }
        std::string guidance_error;
        common_flydelta_outcome_guidance_request request;
        request.stage = "deep_full_generation_frontier";
        request.candidates = guidance_candidates;
        request.selection_limit = top_k;
        request.decision_id = "deep-full-generation-frontier";
        if (adviser(request, guidance, guidance_error) &&
                guidance.selected_order.size() == ranking.size()) {
            std::vector<size_t> reordered;
            std::vector<bool> used(ranking.size(), false);
            for (const size_t local_index : guidance.selected_order) {
                if (local_index >= ranking.size() || used[local_index]) { reordered.clear(); break; }
                used[local_index] = true;
                reordered.push_back(ranking[local_index]);
            }
            const bool actual_order_valid = reordered.size() == ranking.size();
            if (actual_order_valid) ranking = std::move(reordered);
            std::set<size_t> hypothetical(guidance.mixed_policy_order.begin(),
                guidance.mixed_policy_order.end());
            valid_guidance = actual_order_valid && guidance.fallback_reason.empty() &&
                guidance.mixed_policy_order.size() == baseline_ranking.size() &&
                hypothetical.size() == baseline_ranking.size() &&
                !hypothetical.empty() && *hypothetical.rbegin() < baseline_ranking.size();
        }
    }
    std::vector<size_t> execution_indices(ranking.begin(), ranking.begin() + top_k);
    const std::set<size_t> selected_indices(execution_indices.begin(), execution_indices.end());
    if (config.matched_evaluation) {
        result.matched_evaluation_reason = "guidance_or_fixture_unavailable";
        if (valid_guidance && !fixture.guidance_fixture_group_id.empty()) {
            auto & evaluation = result.matched_evaluation;
            evaluation.fixture_group_id = fixture.guidance_fixture_group_id;
            evaluation.history_fingerprint = guidance.history_fingerprint;
            evaluation.candidate_pool_fingerprint =
                common_flydelta_outcome_guidance_pool_fingerprint(guidance_candidates);
            evaluation.evaluation_id = evaluation.candidate_pool_fingerprint;
            evaluation.baseline_cost_available = baseline.inference_cost_available;
            evaluation.baseline_inference_ms = baseline.attributed_inference_ms;
            evaluation.baseline_evidence_ref = baseline.evidence_ref;
            std::vector<size_t> random_order(baseline_ranking.size());
            std::iota(random_order.begin(), random_order.end(), 0);
            uint64_t random_seed = config.matched_seed;
            for (const unsigned char ch : fixture.task_fingerprint + evaluation.candidate_pool_fingerprint) {
                random_seed = (random_seed ^ ch) * 1099511628211ULL;
            }
            std::mt19937_64 random(random_seed);
            std::shuffle(random_order.begin(), random_order.end(), random);
            for (size_t i = 0; i < top_k; ++i) {
                evaluation.baseline_selection.push_back(guidance_candidates[i].arm_id);
                evaluation.guided_selection.push_back(
                    guidance_candidates[guidance.mixed_policy_order[i]].arm_id);
                evaluation.random_ablation_selection.push_back(guidance_candidates[random_order[i]].arm_id);
            }
            std::vector<std::string> union_ids;
            std::string union_error;
            if (common_flydelta_guidance_union_selection(evaluation.baseline_selection,
                    evaluation.guided_selection, evaluation.random_ablation_selection, union_ids, union_error) &&
                    union_ids.size() <= config.matched_max_arms) {
                execution_indices.clear();
                for (const auto & id : union_ids) {
                    const auto it = std::find_if(guidance_candidates.begin(), guidance_candidates.end(),
                        [&](const auto & candidate) { return candidate.arm_id == id; });
                    execution_indices.push_back(baseline_ranking[std::distance(guidance_candidates.begin(), it)]);
                }
                result.matched_evaluation_reason = "pending_full_verification";
            } else {
                result.matched_evaluation_reason = "matched_union_arm_budget_exceeded";
            }
        }
    }
    if (execution_indices.empty()) return true;
    std::vector<std::vector<float>> top_coefficients;
    top_coefficients.reserve(execution_indices.size());
    for (const size_t index : execution_indices) {
        top_coefficients.push_back(diagnostic_trials[index].coefficients);
    }
    std::vector<common_flydelta_counterfactual_trial> full_trials;
    std::vector<common_flydelta_decision_margin> full_margins;
    std::vector<common_flydelta_representation_diagnostics> full_geometries;
    std::vector<bool> full_geometry_available;
    if (!full_generation_batch_runner(
            fixture, result.basis, top_coefficients, full_trials, full_margins,
            full_geometries, full_geometry_available, error) ||
            full_trials.size() != execution_indices.size() || full_margins.size() != execution_indices.size() ||
            full_geometries.size() != execution_indices.size() || full_geometry_available.size() != execution_indices.size()) {
        if (error.empty()) error = "FlyDelta deep full-generation batch result is invalid";
        return false;
    }
    bool comparable_union = baseline.executed && baseline.host_evaluated && !baseline.overlay_applied &&
        !baseline.evidence_ref.empty();
    std::vector<common_flydelta_coefficient_trial> executed_trials;
    for (size_t rank = 0; rank < execution_indices.size(); ++rank) {
        const size_t diagnostic_index = execution_indices[rank];
        const auto & diagnostic = diagnostic_trials[diagnostic_index];
        const auto & candidate = full_trials[rank];
        const auto & margin = full_margins[rank];
        const auto & geometry = full_geometries[rank];
        const bool geometry_available = full_geometry_available[rank];
        if (!common_flydelta_counterfactual_trial_validate(candidate, error) ||
                !common_flydelta_decision_margin_validate(margin, error)) return false;
        common_flydelta_coefficient_trial trial = diagnostic;
        trial.margin = margin;
        trial.margin_comparison.available = baseline_margin.available && margin.available;
        trial.margin_comparison.baseline = baseline_margin;
        trial.margin_comparison.candidate = margin;
        common_flydelta_coefficient_trial_set_observation(trial, baseline, candidate);
        if (!common_flydelta_coefficient_detail::update_dose_metrics(result.basis, coefficient_config, trial, error)) return false;
        trial.geometry_available = geometry_available;
        if (geometry_available &&
                !common_flydelta_representation_diagnostics_validate(geometry, error)) {
            return false;
        }
        trial.geometry = geometry;
        trial.iteration = diagnostic.iteration;
        trial.parent_trial_index = diagnostic_index;
        trial.mutation_kind = selected_indices.count(diagnostic_index)
            ? "full_generation_top_arm" : "qualification_only_full_generation";
        const auto forecast = common_flydelta_coefficient_detail::guidance_candidate(
            diagnostic.coefficients, result.basis, coefficient_config, &diagnostic.margin_comparison);
        trial.guidance_arm_id = forecast.arm_id;
        trial.guidance_basis_id = forecast.basis_identity;
        trial.guidance_margin_sign = forecast.diagnostic_margin_sign;
        trial.search_fitness = diagnostic.search_fitness;
        executed_trials.push_back(std::move(trial));
        const auto & full_trial = executed_trials.back();
        if (result.matched_evaluation_reason == "pending_full_verification") {
            comparable_union = comparable_union && candidate.executed && candidate.host_evaluated &&
                candidate.overlay_applied && !candidate.evidence_ref.empty();
            result.matched_evaluation.union_arm_results.push_back({forecast.arm_id,
                full_trial.outcome, candidate.attributed_inference_ms, candidate.inference_cost_available,
                candidate.evidence_ref});
        }
    }
    // Return exactly the selected K arms, in policy order. Qualification-only
    // arms remain evaluation evidence and cannot enter downstream learning,
    // candidate selection or change equal-score tie-breaks in this job.
    for (size_t rank = 0; rank < top_k; ++rank) {
        const auto it = std::find(execution_indices.begin(), execution_indices.end(), ranking[rank]);
        auto & trial = executed_trials[std::distance(execution_indices.begin(), it)];
        const size_t result_index = result.coefficient_trials.size();
        result.coefficient_trials.push_back(std::move(trial));
        const auto & full_trial = result.coefficient_trials.back();
        const auto & progress = full_trial.semantic_progress;
        const float progress_delta = static_cast<float>(progress.candidate_score - progress.baseline_score);
        if (full_trial.executed && full_trial.host_evaluated && full_trial.overlay_applied &&
                !full_trial.evidence_ref.empty() &&
                common_flydelta_semantic_progress_is_safe_anchor(progress) &&
                progress.outcome == common_flydelta_semantic_progress_outcome::improved &&
                !progress.improved_dimensions.empty() &&
                (!result.provisional_selection.selected ||
                 progress_delta > result.provisional_selection.score)) {
            result.provisional_selection.selected = true;
            result.provisional_selection.trial_index = result_index;
            result.provisional_selection.score = progress_delta;
        }
        if (full_trial.outcome == common_flydelta_counterfactual_outcome::helped &&
                full_trial.executed && full_trial.verifier_known &&
                (!result.coefficient_selection.selected ||
                 full_trial.quality_delta > result.coefficient_selection.score)) {
            result.coefficient_selection.selected = true;
            result.coefficient_selection.trial_index = result_index;
            result.coefficient_selection.score = full_trial.quality_delta;
        }
    }
    if (result.matched_evaluation_reason == "pending_full_verification") {
        result.matched_evaluation_complete = comparable_union;
        result.matched_evaluation_reason = comparable_union
            ? "matched_union_verified" : "matched_union_incomplete_verification_or_overlay";
    }
    return true;
}

bool common_flydelta_run_deep_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_deep_search_config & config,
        const std::vector<common_flydelta_deep_search_direction> & directions,
        const common_flydelta_coefficient_search_runner & diagnostic_runner,
        const common_flydelta_coefficient_search_runner & full_generation_runner,
        common_flydelta_deep_search_result & result,
        std::string & error) {
    if (!diagnostic_runner || !full_generation_runner) {
        error = "FlyDelta deep search runner is invalid";
        return false;
    }
    return common_flydelta_run_deep_search_batched(
        fixture, config, directions, diagnostic_runner,
        make_scalar_batch_runner(diagnostic_runner), full_generation_runner,
        make_scalar_batch_runner(full_generation_runner), result, error);
}

bool common_flydelta_append_deep_search_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_deep_search_config & config,
        const common_flydelta_deep_search_result & result,
        const std::string & experimental_artifact_id,
        std::string & error) {
    error.clear();
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_deep_search_config_validate(config, error) ||
            result.schema_version != 1 || result.layer_index < 0 ||
            result.selected_directions.empty() ||
            result.selected_directions.size() > config.max_rank ||
            experimental_artifact_id.empty() || experimental_artifact_id.size() > 512) {
        if (error.empty()) error = "FlyDelta deep search lifecycle input is invalid";
        return false;
    }
    if (!common_flydelta_low_rank_basis_validate(result.basis, config.max_rank, error) ||
            result.basis.layer_index != result.layer_index ||
            result.basis.vectors.size() != result.selected_directions.size()) {
        if (error.empty()) error = "FlyDelta deep search lifecycle basis is invalid";
        return false;
    }
    return common_flydelta_append_coefficient_search_lifecycle(
        store, context, fixture, result.basis, config.coefficients,
        result.coefficient_trials, experimental_artifact_id, error);
}
