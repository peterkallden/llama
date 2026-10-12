#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <algorithm>
#include <utility>

bool common_flydelta_run_low_rank_coefficient_search_batched_staged(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & diagnostic_runner,
        const common_flydelta_coefficient_search_batch_runner & diagnostic_batch_runner,
        const common_flydelta_coefficient_search_runner & full_generation_runner,
        const common_flydelta_coefficient_search_batch_runner & full_generation_batch_runner,
        const size_t full_generation_top_k,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error) {
    error.clear();
    if (full_generation_top_k == 0 || full_generation_top_k > 16 ||
            !diagnostic_runner || !diagnostic_batch_runner ||
            !full_generation_runner || !full_generation_batch_runner) {
        error = "FlyDelta staged coefficient search input is invalid";
        return false;
    }

    std::vector<common_flydelta_coefficient_trial> diagnostic_trials;
    common_flydelta_coefficient_selection diagnostic_selection;
    if (!common_flydelta_run_low_rank_coefficient_search_batched(
            fixture, basis, config, diagnostic_runner, diagnostic_batch_runner,
            diagnostic_trials, diagnostic_selection, error)) return false;
    trials = diagnostic_trials;
    selection = {};

    std::vector<size_t> ranking;
    ranking.reserve(diagnostic_trials.size());
    for (size_t index = 0; index < diagnostic_trials.size(); ++index) {
        if (diagnostic_trials[index].executed) ranking.push_back(index);
    }
    std::stable_sort(ranking.begin(), ranking.end(), [&](const size_t left, const size_t right) {
        if (diagnostic_trials[left].search_fitness != diagnostic_trials[right].search_fitness) {
            return diagnostic_trials[left].search_fitness > diagnostic_trials[right].search_fitness;
        }
        return diagnostic_trials[left].requested_strength < diagnostic_trials[right].requested_strength;
    });
    const size_t top_k = std::min(full_generation_top_k, ranking.size());
    if (top_k == 0) return true;

    common_flydelta_counterfactual_trial baseline;
    common_flydelta_decision_margin baseline_margin;
    common_flydelta_representation_diagnostics baseline_geometry;
    bool baseline_geometry_available = false;
    if (!full_generation_runner(
            fixture, basis, std::vector<float>(basis.vectors.size(), 0.0f), false,
            baseline, baseline_margin, baseline_geometry,
            baseline_geometry_available, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error)) return false;

    std::vector<std::vector<float>> top_coefficients;
    top_coefficients.reserve(top_k);
    for (size_t rank = 0; rank < top_k; ++rank) {
        top_coefficients.push_back(diagnostic_trials[ranking[rank]].coefficients);
    }
    std::vector<common_flydelta_counterfactual_trial> full_trials;
    std::vector<common_flydelta_decision_margin> full_margins;
    std::vector<common_flydelta_representation_diagnostics> full_geometries;
    std::vector<bool> full_geometry_available;
    if (!full_generation_batch_runner(
            fixture, basis, top_coefficients, full_trials, full_margins,
            full_geometries, full_geometry_available, error) ||
            full_trials.size() != top_k || full_margins.size() != top_k ||
            full_geometries.size() != top_k || full_geometry_available.size() != top_k) {
        if (error.empty()) error = "FlyDelta staged full-generation batch result is invalid";
        return false;
    }

    for (size_t rank = 0; rank < top_k; ++rank) {
        const size_t diagnostic_index = ranking[rank];
        const auto & diagnostic = diagnostic_trials[diagnostic_index];
        const auto & candidate = full_trials[rank];
        if (!common_flydelta_counterfactual_trial_validate(candidate, error) ||
                !common_flydelta_decision_margin_validate(full_margins[rank], error)) return false;
        if (full_geometry_available[rank] &&
                !common_flydelta_representation_diagnostics_validate(full_geometries[rank], error)) {
            return false;
        }
        common_flydelta_coefficient_trial full_trial;
        full_trial.coefficients = diagnostic.coefficients;
        full_trial.requested_coefficients = diagnostic.requested_coefficients;
        full_trial.requested_strength = diagnostic.requested_strength;
        full_trial.executed_strength = diagnostic.executed_strength;
        full_trial.margin = full_margins[rank];
        full_trial.margin_comparison.available = baseline_margin.available && full_margins[rank].available;
        full_trial.margin_comparison.baseline = baseline_margin;
        full_trial.margin_comparison.candidate = full_margins[rank];
        full_trial.outcome = common_flydelta_classify_counterfactual(baseline, candidate);
        full_trial.quality_delta = candidate.quality - baseline.quality;
        full_trial.executed = candidate.executed;
        full_trial.verifier_known = baseline.verifier_known && candidate.verifier_known;
        full_trial.geometry_available = full_geometry_available[rank];
        full_trial.geometry = full_geometries[rank];
        full_trial.iteration = diagnostic.iteration;
        full_trial.parent_trial_index = diagnostic_index;
        full_trial.mutation_kind = "full_generation_top_arm";
        full_trial.search_fitness = diagnostic.search_fitness;
        const size_t result_index = trials.size();
        trials.push_back(std::move(full_trial));
        const auto & selected = trials.back();
        if (selected.outcome == common_flydelta_counterfactual_outcome::helped &&
                selected.executed && selected.verifier_known &&
                (!selection.selected || selected.quality_delta > selection.score)) {
            selection.selected = true;
            selection.trial_index = result_index;
            selection.score = selected.quality_delta;
        }
    }
    return true;
}
