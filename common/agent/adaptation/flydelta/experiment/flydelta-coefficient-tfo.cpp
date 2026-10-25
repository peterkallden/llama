#include "agent/adaptation/flydelta/experiment/flydelta-coefficient-execution-internal.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <utility>

namespace common_flydelta_coefficient_detail {

bool run_tfo_lite_coefficient_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & baseline_runner,
        const common_flydelta_coefficient_search_batch_runner & batch_runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error,
        const common_flydelta_outcome_guidance_adviser & adviser) {
    trials.clear();
    selection = {};

    std::vector<float> zero(basis.vectors.size(), 0.0f);
    common_flydelta_counterfactual_trial baseline;
    common_flydelta_decision_margin baseline_margin;
    common_flydelta_representation_diagnostics baseline_geometry;
    bool baseline_geometry_available = false;
    if (!baseline_runner(fixture, basis, zero, false, baseline, baseline_margin,
                baseline_geometry, baseline_geometry_available, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error) ||
            (baseline_geometry_available &&
             !common_flydelta_representation_diagnostics_validate(baseline_geometry, error))) {
        return false;
    }

    std::mt19937_64 generator(config.seed);
    std::uniform_real_distribution<float> distribution(-config.step, config.step);
    std::vector<std::vector<float>> population;
    std::vector<size_t> population_parents;
    std::vector<std::string> population_mutations;
    const size_t candidate_limit = std::min(config.max_candidates,
        config.population_size * config.iterations);

    for (size_t index = 0; index < config.population_size && population.size() < candidate_limit;
            ++index) {
        std::vector<float> coefficients(basis.vectors.size(), 0.0f);
        const size_t coordinate_count = config.population_size > 1
            ? std::min(config.population_size - 1, basis.vectors.size() * 2)
            : 0;
        if (index < coordinate_count) {
            coefficients[index / 2] = index % 2 == 0 ? config.step : -config.step;
        } else {
            for (float & coefficient : coefficients) coefficient = distribution(generator);
        }
        coefficients = bound_coefficients(std::move(coefficients), config.max_l2_norm);
        if (!contains_coefficients(population, coefficients)) {
            population.push_back(std::move(coefficients));
            population_parents.push_back(static_cast<size_t>(-1));
            population_mutations.push_back(index < coordinate_count
                ? "initial_coordinate" : "initial_mixed");
        }
    }

    size_t evaluated = 0;
    common_flydelta_dose_state dose_state;
    std::vector<std::vector<float>> evaluated_coefficients;
    for (size_t iteration = 0; iteration < config.iterations && evaluated < candidate_limit;
            ++iteration) {
        std::vector<size_t> pending_indices;
        std::vector<std::vector<float>> pending_coefficients;
        for (size_t population_index = 0; population_index < population.size(); ++population_index) {
            const auto & coefficients = population[population_index];
            if (evaluated >= candidate_limit || contains_coefficients(
                    evaluated_coefficients, coefficients)) {
                continue;
            }
            pending_indices.push_back(population_index);
            pending_coefficients.push_back(coefficients);
        }
        const size_t remaining_budget = candidate_limit - evaluated;
        if (pending_indices.size() > remaining_budget) {
            pending_indices.resize(remaining_budget);
            pending_coefficients.resize(remaining_budget);
        }
        if (adviser && !pending_coefficients.empty()) {
            std::vector<common_flydelta_outcome_guidance_candidate> guidance_candidates;
            guidance_candidates.reserve(pending_coefficients.size());
            for (const auto & coefficients : pending_coefficients) {
                guidance_candidates.push_back(guidance_candidate(coefficients, basis, config));
            }
            common_flydelta_outcome_guidance_decision guidance;
            std::string guidance_error;
            common_flydelta_outcome_guidance_request request;
            request.stage = "tfo_population";
            request.round = iteration;
            request.decision_id = "tfo-population-" + std::to_string(iteration);
            request.candidates = guidance_candidates;
            request.selection_limit = pending_coefficients.size();
            if (adviser(request, guidance, guidance_error) &&
                    guidance.selected_order.size() == pending_coefficients.size()) {
                std::vector<size_t> ordered_indices;
                std::vector<std::vector<float>> ordered_coefficients;
                ordered_indices.reserve(pending_indices.size());
                ordered_coefficients.reserve(pending_coefficients.size());
                std::vector<bool> used(pending_indices.size(), false);
                for (const size_t order_index : guidance.selected_order) {
                    if (order_index >= pending_indices.size() || used[order_index]) {
                        ordered_indices.clear();
                        break;
                    }
                    used[order_index] = true;
                    ordered_indices.push_back(pending_indices[order_index]);
                    ordered_coefficients.push_back(pending_coefficients[order_index]);
                }
                if (ordered_indices.size() == pending_indices.size()) {
                    pending_indices = std::move(ordered_indices);
                    pending_coefficients = std::move(ordered_coefficients);
                }
            }
        }
        std::vector<coefficient_arm_result> pending_results;
        if (!run_coefficient_arms_batched(fixture, basis, config, batch_runner,
                dose_state, pending_coefficients, pending_results, error)) return false;
        // The adviser changes execution order only. Feed results back in the
        // population's original order so tie-breaks and next-generation
        // parent selection do not depend on physical scheduling.
        std::vector<size_t> result_order(pending_results.size());
        std::iota(result_order.begin(), result_order.end(), 0);
        std::stable_sort(result_order.begin(), result_order.end(), [&](size_t left, size_t right) {
            return pending_indices[left] < pending_indices[right];
        });
        for (const size_t pending : result_order) {
            const size_t population_index = pending_indices[pending];
            const auto & arm = pending_results[pending];
            common_flydelta_coefficient_trial trial;
            trial.coefficients = arm.executed_coefficients;
            trial.requested_coefficients = arm.requested_coefficients;
            trial.requested_strength = norm(arm.requested_coefficients);
            trial.executed_strength = norm(arm.executed_coefficients);
            trial.dose_action = arm.dose_decision.action;
            trial.relative_dose = arm.dose_decision.relative_dose;
            trial.dose_evaluated = arm.dose_evaluated;
            trial.dose_safety_limited = arm.dose_safety_limited;
            trial.dose_reason = arm.dose_reason;
            trial.margin = arm.margin;
            trial.margin_comparison.available = baseline_margin.available && arm.margin.available;
            trial.margin_comparison.baseline = baseline_margin;
            trial.margin_comparison.candidate = arm.margin;
            common_flydelta_coefficient_trial_set_observation(trial, baseline, arm.counterfactual);
            if (!update_dose_metrics(basis, config, trial, error)) return false;
            if (!copy_geometry(arm.geometry, arm.geometry_available, trial, error)) return false;
            trial.iteration = iteration;
            trial.parent_trial_index = population_parents[population_index];
            trial.mutation_kind = population_mutations[population_index];
            const auto forecast = guidance_candidate(
                population[population_index], basis, config);
            trial.guidance_arm_id = forecast.arm_id;
            trial.guidance_basis_id = forecast.basis_identity;
            trial.guidance_margin_sign = forecast.diagnostic_margin_sign;
            trial.search_fitness = diagnostic_fitness(
                trial, baseline_margin, arm.margin, config);
            evaluated_coefficients.push_back(pending_coefficients[pending]);
            trials.push_back(std::move(trial));
            ++evaluated;
        }
        if (evaluated >= candidate_limit || iteration + 1 >= config.iterations) break;

        const common_flydelta_coefficient_trial * best = nullptr;
        for (size_t index = 0; index < trials.size(); ++index) {
            const auto & trial = trials[index];
            if (trial.iteration != iteration ||
                    trial.outcome == common_flydelta_counterfactual_outcome::harmed) continue;
            if (best == nullptr || better_diagnostic_trial(trial, *best)) {
                best = &trial;
            }
        }
        const std::vector<float> anchor = best == nullptr ? zero : best->coefficients;
        const size_t anchor_index = best == nullptr ? static_cast<size_t>(-1) :
            static_cast<size_t>(best - trials.data());
        const float radius = config.step * config.exploration_scale /
            static_cast<float>(iteration + 2);
        std::uniform_real_distribution<float> local_distribution(-radius, radius);
        population.clear();
        population_parents.clear();
        population_mutations.clear();
        population.push_back(anchor);
        population_parents.push_back(anchor_index);
        population_mutations.push_back("forage_anchor");
        for (size_t index = 1; index < config.population_size; ++index) {
            std::vector<float> coefficients = anchor;
            for (float & coefficient : coefficients) coefficient += local_distribution(generator);
            coefficients = bound_coefficients(std::move(coefficients), config.max_l2_norm);
            if (!contains_coefficients(population, coefficients)) {
                population.push_back(std::move(coefficients));
                population_parents.push_back(anchor_index);
                population_mutations.push_back("forage_perturbation");
            }
        }
    }

    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (!valid_outcome(trial.outcome) ||
                trial.outcome != common_flydelta_counterfactual_outcome::helped ||
                !trial.executed || !trial.verifier_known) continue;
        if (!selection.selected || trial.quality_delta > selection.score ||
                (trial.quality_delta == selection.score &&
                 norm(trial.coefficients) < norm(trials[selection.trial_index].coefficients))) {
            selection.selected = true;
            selection.trial_index = index;
            selection.score = trial.quality_delta;
        }
    }
    return true;
}

} // namespace common_flydelta_coefficient_detail
