#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <random>

namespace {

bool finite(float value) {
    return std::isfinite(value);
}

float norm(const std::vector<float> & values) {
    float sum = 0.0f;
    for (const float value : values) sum += value * value;
    return std::sqrt(sum);
}

bool valid_outcome(common_flydelta_counterfactual_outcome outcome) {
    switch (outcome) {
        case common_flydelta_counterfactual_outcome::unknown:
        case common_flydelta_counterfactual_outcome::helped:
        case common_flydelta_counterfactual_outcome::neutral:
        case common_flydelta_counterfactual_outcome::harmed:
            return true;
    }
    return false;
}

bool same_coefficients(const std::vector<float> & left, const std::vector<float> & right) {
    return left == right;
}

std::vector<float> bound_coefficients(
        std::vector<float> coefficients, float max_l2_norm) {
    const float value = norm(coefficients);
    if (value > max_l2_norm && value > std::numeric_limits<float>::epsilon()) {
        const float scale = max_l2_norm / value;
        for (float & coefficient : coefficients) coefficient *= scale;
    }
    return coefficients;
}

std::vector<float> rescale_coefficients(
        const std::vector<float> & coefficients, float requested_strength,
        float target_strength) {
    if (requested_strength <= std::numeric_limits<float>::epsilon()) {
        return coefficients;
    }
    std::vector<float> result = coefficients;
    const float factor = target_strength / requested_strength;
    for (float & value : result) value *= factor;
    return result;
}

float diagnostic_fitness(
        const common_flydelta_coefficient_trial & trial,
        const common_flydelta_decision_margin & baseline_margin,
        const common_flydelta_decision_margin & margin,
        const common_flydelta_coefficient_search_config & config) {
    float score = trial.quality_delta;
    if (baseline_margin.available && margin.available) {
        score = margin.normalized_delta() - baseline_margin.normalized_delta();
    }
    score -= config.norm_penalty * norm(trial.coefficients);
    if (trial.geometry_available) {
        score -= config.leakage_penalty * trial.geometry.leakage;
    }
    if (trial.outcome == common_flydelta_counterfactual_outcome::harmed) score -= 1.0f;
    return score;
}

bool copy_geometry(
        const common_flydelta_representation_diagnostics & geometry,
        bool geometry_available,
        common_flydelta_coefficient_trial & trial,
        std::string & error) {
    trial.geometry_available = geometry_available;
    if (!geometry_available) return true;
    if (!common_flydelta_representation_diagnostics_validate(geometry, error)) {
        return false;
    }
    trial.geometry = geometry;
    return true;
}

struct coefficient_arm_result {
    std::vector<float> requested_coefficients;
    std::vector<float> executed_coefficients;
    common_flydelta_counterfactual_trial counterfactual;
    common_flydelta_decision_margin margin;
    common_flydelta_representation_diagnostics geometry;
    bool geometry_available = false;
    common_flydelta_dose_decision dose_decision;
    bool dose_evaluated = false;
    bool dose_safety_limited = false;
    std::string dose_reason;
};

bool run_coefficient_arm(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & runner,
        common_flydelta_dose_state & dose_state,
        const std::vector<float> & coefficients,
        coefficient_arm_result & result,
        std::string & error) {
    result = {};
    result.requested_coefficients = coefficients;
    result.executed_coefficients = coefficients;
    const float requested_strength = norm(coefficients);
    if (!runner(fixture, basis, coefficients, true, result.counterfactual,
            result.margin, result.geometry, result.geometry_available, error) ||
            !common_flydelta_counterfactual_trial_validate(result.counterfactual, error) ||
            !common_flydelta_decision_margin_validate(result.margin, error)) return false;
    if (result.geometry_available &&
            !common_flydelta_representation_diagnostics_validate(result.geometry, error)) {
        return false;
    }
    if (!config.use_dose_controller || requested_strength <= std::numeric_limits<float>::epsilon()) {
        result.dose_decision.action = common_flydelta_dose_action::accept;
        result.dose_reason = config.use_dose_controller
            ? "zero coefficient dose" : "dose controller disabled";
        return true;
    }
    common_flydelta_dose_observation observation{
        result.geometry_available, requested_strength,
        result.geometry_available ? result.geometry.shift_norm : 0.0f,
        result.geometry_available ? result.geometry.progress : 0.0f,
        result.geometry_available ? result.geometry.leakage : 0.0f,
    };
    if (!common_flydelta_dose_observe(
            config.dose_policy, dose_state, observation,
            result.dose_decision, error)) return false;
    result.dose_evaluated = true;
    result.dose_safety_limited = result.dose_decision.safety_limited;
    result.dose_reason = result.dose_decision.reason;
    if (result.dose_decision.action == common_flydelta_dose_action::retry_lower &&
            result.dose_decision.proposed_safe_strength && config.max_dose_retries > 0) {
        result.executed_coefficients = rescale_coefficients(
            coefficients, requested_strength, *result.dose_decision.proposed_safe_strength);
        if (!runner(fixture, basis, result.executed_coefficients, true,
                result.counterfactual, result.margin, result.geometry,
                result.geometry_available, error) ||
                !common_flydelta_counterfactual_trial_validate(result.counterfactual, error) ||
                !common_flydelta_decision_margin_validate(result.margin, error)) return false;
        if (result.geometry_available &&
                !common_flydelta_representation_diagnostics_validate(result.geometry, error)) {
            return false;
        }
        observation = {
            result.geometry_available, *result.dose_decision.proposed_safe_strength,
            result.geometry_available ? result.geometry.shift_norm : 0.0f,
            result.geometry_available ? result.geometry.progress : 0.0f,
            result.geometry_available ? result.geometry.leakage : 0.0f,
        };
        if (!common_flydelta_dose_observe(
                config.dose_policy, dose_state, observation,
                result.dose_decision, error)) return false;
        result.dose_safety_limited = true;
        result.dose_reason = "retry_lower: " + result.dose_decision.reason;
    }
    return true;
}

bool run_coefficient_arms_batched(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_batch_runner & batch_runner,
        common_flydelta_dose_state & dose_state,
        const std::vector<std::vector<float>> & requested,
        std::vector<coefficient_arm_result> & results,
        std::string & error) {
    results.clear();
    if (requested.empty()) return true;

    auto run_batch_waves = [&](const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & batch_counterfactuals,
            std::vector<common_flydelta_decision_margin> & batch_margins,
            std::vector<common_flydelta_representation_diagnostics> & batch_geometries,
            std::vector<bool> & batch_geometry_available) {
        batch_counterfactuals.clear();
        batch_margins.clear();
        batch_geometries.clear();
        batch_geometry_available.clear();
        if (coefficients.empty()) return true;
        const size_t wave_size = config.max_batch_arms == 0
            ? coefficients.size() : config.max_batch_arms;
        for (size_t start = 0; start < coefficients.size(); start += wave_size) {
            const size_t end = std::min(coefficients.size(), start + wave_size);
            std::vector<std::vector<float>> wave_coefficients(
                coefficients.begin() + start, coefficients.begin() + end);
            std::vector<common_flydelta_counterfactual_trial> wave_counterfactuals;
            std::vector<common_flydelta_decision_margin> wave_margins;
            std::vector<common_flydelta_representation_diagnostics> wave_geometries;
            std::vector<bool> wave_geometry_available;
            if (!batch_runner(fixture, basis, wave_coefficients, wave_counterfactuals,
                    wave_margins, wave_geometries, wave_geometry_available, error) ||
                    wave_counterfactuals.size() != wave_coefficients.size() ||
                    wave_margins.size() != wave_coefficients.size() ||
                    wave_geometries.size() != wave_coefficients.size() ||
                    wave_geometry_available.size() != wave_coefficients.size()) {
                if (error.empty()) {
                    error = "FlyDelta coefficient batch returned an incomplete arm wave";
                }
                return false;
            }
            batch_counterfactuals.insert(batch_counterfactuals.end(),
                std::make_move_iterator(wave_counterfactuals.begin()),
                std::make_move_iterator(wave_counterfactuals.end()));
            batch_margins.insert(batch_margins.end(),
                std::make_move_iterator(wave_margins.begin()),
                std::make_move_iterator(wave_margins.end()));
            batch_geometries.insert(batch_geometries.end(),
                std::make_move_iterator(wave_geometries.begin()),
                std::make_move_iterator(wave_geometries.end()));
            batch_geometry_available.insert(batch_geometry_available.end(),
                wave_geometry_available.begin(), wave_geometry_available.end());
        }
        return true;
    };

    std::vector<common_flydelta_counterfactual_trial> counterfactuals;
    std::vector<common_flydelta_decision_margin> margins;
    std::vector<common_flydelta_representation_diagnostics> geometries;
    std::vector<bool> geometry_available;
    if (!run_batch_waves(requested, counterfactuals, margins, geometries,
            geometry_available) ||
            counterfactuals.size() != requested.size() ||
            margins.size() != requested.size() ||
            geometries.size() != requested.size() ||
            geometry_available.size() != requested.size()) {
        if (error.empty()) error = "FlyDelta coefficient batch returned an incomplete arm set";
        return false;
    }
    results.resize(requested.size());
    std::vector<size_t> retry_indices;
    std::vector<std::vector<float>> retry_coefficients;
    for (size_t index = 0; index < requested.size(); ++index) {
        auto & result = results[index];
        result = {};
        result.requested_coefficients = requested[index];
        result.executed_coefficients = requested[index];
        result.counterfactual = std::move(counterfactuals[index]);
        result.margin = std::move(margins[index]);
        result.geometry = std::move(geometries[index]);
        result.geometry_available = geometry_available[index];
        if (!common_flydelta_counterfactual_trial_validate(result.counterfactual, error) ||
                !common_flydelta_decision_margin_validate(result.margin, error) ||
                (result.geometry_available &&
                 !common_flydelta_representation_diagnostics_validate(result.geometry, error))) {
            return false;
        }
        const float requested_strength = norm(requested[index]);
        if (!config.use_dose_controller ||
                requested_strength <= std::numeric_limits<float>::epsilon()) {
            result.dose_decision.action = common_flydelta_dose_action::accept;
            result.dose_reason = config.use_dose_controller
                ? "zero coefficient dose" : "dose controller disabled";
            continue;
        }
        common_flydelta_dose_observation observation{
            result.geometry_available, requested_strength,
            result.geometry_available ? result.geometry.shift_norm : 0.0f,
            result.geometry_available ? result.geometry.progress : 0.0f,
            result.geometry_available ? result.geometry.leakage : 0.0f,
        };
        if (!common_flydelta_dose_observe(config.dose_policy, dose_state,
                observation, result.dose_decision, error)) return false;
        result.dose_evaluated = true;
        result.dose_safety_limited = result.dose_decision.safety_limited;
        result.dose_reason = result.dose_decision.reason;
        if (result.dose_decision.action == common_flydelta_dose_action::retry_lower &&
                result.dose_decision.proposed_safe_strength && config.max_dose_retries > 0) {
            result.executed_coefficients = rescale_coefficients(
                requested[index], requested_strength,
                *result.dose_decision.proposed_safe_strength);
            retry_indices.push_back(index);
            retry_coefficients.push_back(result.executed_coefficients);
        }
    }
    if (retry_coefficients.empty()) return true;
    counterfactuals.clear();
    margins.clear();
    geometries.clear();
    geometry_available.clear();
    if (!run_batch_waves(retry_coefficients, counterfactuals, margins, geometries,
            geometry_available) ||
            counterfactuals.size() != retry_coefficients.size() ||
            margins.size() != retry_coefficients.size() ||
            geometries.size() != retry_coefficients.size() ||
            geometry_available.size() != retry_coefficients.size()) {
        if (error.empty()) error = "FlyDelta coefficient retry batch returned an incomplete arm set";
        return false;
    }
    for (size_t retry = 0; retry < retry_indices.size(); ++retry) {
        auto & result = results[retry_indices[retry]];
        result.counterfactual = std::move(counterfactuals[retry]);
        result.margin = std::move(margins[retry]);
        result.geometry = std::move(geometries[retry]);
        result.geometry_available = geometry_available[retry];
        if (!common_flydelta_counterfactual_trial_validate(result.counterfactual, error) ||
                !common_flydelta_decision_margin_validate(result.margin, error) ||
                (result.geometry_available &&
                 !common_flydelta_representation_diagnostics_validate(result.geometry, error))) {
            return false;
        }
        const float executed_strength = norm(result.executed_coefficients);
        common_flydelta_dose_observation observation{
            result.geometry_available, executed_strength,
            result.geometry_available ? result.geometry.shift_norm : 0.0f,
            result.geometry_available ? result.geometry.progress : 0.0f,
            result.geometry_available ? result.geometry.leakage : 0.0f,
        };
        if (!common_flydelta_dose_observe(config.dose_policy, dose_state,
                observation, result.dose_decision, error)) return false;
        result.dose_safety_limited = true;
        result.dose_reason = "retry_lower: " + result.dose_decision.reason;
    }
    return true;
}

bool contains_coefficients(
        const std::vector<std::vector<float>> & values,
        const std::vector<float> & candidate) {
    return std::any_of(values.begin(), values.end(), [&](const auto & value) {
        return same_coefficients(value, candidate);
    });
}

bool better_diagnostic_trial(
        const common_flydelta_coefficient_trial & candidate,
        const common_flydelta_coefficient_trial & current) {
    if (candidate.search_fitness != current.search_fitness) {
        return candidate.search_fitness > current.search_fitness;
    }
    return norm(candidate.coefficients) < norm(current.coefficients);
}

} // namespace

static bool run_tfo_lite_coefficient_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & baseline_runner,
        const common_flydelta_coefficient_search_batch_runner & batch_runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error) {
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
        std::vector<coefficient_arm_result> pending_results;
        if (!run_coefficient_arms_batched(fixture, basis, config, batch_runner,
                dose_state, pending_coefficients, pending_results, error)) return false;
        for (size_t pending = 0; pending < pending_results.size(); ++pending) {
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
            trial.outcome = common_flydelta_classify_counterfactual(
                baseline, arm.counterfactual);
            trial.quality_delta = arm.counterfactual.quality - baseline.quality;
            trial.executed = arm.counterfactual.executed;
            trial.verifier_known = baseline.verifier_known && arm.counterfactual.verifier_known;
            if (!copy_geometry(arm.geometry, arm.geometry_available, trial, error)) return false;
            trial.iteration = iteration;
            trial.parent_trial_index = population_parents[population_index];
            trial.mutation_kind = population_mutations[population_index];
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
        if (!valid_outcome(trial.outcome) || trial.outcome != common_flydelta_counterfactual_outcome::helped ||
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

bool common_flydelta_run_low_rank_coefficient_search_batched(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & baseline_runner,
        const common_flydelta_coefficient_search_batch_runner & batch_runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error) {
    error.clear();
    trials.clear();
    selection = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_low_rank_basis_validate(basis, 16, error) ||
            !common_flydelta_coefficient_search_config_validate(
            config, basis.vectors.size(), error) || !baseline_runner || !batch_runner) {
        if (error.empty()) error = "FlyDelta coefficient search input is invalid";
        return false;
    }
    if (config.strategy == common_flydelta_coefficient_search_strategy::tfo_lite) {
        return run_tfo_lite_coefficient_search(
            fixture, basis, config, baseline_runner, batch_runner,
            trials, selection, error);
    }
    std::vector<std::vector<float>> proposals;
    if (!common_flydelta_propose_low_rank_coefficients(
            config, basis.vectors.size(), proposals, error)) return false;
    common_flydelta_counterfactual_trial baseline;
    common_flydelta_decision_margin baseline_margin;
    common_flydelta_representation_diagnostics baseline_geometry;
    bool baseline_geometry_available = false;
    if (!baseline_runner(fixture, basis, proposals.front(), false, baseline, baseline_margin,
                baseline_geometry, baseline_geometry_available, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error) ||
            (baseline_geometry_available &&
             !common_flydelta_representation_diagnostics_validate(baseline_geometry, error))) return false;
    common_flydelta_dose_state dose_state;
    std::vector<std::vector<float>> requested_coefficients;
    for (size_t index = 1; index < proposals.size(); ++index) {
        if (norm(proposals[index]) <= config.max_l2_norm) {
            requested_coefficients.push_back(proposals[index]);
        }
    }
    std::vector<coefficient_arm_result> arms;
    if (!run_coefficient_arms_batched(fixture, basis, config, batch_runner,
            dose_state, requested_coefficients, arms, error)) return false;
    for (const auto & arm : arms) {
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
        trial.outcome = common_flydelta_classify_counterfactual(
            baseline, arm.counterfactual);
        trial.quality_delta = arm.counterfactual.quality - baseline.quality;
        trial.executed = arm.counterfactual.executed;
        trial.verifier_known = baseline.verifier_known && arm.counterfactual.verifier_known;
        if (!copy_geometry(arm.geometry, arm.geometry_available, trial, error)) return false;
        trial.search_fitness = diagnostic_fitness(
            trial, baseline_margin, arm.margin, config);
        trials.push_back(std::move(trial));
    }
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (!valid_outcome(trial.outcome) || trial.outcome != common_flydelta_counterfactual_outcome::helped ||
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

bool common_flydelta_run_low_rank_coefficient_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error) {
    if (!runner) {
        error = "FlyDelta coefficient search runner is invalid";
        return false;
    }
    const common_flydelta_coefficient_search_batch_runner batch_runner =
        [&](const common_flydelta_experiment_fixture & current_fixture,
            const common_flydelta_low_rank_basis & current_basis,
            const std::vector<std::vector<float>> & coefficients,
            std::vector<common_flydelta_counterfactual_trial> & batch_trials,
            std::vector<common_flydelta_decision_margin> & batch_margins,
            std::vector<common_flydelta_representation_diagnostics> & batch_geometries,
            std::vector<bool> & batch_geometry_available,
            std::string & batch_error) {
            batch_trials.clear();
            batch_margins.clear();
            batch_geometries.clear();
            batch_geometry_available.clear();
            for (const auto & values : coefficients) {
                common_flydelta_counterfactual_trial trial;
                common_flydelta_decision_margin margin;
                common_flydelta_representation_diagnostics geometry;
                bool geometry_available = false;
                if (!runner(current_fixture, current_basis, values, true, trial, margin,
                        geometry, geometry_available, batch_error)) return false;
                batch_trials.push_back(std::move(trial));
                batch_margins.push_back(std::move(margin));
                batch_geometries.push_back(std::move(geometry));
                batch_geometry_available.push_back(geometry_available);
            }
            return true;
        };
    return common_flydelta_run_low_rank_coefficient_search_batched(
        fixture, basis, config, runner, batch_runner, trials, selection, error);
}

bool common_flydelta_append_coefficient_search_lifecycle(
        common_learning_lifecycle_store & store,
        const common_flydelta_lifecycle_event_context & context,
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const std::vector<common_flydelta_coefficient_trial> & trials,
        const std::string & experimental_artifact_id,
        std::string & error) {
    error.clear();
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_low_rank_basis_validate(basis, 16, error) ||
            !common_flydelta_coefficient_search_config_validate(
                config, basis.vectors.size(), error) ||
            experimental_artifact_id.empty() || experimental_artifact_id.size() > 512) {
        if (error.empty()) error = "FlyDelta coefficient lifecycle input is invalid";
        return false;
    }
    const std::string kind = std::string("coefficient-") +
        common_flydelta_coefficient_search_strategy_name(config.strategy);
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (!trial.executed) continue;
        common_flydelta_search_observation observation;
        observation.experiment_id = fixture.id;
        observation.candidate_id = experimental_artifact_id + "/coefficient/" +
            std::to_string(index);
        observation.search_kind = kind;
        observation.experimental_artifact_id = experimental_artifact_id;
        observation.fixture_baseline_ref = fixture.id;
        observation.surface_parent_best_ref = experimental_artifact_id;
        observation.outcome = trial.outcome;
        observation.host_evaluated = trial.executed;
        observation.verifier_known = trial.verifier_known;
        observation.diagnostics_available = trial.geometry_available;
        if (trial.geometry_available) {
            observation.diagnostics = trial.geometry;
        }
        observation.coefficients = trial.coefficients;
        observation.search_fitness = trial.search_fitness;
        observation.sequence_margin_available = trial.margin_comparison.available;
        observation.sequence_margin = trial.margin_comparison;
        observation.budget_remaining = index + 1 < trials.size();

        common_flydelta_search_decision decision;
        if (!common_flydelta_decide_search_disposition(
                observation, decision, error)) return false;

        common_flydelta_candidate_lineage lineage;
        lineage.candidate_id = observation.candidate_id;
        lineage.parent_candidate_id = experimental_artifact_id;
        if (trial.parent_trial_index < trials.size()) {
            lineage.parent_candidate_id = experimental_artifact_id + "/coefficient/" +
                std::to_string(trial.parent_trial_index);
        }
        lineage.mutation_kind = trial.mutation_kind.empty()
            ? "coefficient_arm" : trial.mutation_kind;
        lineage.generation = static_cast<uint32_t>(trial.iteration);
        lineage.direction_id = experimental_artifact_id + "/basis";
        lineage.layer_indices = {static_cast<uint32_t>(basis.layer_index)};
        lineage.scale = norm(trial.coefficients);
        lineage.intervention_budget = config.max_l2_norm;

        const std::string suffix = "/coefficient/" + std::to_string(index);
        common_flydelta_lifecycle_event_context arm_context = context;
        arm_context.event_id = context.event_id + suffix;
        arm_context.idempotency_key = context.idempotency_key + suffix;
        arm_context.source_id = context.source_id + suffix;
        if (!common_flydelta_append_search_lifecycle(
                store, arm_context, observation, decision, &lineage, error)) return false;
    }
    return true;
}
