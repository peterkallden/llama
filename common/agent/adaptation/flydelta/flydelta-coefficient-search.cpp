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

float dot(const std::vector<float> & left, const std::vector<float> & right) {
    float value = 0.0f;
    for (size_t i = 0; i < left.size(); ++i) value += left[i] * right[i];
    return value;
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

const char * common_flydelta_coefficient_search_strategy_name(
        common_flydelta_coefficient_search_strategy strategy) {
    switch (strategy) {
        case common_flydelta_coefficient_search_strategy::coordinate: return "coordinate";
        case common_flydelta_coefficient_search_strategy::tfo_lite: return "tfo_lite";
    }
    return "unknown";
}

bool common_flydelta_low_rank_basis_validate(
        const common_flydelta_low_rank_basis & basis,
        size_t max_rank,
        std::string & error) {
    error.clear();
    if (basis.schema_version != 1 || basis.dimension == 0 ||
            basis.dimension > (1U << 20) || basis.layer_index < 0 ||
            basis.vectors.empty() || basis.vectors.size() > max_rank) {
        error = "FlyDelta low-rank basis identity or bounds are invalid";
        return false;
    }
    for (const auto & vector : basis.vectors) {
        if (vector.size() != basis.dimension || norm(vector) <= std::numeric_limits<float>::epsilon()) {
            error = "FlyDelta low-rank basis vector is invalid";
            return false;
        }
        for (const float value : vector) {
            if (!finite(value)) {
                error = "FlyDelta low-rank basis contains a non-finite value";
                return false;
            }
        }
    }
    return true;
}

bool common_flydelta_build_low_rank_basis(
        size_t dimension,
        size_t max_rank,
        const std::vector<common_flydelta_direction_candidate> & candidates,
        common_flydelta_low_rank_basis & basis,
        std::string & error) {
    error.clear();
    basis = {};
    if (dimension == 0 || dimension > (1U << 20) || max_rank == 0 || max_rank > 16 ||
            candidates.empty() || candidates.size() > 256) {
        error = "FlyDelta low-rank basis input bounds are invalid";
        return false;
    }
    basis.dimension = dimension;
    basis.layer_index = candidates.front().layer_index;
    for (const auto & candidate : candidates) {
        if (!common_flydelta_direction_candidate_validate(candidate, dimension, error) ||
                candidate.layer_index != basis.layer_index) {
            if (error.empty()) error = "FlyDelta low-rank candidates are incompatible";
            return false;
        }
        if (basis.vectors.size() == max_rank) break;
        std::vector<float> vector = candidate.values;
        for (const auto & existing : basis.vectors) {
            const float projection = dot(vector, existing);
            for (size_t i = 0; i < vector.size(); ++i) vector[i] -= projection * existing[i];
        }
        const float vector_norm = norm(vector);
        if (!finite(vector_norm)) {
            error = "FlyDelta low-rank basis normalization failed";
            return false;
        }
        if (vector_norm <= 0.000001f) continue;
        for (float & value : vector) value /= vector_norm;
        basis.vectors.push_back(std::move(vector));
    }
    if (basis.vectors.empty()) {
        error = "FlyDelta low-rank candidates do not span a nonzero basis";
        return false;
    }
    return common_flydelta_low_rank_basis_validate(basis, max_rank, error);
}

bool common_flydelta_coefficient_search_config_validate(
        const common_flydelta_coefficient_search_config & config,
        size_t rank,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || rank == 0 || rank > 16 || !finite(config.step) ||
            config.step <= 0.0f || config.step > 1.0f || config.max_candidates == 0 ||
            config.max_candidates > 64 || !finite(config.max_l2_norm) ||
            config.max_l2_norm <= 0.0f || config.max_l2_norm > 1.0f) {
        error = "FlyDelta coefficient search configuration is invalid";
        return false;
    }
    if (config.max_batch_arms > 64) {
        error = "FlyDelta coefficient batch wave limit is invalid";
        return false;
    }
    switch (config.strategy) {
        case common_flydelta_coefficient_search_strategy::coordinate:
        case common_flydelta_coefficient_search_strategy::tfo_lite:
            break;
        default:
            error = "FlyDelta coefficient search strategy is invalid";
            return false;
    }
    if (!finite(config.norm_penalty) || config.norm_penalty < 0.0f ||
            config.norm_penalty > 1.0f || !finite(config.leakage_penalty) ||
            config.leakage_penalty < 0.0f || config.leakage_penalty > 1.0f ||
            config.max_dose_retries > 1 ||
            (config.use_dose_controller &&
             !common_flydelta_dose_policy_validate(config.dose_policy, error))) {
        error = "FlyDelta coefficient search penalties are invalid";
        return false;
    }
    if (config.strategy == common_flydelta_coefficient_search_strategy::tfo_lite &&
            (config.population_size == 0 || config.population_size > 16 ||
             config.iterations == 0 || config.iterations > 16 ||
             !finite(config.exploration_scale) || config.exploration_scale <= 0.0f ||
             config.exploration_scale > 2.0f)) {
        error = "FlyDelta TFO-lite configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_propose_low_rank_coefficients(
        const common_flydelta_coefficient_search_config & config,
        size_t rank,
        std::vector<std::vector<float>> & proposals,
        std::string & error) {
    error.clear();
    proposals.clear();
    if (!common_flydelta_coefficient_search_config_validate(config, rank, error)) return false;
    proposals.push_back(std::vector<float>(rank, 0.0f));
    for (size_t index = 0; index < rank && proposals.size() + 1 < config.max_candidates; ++index) {
        std::vector<float> positive(rank, 0.0f);
        positive[index] = config.step;
        proposals.push_back(positive);
        if (proposals.size() >= config.max_candidates) break;
        positive[index] = -config.step;
        proposals.push_back(std::move(positive));
    }
    return true;
}

bool common_flydelta_propose_shallow_rank_two_controls(
        const common_flydelta_coefficient_search_config & config,
        bool include_opposite_control,
        std::vector<std::vector<float>> & proposals,
        std::string & error) {
    error.clear();
    proposals.clear();
    if (!common_flydelta_coefficient_search_config_validate(config, 2, error)) return false;
    const size_t required = include_opposite_control ? 4 : 3;
    if (config.max_candidates < required) {
        error = "FlyDelta Shallow controls exceed coefficient candidate bound";
        return false;
    }
    const float diagonal = config.step / std::sqrt(2.0f);
    proposals = {{config.step, 0.0f}, {0.0f, config.step}, {diagonal, diagonal}};
    if (include_opposite_control) proposals.push_back({diagonal, -diagonal});
    for (auto & proposal : proposals) {
        proposal = bound_coefficients(std::move(proposal), config.max_l2_norm);
    }
    return true;
}

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
