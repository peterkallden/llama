#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <algorithm>
#include <cmath>
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
    if (trial.outcome == common_flydelta_counterfactual_outcome::harmed) score -= 1.0f;
    return score;
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

float common_flydelta_decision_margin::normalized_delta() const {
    if (positive_token_count == 0 || negative_token_count == 0) return 0.0f;
    return positive_total_logprob / static_cast<float>(positive_token_count) -
        negative_total_logprob / static_cast<float>(negative_token_count);
}

bool common_flydelta_decision_margin_validate(
        const common_flydelta_decision_margin & margin,
        std::string & error) {
    error.clear();
    if (!margin.available) return true;
    if (!finite(margin.positive_total_logprob) || !finite(margin.negative_total_logprob) ||
            margin.positive_token_count == 0 || margin.negative_token_count == 0 ||
            !finite(margin.total_delta()) || !finite(margin.normalized_delta())) {
        error = "FlyDelta decision margin is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_coefficient_search_config_validate(
        const common_flydelta_coefficient_search_config & config,
        size_t rank,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || rank == 0 || rank > 16 || !finite(config.step) ||
            config.step <= 0.0f || config.step > 1.0f || config.max_candidates == 0 ||
            config.max_candidates > 64 || !finite(config.max_l2_norm) ||
            config.max_l2_norm <= 0.0f || config.max_l2_norm > 1.0f ||
            config.population_size == 0 || config.population_size > 16 ||
            config.iterations == 0 || config.iterations > 16 ||
            !finite(config.exploration_scale) || config.exploration_scale <= 0.0f ||
            config.exploration_scale > 2.0f || !finite(config.norm_penalty) ||
            config.norm_penalty < 0.0f || config.norm_penalty > 1.0f) {
        error = "FlyDelta coefficient search configuration is invalid";
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

static bool run_tfo_lite_coefficient_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error) {
    trials.clear();
    selection = {};

    std::vector<float> zero(basis.vectors.size(), 0.0f);
    common_flydelta_counterfactual_trial baseline;
    common_flydelta_decision_margin baseline_margin;
    if (!runner(fixture, basis, zero, false, baseline, baseline_margin, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error)) {
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
    std::vector<std::vector<float>> evaluated_coefficients;
    for (size_t iteration = 0; iteration < config.iterations && evaluated < candidate_limit;
            ++iteration) {
        for (size_t population_index = 0; population_index < population.size(); ++population_index) {
            const auto & coefficients = population[population_index];
            if (evaluated >= candidate_limit || contains_coefficients(
                    evaluated_coefficients, coefficients)) {
                continue;
            }
            common_flydelta_counterfactual_trial candidate;
            common_flydelta_decision_margin margin;
            if (!runner(fixture, basis, coefficients, true, candidate, margin, error) ||
                    !common_flydelta_counterfactual_trial_validate(candidate, error) ||
                    !common_flydelta_decision_margin_validate(margin, error)) {
                return false;
            }
            common_flydelta_coefficient_trial trial;
            trial.coefficients = coefficients;
            trial.margin = margin;
            trial.outcome = common_flydelta_classify_counterfactual(baseline, candidate);
            trial.quality_delta = candidate.quality - baseline.quality;
            trial.sequence_margin_delta = margin.available && baseline_margin.available
                ? margin.normalized_delta() - baseline_margin.normalized_delta() : 0.0f;
            trial.executed = candidate.executed;
            trial.verifier_known = baseline.verifier_known && candidate.verifier_known;
            trial.iteration = iteration;
            trial.parent_trial_index = population_parents[population_index];
            trial.mutation_kind = population_mutations[population_index];
            trial.search_fitness = diagnostic_fitness(
                trial, baseline_margin, margin, config);
            evaluated_coefficients.push_back(coefficients);
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

bool common_flydelta_run_low_rank_coefficient_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error) {
    error.clear();
    trials.clear();
    selection = {};
    if (!common_flydelta_experiment_fixture_validate(fixture, error) ||
            !common_flydelta_low_rank_basis_validate(basis, 16, error) ||
            !common_flydelta_coefficient_search_config_validate(
            config, basis.vectors.size(), error) || !runner) {
        if (error.empty()) error = "FlyDelta coefficient search input is invalid";
        return false;
    }
    if (config.strategy == common_flydelta_coefficient_search_strategy::tfo_lite) {
        return run_tfo_lite_coefficient_search(
            fixture, basis, config, runner, trials, selection, error);
    }
    std::vector<std::vector<float>> proposals;
    if (!common_flydelta_propose_low_rank_coefficients(
            config, basis.vectors.size(), proposals, error)) return false;
    common_flydelta_counterfactual_trial baseline;
    common_flydelta_decision_margin baseline_margin;
    if (!runner(fixture, basis, proposals.front(), false, baseline, baseline_margin, error) ||
            !common_flydelta_counterfactual_trial_validate(baseline, error) ||
            !common_flydelta_decision_margin_validate(baseline_margin, error)) return false;
    for (size_t index = 1; index < proposals.size(); ++index) {
        const auto & coefficients = proposals[index];
        if (norm(coefficients) > config.max_l2_norm) continue;
        common_flydelta_counterfactual_trial candidate;
        common_flydelta_decision_margin margin;
        if (!runner(fixture, basis, coefficients, true, candidate, margin, error) ||
                !common_flydelta_counterfactual_trial_validate(candidate, error) ||
                !common_flydelta_decision_margin_validate(margin, error)) return false;
        common_flydelta_coefficient_trial trial;
        trial.coefficients = coefficients;
        trial.margin = margin;
        trial.outcome = common_flydelta_classify_counterfactual(baseline, candidate);
        trial.quality_delta = candidate.quality - baseline.quality;
        trial.sequence_margin_delta = margin.available && baseline_margin.available
            ? margin.normalized_delta() - baseline_margin.normalized_delta() : 0.0f;
        trial.executed = candidate.executed;
        trial.verifier_known = baseline.verifier_known && candidate.verifier_known;
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
        observation.outcome = trial.outcome;
        observation.host_verified = true;
        observation.coefficients = trial.coefficients;
        observation.search_fitness = trial.search_fitness;
        observation.sequence_margin_available = trial.margin.available;
        observation.sequence_margin_delta = trial.sequence_margin_delta;
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
