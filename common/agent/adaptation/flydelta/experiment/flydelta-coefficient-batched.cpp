#include "agent/adaptation/flydelta/experiment/flydelta-coefficient-execution-internal.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>

namespace common_flydelta_coefficient_detail {

namespace {

bool finite(float value) {
    return std::isfinite(value);
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

} // namespace

float norm(const std::vector<float> & values) {
    float sum = 0.0f;
    for (const float value : values) sum += value * value;
    return std::sqrt(sum);
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

bool contains_coefficients(
        const std::vector<std::vector<float>> & values,
        const std::vector<float> & candidate) {
    return std::any_of(values.begin(), values.end(), [&](const auto & value) {
        return same_coefficients(value, candidate);
    });
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

bool better_diagnostic_trial(
        const common_flydelta_coefficient_trial & candidate,
        const common_flydelta_coefficient_trial & current) {
    if (candidate.search_fitness != current.search_fitness) {
        return candidate.search_fitness > current.search_fitness;
    }
    return norm(candidate.coefficients) < norm(current.coefficients);
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
    std::vector<common_flydelta_dose_state> calibrated_states(
        config.experimental_dose_mode == common_flydelta_coefficient_dose_mode::raw_coefficients
        ? 0 : requested.size());
    for (size_t index = 0; index < requested.size(); ++index) {
        auto & active_dose_state = config.experimental_dose_mode ==
            common_flydelta_coefficient_dose_mode::raw_coefficients ? dose_state : calibrated_states[index];
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
        if (!common_flydelta_dose_observe(config.dose_policy, active_dose_state,
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
        auto & active_dose_state = config.experimental_dose_mode ==
            common_flydelta_coefficient_dose_mode::raw_coefficients
            ? dose_state : calibrated_states[retry_indices[retry]];
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
        if (!common_flydelta_dose_observe(config.dose_policy, active_dose_state,
                observation, result.dose_decision, error)) return false;
        result.dose_safety_limited = true;
        result.dose_reason = "retry_lower: " + result.dose_decision.reason;
    }
    return true;
}

} // namespace common_flydelta_coefficient_detail

bool common_flydelta_run_low_rank_coefficient_search_batched(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & baseline_runner,
        const common_flydelta_coefficient_search_batch_runner & batch_runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error,
        const common_flydelta_outcome_guidance_adviser & adviser) {
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
        return common_flydelta_coefficient_detail::run_tfo_lite_coefficient_search(
            fixture, basis, config, baseline_runner, batch_runner,
            trials, selection, error, adviser);
    }
    std::vector<std::vector<float>> proposals;
    std::vector<float> requested_fractions;
    auto proposal_config = config;
    const bool refine_dose = config.experimental_dose_mode ==
        common_flydelta_coefficient_dose_mode::direction_dose && config.max_candidates >= 5;
    if (refine_dose) proposal_config.max_candidates -= 2;
    if (!common_flydelta_propose_calibrated_coefficients(
            fixture, basis, proposal_config, proposals, error, &requested_fractions)) return false;
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
        if (config.experimental_dose_mode != common_flydelta_coefficient_dose_mode::raw_coefficients ||
                common_flydelta_coefficient_detail::norm(proposals[index]) <= config.max_l2_norm) {
            requested_coefficients.push_back(proposals[index]);
        }
    }
    std::vector<common_flydelta_coefficient_detail::coefficient_arm_result> arms;
    if (!common_flydelta_coefficient_detail::run_coefficient_arms_batched(
            fixture, basis, config, batch_runner,
            dose_state, requested_coefficients, arms, error)) return false;
    const size_t coarse_arm_count = arms.size();
    if (refine_dose && !arms.empty()) {
        size_t best = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < arms.size(); ++i) {
            if (config.use_dose_controller && arms[i].dose_evaluated &&
                    arms[i].dose_decision.action != common_flydelta_dose_action::accept) continue;
            common_flydelta_coefficient_trial candidate;
            common_flydelta_coefficient_trial_set_observation(candidate, baseline, arms[i].counterfactual);
            candidate.geometry_available = arms[i].geometry_available;
            candidate.geometry = arms[i].geometry;
            candidate.coefficients = arms[i].executed_coefficients;
            const float score = common_flydelta_coefficient_detail::diagnostic_fitness(
                candidate, baseline_margin, arms[i].margin, config);
            if (std::isfinite(score) && score > best_score && arms[i].margin.available) {
                best = i; best_score = score;
            }
        }
        if (std::isfinite(best_score)) {
            std::vector<std::vector<float>> refinement;
            std::vector<float> refinement_doses;
            for (float factor : {0.75f, 1.25f}) {
                auto coefficients = arms[best].executed_coefficients;
                float previous_norm = 0;
                if (!common_flydelta_intervention_norm(basis, coefficients, previous_norm, error)) return false;
                const float dose = previous_norm / config.residual_calibration.median_norm * factor;
                if (dose > 1) continue;
                for (float & coefficient : coefficients) coefficient *= factor;
                coefficients = common_flydelta_coefficient_detail::bound_coefficients(
                    std::move(coefficients), config.max_l2_norm);
                float actual = 0;
                if (!common_flydelta_intervention_norm(basis, coefficients, actual, error)) return false;
                if (actual <= 1e-12f) continue;
                bool duplicate = false;
                for (const auto & previous : proposals) {
                    auto difference = coefficients;
                    for (size_t i = 0; i < difference.size(); ++i) difference[i] -= previous[i];
                    float distance = 0;
                    if (!common_flydelta_intervention_norm(basis, difference, distance, error)) return false;
                    if (distance <= actual * 1e-6f) { duplicate = true; break; }
                }
                if (duplicate) continue;
                refinement.push_back(coefficients);
                proposals.push_back(coefficients);
                refinement_doses.push_back(dose);
            }
            std::vector<common_flydelta_coefficient_detail::coefficient_arm_result> refined;
            if (!common_flydelta_coefficient_detail::run_coefficient_arms_batched(
                    fixture, basis, config, batch_runner, dose_state, refinement, refined, error)) return false;
            arms.insert(arms.end(), std::make_move_iterator(refined.begin()), std::make_move_iterator(refined.end()));
            requested_fractions.insert(requested_fractions.end(), refinement_doses.begin(), refinement_doses.end());
        }
    }
    for (size_t arm_index = 0; arm_index < arms.size(); ++arm_index) {
        const auto & arm = arms[arm_index];
        common_flydelta_coefficient_trial trial;
        trial.coefficients = arm.executed_coefficients;
        trial.requested_coefficients = arm.requested_coefficients;
        trial.requested_strength = common_flydelta_coefficient_detail::norm(
            arm.requested_coefficients);
        trial.executed_strength = common_flydelta_coefficient_detail::norm(
            arm.executed_coefficients);
        if (!common_flydelta_intervention_norm(basis, arm.requested_coefficients,
                trial.requested_intervention_norm, error) ||
            !common_flydelta_intervention_norm(basis, arm.executed_coefficients,
                trial.applied_intervention_norm, error)) return false;
        trial.dose_action = arm.dose_decision.action;
        trial.relative_dose = arm.dose_decision.relative_dose;
        if (config.experimental_dose_mode != common_flydelta_coefficient_dose_mode::raw_coefficients) {
            trial.requested_residual_fraction = requested_fractions.at(arm_index + 1);
            trial.requested_intervention_norm = trial.requested_residual_fraction *
                config.residual_calibration.median_norm;
            trial.applied_residual_fraction = trial.applied_intervention_norm /
                config.residual_calibration.median_norm;
            trial.residual_calibration_ref = config.residual_calibration.reference;
            trial.mutation_kind = arm_index < coarse_arm_count ? "calibrated_direction" : "dose_refinement";
        }
        trial.dose_evaluated = arm.dose_evaluated;
        trial.dose_safety_limited = arm.dose_safety_limited;
        trial.dose_reason = arm.dose_reason;
        trial.margin = arm.margin;
        trial.margin_comparison.available = baseline_margin.available && arm.margin.available;
        trial.margin_comparison.baseline = baseline_margin;
        trial.margin_comparison.candidate = arm.margin;
        common_flydelta_coefficient_trial_set_observation(trial, baseline, arm.counterfactual);
        if (!common_flydelta_coefficient_detail::copy_geometry(
                arm.geometry, arm.geometry_available, trial, error)) return false;
        trial.search_fitness = common_flydelta_coefficient_detail::diagnostic_fitness(
            trial, baseline_margin, arm.margin, config);
        trials.push_back(std::move(trial));
    }
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (config.experimental_dose_mode != common_flydelta_coefficient_dose_mode::raw_coefficients &&
                config.use_dose_controller && trial.dose_evaluated &&
                trial.dose_action != common_flydelta_dose_action::accept) continue;
        if (!common_flydelta_coefficient_detail::valid_outcome(trial.outcome) ||
                trial.outcome != common_flydelta_counterfactual_outcome::helped ||
                !trial.executed || !trial.verifier_known) continue;
        if (!selection.selected || trial.quality_delta > selection.score ||
                (trial.quality_delta == selection.score &&
                 common_flydelta_coefficient_detail::norm(trial.coefficients) <
                 common_flydelta_coefficient_detail::norm(
                     trials[selection.trial_index].coefficients))) {
            selection.selected = true;
            selection.trial_index = index;
            selection.score = trial.quality_delta;
        }
    }
    return true;
}
