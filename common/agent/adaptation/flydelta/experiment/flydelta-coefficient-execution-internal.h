#pragma once

#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

namespace common_flydelta_coefficient_detail {

float norm(const std::vector<float> & values);

bool valid_outcome(common_flydelta_counterfactual_outcome outcome);

std::vector<float> bound_coefficients(
        std::vector<float> coefficients, float max_l2_norm);

float diagnostic_fitness(
        const common_flydelta_coefficient_trial & trial,
        const common_flydelta_decision_margin & baseline_margin,
        const common_flydelta_decision_margin & margin,
        const common_flydelta_coefficient_search_config & config);

bool same_coefficients(
        const std::vector<float> & left,
        const std::vector<float> & right);

bool copy_geometry(
        const common_flydelta_representation_diagnostics & geometry,
        bool geometry_available,
        common_flydelta_coefficient_trial & trial,
        std::string & error);

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

bool run_coefficient_arms_batched(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_batch_runner & batch_runner,
        common_flydelta_dose_state & dose_state,
        const std::vector<std::vector<float>> & requested,
        std::vector<coefficient_arm_result> & results,
        std::string & error);

bool contains_coefficients(
        const std::vector<std::vector<float>> & values,
        const std::vector<float> & candidate);

bool better_diagnostic_trial(
        const common_flydelta_coefficient_trial & candidate,
        const common_flydelta_coefficient_trial & current);

bool run_tfo_lite_coefficient_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & baseline_runner,
        const common_flydelta_coefficient_search_batch_runner & batch_runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error);

} // namespace common_flydelta_coefficient_detail
