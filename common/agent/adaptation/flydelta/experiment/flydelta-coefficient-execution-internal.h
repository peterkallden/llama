#pragma once

#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"
#include <algorithm>
#include <cmath>
#include <sstream>

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

inline common_flydelta_outcome_guidance_candidate guidance_candidate(
        const std::vector<float> & coefficients,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_margin_comparison * margin = nullptr) {
    common_flydelta_outcome_guidance_candidate candidate;
    std::ostringstream id;
    id << "coeff:" << basis.layer_index;
    for (float value : coefficients) id << ':' << std::hexfloat << value;
    candidate.arm_id = id.str();
    candidate.rank = basis.vectors.size();
    uint64_t basis_hash = 1469598103934665603ULL;
    std::ostringstream basis_values;
    for (const auto & vector : basis.vectors) {
        for (const float value : vector) basis_values << std::hexfloat << value << ',';
        basis_values << ';';
    }
    for (const unsigned char ch : basis_values.str()) {
        basis_hash ^= ch;
        basis_hash *= 1099511628211ULL;
    }
    std::ostringstream basis_id;
    basis_id << "layer:" << basis.layer_index << ":rank:" << basis.vectors.size()
        << ":fnv64:" << std::hex << basis_hash;
    candidate.basis_identity = basis_id.str();
    if (config.guidance_model_layer_count > 1 && config.guidance_layer_index >= 0) {
        candidate.relative_layer = static_cast<double>(config.guidance_layer_index) /
            static_cast<double>(config.guidance_model_layer_count - 1);
    }
    const float n = norm(coefficients);
    candidate.relative_dose = config.max_l2_norm > 0
        ? std::min(1.0, static_cast<double>(n / config.max_l2_norm)) : 0.0;
    size_t nonzero = 0;
    for (float value : coefficients) if (std::abs(value) > 1e-7f) ++nonzero;
    candidate.mixed_coefficients = nonzero > 1;
    if (margin && margin->available) {
        const float delta = margin->normalized_delta();
        candidate.diagnostic_margin_sign = delta > 1e-7f ? 1 : (delta < -1e-7f ? -1 : 0);
    }
    return candidate;
}

bool run_tfo_lite_coefficient_search(
        const common_flydelta_experiment_fixture & fixture,
        const common_flydelta_low_rank_basis & basis,
        const common_flydelta_coefficient_search_config & config,
        const common_flydelta_coefficient_search_runner & baseline_runner,
        const common_flydelta_coefficient_search_batch_runner & batch_runner,
        std::vector<common_flydelta_coefficient_trial> & trials,
        common_flydelta_coefficient_selection & selection,
        std::string & error,
        const common_flydelta_outcome_guidance_adviser & adviser);

} // namespace common_flydelta_coefficient_detail
