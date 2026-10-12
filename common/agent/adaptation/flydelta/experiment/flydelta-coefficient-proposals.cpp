#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <cmath>
#include <limits>
#include <utility>

namespace {

bool finite(float value) {
    return std::isfinite(value);
}

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

} // namespace

const char * common_flydelta_coefficient_search_strategy_name(
        common_flydelta_coefficient_search_strategy strategy) {
    switch (strategy) {
        case common_flydelta_coefficient_search_strategy::coordinate: return "coordinate";
        case common_flydelta_coefficient_search_strategy::tfo_lite: return "tfo_lite";
    }
    return "unknown";
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
