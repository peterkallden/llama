#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace {
bool finite_value(float value) { return std::isfinite(value); }

float l2_norm(const std::vector<float> & values) {
    float total = 0.0f;
    for (const float value : values) total += value * value;
    return std::sqrt(total);
}

bool solve_ridge_system(
        const std::vector<std::vector<float>> & matrix,
        const std::vector<float> & rhs,
        std::vector<float> & solution) {
    constexpr size_t max_dimension = 64;
    const size_t dimension = rhs.size();
    if (dimension == 0 || dimension > max_dimension || matrix.size() != dimension) return false;
    std::vector<std::vector<double>> augmented(
        dimension, std::vector<double>(dimension + 1, 0.0));
    for (size_t row = 0; row < dimension; ++row) {
        if (matrix[row].size() != dimension || !finite_value(rhs[row])) return false;
        for (size_t column = 0; column < dimension; ++column) {
            if (!finite_value(matrix[row][column])) return false;
            augmented[row][column] = matrix[row][column];
        }
        augmented[row][dimension] = rhs[row];
    }
    for (size_t pivot = 0; pivot < dimension; ++pivot) {
        size_t best = pivot;
        for (size_t row = pivot + 1; row < dimension; ++row) {
            if (std::fabs(augmented[row][pivot]) > std::fabs(augmented[best][pivot])) best = row;
        }
        if (std::fabs(augmented[best][pivot]) <= 1e-12) return false;
        if (best != pivot) std::swap(augmented[best], augmented[pivot]);
        const double divisor = augmented[pivot][pivot];
        for (size_t column = pivot; column <= dimension; ++column) augmented[pivot][column] /= divisor;
        for (size_t row = 0; row < dimension; ++row) {
            if (row == pivot) continue;
            const double factor = augmented[row][pivot];
            if (std::fabs(factor) <= 1e-18) continue;
            for (size_t column = pivot; column <= dimension; ++column) {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }
    solution.resize(dimension);
    for (size_t index = 0; index < dimension; ++index) {
        solution[index] = static_cast<float>(augmented[index][dimension]);
        if (!finite_value(solution[index])) return false;
    }
    return true;
}
}

bool common_flydelta_orthogonal_search_config_validate(
        const common_flydelta_orthogonal_search_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.minimum_arms < 3 || config.minimum_arms > 64 ||
            config.maximum_arms < config.minimum_arms || config.maximum_arms > 128 ||
            !finite_value(config.minimum_residual_norm) || config.minimum_residual_norm <= 0.0f ||
            !finite_value(config.minimum_fit_quality) || config.minimum_fit_quality < 0.0f ||
            config.minimum_fit_quality > 1.0f || !finite_value(config.ridge) || config.ridge <= 0.0f ||
            !finite_value(config.minimum_cosine) || config.minimum_cosine < -1.0f ||
            config.minimum_cosine > 1.0f || !finite_value(config.maximum_leakage) ||
            config.maximum_leakage < 0.0f || !finite_value(config.maximum_shift_norm) ||
            config.maximum_shift_norm <= 0.0f || !finite_value(config.geometric_leakage_penalty) ||
            config.geometric_leakage_penalty < 0.0f) {
        error = "FlyDelta orthogonal-search configuration is invalid";
        return false;
    }
    return true;
}

const char * common_flydelta_orthogonal_response_signal_name(
        common_flydelta_orthogonal_response_signal signal) {
    switch (signal) {
        case common_flydelta_orthogonal_response_signal::none: return "none";
        case common_flydelta_orthogonal_response_signal::decision_margin: return "decision_margin";
        case common_flydelta_orthogonal_response_signal::geometric_response: return "geometric_response";
    }
    return "unknown";
}

bool common_flydelta_build_orthogonal_search_direction(
        const common_flydelta_orthogonal_search_config & config,
        const std::vector<float> & rank1_direction,
        const std::vector<common_flydelta_orthogonal_search_arm> & arms,
        common_flydelta_orthogonal_search_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_orthogonal_search_config_validate(config, error) ||
            rank1_direction.empty() || arms.size() > config.maximum_arms) {
        if (error.empty()) error = "FlyDelta orthogonal-search input is invalid";
        return false;
    }
    const float rank1_norm = l2_norm(rank1_direction);
    if (!finite_value(rank1_norm) || rank1_norm <= std::numeric_limits<float>::epsilon()) {
        error = "FlyDelta orthogonal-search rank-one direction must not be zero";
        return false;
    }
    std::vector<float> axis(rank1_direction.size());
    for (size_t i = 0; i < axis.size(); ++i) axis[i] = rank1_direction[i] / rank1_norm;
    std::vector<const common_flydelta_orthogonal_search_arm *> margin_arms;
    std::vector<const common_flydelta_orthogonal_search_arm *> geometric_arms;
    for (const auto & arm : arms) {
        if (!arm.safe_to_continue || arm.outcome == common_flydelta_counterfactual_outcome::harmed) continue;
        if (arm.intervention.size() != axis.size() ||
                (arm.decision_margin_available && !finite_value(arm.decision_margin_delta)) ||
                (arm.geometric_response_available && !finite_value(arm.geometric_response))) {
            error = "FlyDelta orthogonal-search arm dimensions are invalid";
            return false;
        }
        for (const float value : arm.intervention) {
            if (!finite_value(value)) {
                error = "FlyDelta orthogonal-search arm contains a non-finite value";
                return false;
            }
        }
        if (arm.decision_margin_available) margin_arms.push_back(&arm);
        if (arm.geometric_response_available) geometric_arms.push_back(&arm);
    }
    const auto response_signal = margin_arms.size() >= config.minimum_arms
        ? common_flydelta_orthogonal_response_signal::decision_margin
        : geometric_arms.size() >= config.minimum_arms
            ? common_flydelta_orthogonal_response_signal::geometric_response
            : common_flydelta_orthogonal_response_signal::none;
    const auto & usable = response_signal == common_flydelta_orthogonal_response_signal::decision_margin
        ? margin_arms : geometric_arms;
    if (usable.size() < config.minimum_arms) return true;
    const auto response_for = [&](const common_flydelta_orthogonal_search_arm & arm) {
        return response_signal == common_flydelta_orthogonal_response_signal::decision_margin
            ? arm.decision_margin_delta : arm.geometric_response;
    };
    std::vector<float> mean(axis.size(), 0.0f);
    float mean_response = 0.0f;
    for (const auto * arm : usable) {
        mean_response += response_for(*arm);
        for (size_t i = 0; i < mean.size(); ++i) mean[i] += arm->intervention[i];
    }
    const float count = static_cast<float>(usable.size());
    mean_response /= count;
    for (float & value : mean) value /= count;
    std::vector<std::vector<float>> normal_matrix(axis.size(), std::vector<float>(axis.size(), 0.0f));
    std::vector<float> normal_rhs(axis.size(), 0.0f);
    std::vector<std::vector<float>> residuals;
    std::vector<float> centered_responses;
    residuals.reserve(usable.size());
    centered_responses.reserve(usable.size());
    float response_energy = 0.0f;
    float fitted_energy = 0.0f;
    float covariance = 0.0f;
    for (const auto * arm : usable) {
        std::vector<float> residual(axis.size());
        for (size_t i = 0; i < residual.size(); ++i) residual[i] = arm->intervention[i] - mean[i];
        float along_axis = 0.0f;
        for (size_t i = 0; i < residual.size(); ++i) along_axis += residual[i] * axis[i];
        for (size_t i = 0; i < residual.size(); ++i) residual[i] -= along_axis * axis[i];
        const float response = response_for(*arm) - mean_response;
        response_energy += response * response;
        const float residual_norm = l2_norm(residual);
        if (residual_norm > std::numeric_limits<float>::epsilon()) {
            residuals.push_back(residual);
            centered_responses.push_back(response);
            for (size_t row = 0; row < residual.size(); ++row) {
                normal_rhs[row] += residual[row] * response;
                for (size_t column = 0; column < residual.size(); ++column) {
                    normal_matrix[row][column] += residual[row] * residual[column];
                }
            }
        }
    }
    for (size_t index = 0; index < normal_matrix.size(); ++index) normal_matrix[index][index] += config.ridge;
    std::vector<float> gradient;
    if (!solve_ridge_system(normal_matrix, normal_rhs, gradient)) return true;
    const float gradient_norm = l2_norm(gradient);
    if (!finite_value(gradient_norm) || gradient_norm <= config.minimum_residual_norm) return true;
    float axis_component = 0.0f;
    for (size_t i = 0; i < gradient.size(); ++i) axis_component += gradient[i] * axis[i];
    for (size_t i = 0; i < gradient.size(); ++i) gradient[i] -= axis_component * axis[i];
    const float residual_gradient_norm = l2_norm(gradient);
    if (!finite_value(residual_gradient_norm) || residual_gradient_norm <= config.minimum_residual_norm) return true;
    for (size_t index = 0; index < residuals.size(); ++index) {
        float predicted = 0.0f;
        for (size_t i = 0; i < residuals[index].size(); ++i) predicted += gradient[i] * residuals[index][i];
        fitted_energy += predicted * predicted;
        covariance += predicted * centered_responses[index];
    }
    const float fit_denominator = std::sqrt(std::max(fitted_energy * response_energy + config.ridge, 0.0f));
    const float fit_quality = fit_denominator > std::numeric_limits<float>::epsilon()
        ? covariance / fit_denominator : 0.0f;
    if (!finite_value(fit_quality) || fit_quality < config.minimum_fit_quality) return true;
    result.available = true;
    result.source_arm_count = usable.size();
    result.residual_norm = residual_gradient_norm;
    result.fit_quality = fit_quality;
    result.response_signal = response_signal;
    result.direction = std::move(gradient);
    const float direction_norm = l2_norm(result.direction);
    for (float & value : result.direction) value /= direction_norm;
    return true;
}

bool common_flydelta_prepare_orthogonal_search_input(
        const common_flydelta_orthogonal_search_config & config,
        const common_flydelta_bootstrap_zoom_state & state,
        common_flydelta_orthogonal_search_input & input,
        std::string & error) {
    error.clear();
    input = {};
    if (!common_flydelta_orthogonal_search_config_validate(config, error) ||
            !common_flydelta_bootstrap_zoom_state_validate(state, error) ||
            state.local_layers.empty() || state.completed_trials.empty()) {
        if (error.empty()) error = "FlyDelta orthogonal search requires persisted local BootstrapZoom trials";
        return false;
    }
    input.local_layers = state.local_layers;
    const auto flatten = [&](const common_flydelta_bootstrap_zoom_candidate & candidate,
            std::vector<float> & profile) {
        profile.assign(input.local_layers.size(), 0.0f);
        for (size_t index = 0; index < candidate.layer_indices.size(); ++index) {
            const auto it = std::lower_bound(input.local_layers.begin(), input.local_layers.end(),
                candidate.layer_indices[index]);
            if (it == input.local_layers.end() || *it != candidate.layer_indices[index]) {
                error = "FlyDelta BootstrapZoom trial falls outside its persisted local region";
                return false;
            }
            profile[static_cast<size_t>(it - input.local_layers.begin())] = candidate.layer_weights[index];
        }
        return true;
    };
    const size_t rank1_index = state.selection.selected ? state.selection.trial_index : 0;
    if (rank1_index >= state.completed_trials.size() ||
            !flatten(state.completed_trials[rank1_index].candidate, input.rank1_intervention)) {
        if (error.empty()) error = "FlyDelta rank-one BootstrapZoom trial is unavailable";
        return false;
    }
    for (const auto & trial : state.completed_trials) {
        common_flydelta_orthogonal_search_arm arm;
        if (!flatten(trial.candidate, arm.intervention)) return false;
        arm.decision_margin_available = trial.margin_available;
        arm.decision_margin_delta = trial.margin_delta;
        arm.geometric_response_available = trial.diagnostics_available;
        if (trial.diagnostics_available) {
            arm.geometric_response = trial.diagnostics.progress *
                std::max(0.0f, trial.diagnostics.cosine) -
                config.geometric_leakage_penalty * trial.diagnostics.leakage;
        }
        arm.safe_to_continue = trial.host_evaluated &&
            trial.outcome != common_flydelta_counterfactual_outcome::harmed &&
            trial.diagnostics_available && trial.diagnostics.cosine >= config.minimum_cosine &&
            trial.diagnostics.progress > 0.0f && trial.diagnostics.leakage <= config.maximum_leakage &&
            trial.diagnostics.shift_norm <= config.maximum_shift_norm;
        arm.outcome = trial.outcome;
        input.arms.push_back(std::move(arm));
    }
    return true;
}
