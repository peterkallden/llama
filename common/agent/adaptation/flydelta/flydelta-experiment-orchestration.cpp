#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace {
bool finite(float value) { return std::isfinite(value); }

float l2_norm(const std::vector<float> & values) {
    float total = 0.0f;
    for (const float value : values) total += value * value;
    return std::sqrt(total);
}

// Orthogonal search operates in the small persisted BootstrapZoom profile
// space (normally two or three local layers). Keep the solve bounded so a
// malformed/high-dimensional input cannot turn this CPU control-plane helper
// into an unexpectedly large dense allocation.
bool solve_ridge_system(
        const std::vector<std::vector<float>> & matrix,
        const std::vector<float> & rhs,
        std::vector<float> & solution) {
    constexpr size_t max_dimension = 64;
    const size_t dimension = rhs.size();
    if (dimension == 0 || dimension > max_dimension || matrix.size() != dimension) {
        return false;
    }
    std::vector<std::vector<double>> augmented(
        dimension, std::vector<double>(dimension + 1, 0.0));
    for (size_t row = 0; row < dimension; ++row) {
        if (matrix[row].size() != dimension || !finite(rhs[row])) return false;
        for (size_t column = 0; column < dimension; ++column) {
            if (!finite(matrix[row][column])) return false;
            augmented[row][column] = matrix[row][column];
        }
        augmented[row][dimension] = rhs[row];
    }

    for (size_t pivot = 0; pivot < dimension; ++pivot) {
        size_t best = pivot;
        for (size_t row = pivot + 1; row < dimension; ++row) {
            if (std::fabs(augmented[row][pivot]) > std::fabs(augmented[best][pivot])) {
                best = row;
            }
        }
        if (std::fabs(augmented[best][pivot]) <= 1e-12) return false;
        if (best != pivot) std::swap(augmented[best], augmented[pivot]);
        const double divisor = augmented[pivot][pivot];
        for (size_t column = pivot; column <= dimension; ++column) {
            augmented[pivot][column] /= divisor;
        }
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
        if (!finite(solution[index])) return false;
    }
    return true;
}

bool valid_zoom_phase(common_flydelta_bootstrap_zoom_phase phase) {
    return phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom ||
        phase == common_flydelta_bootstrap_zoom_phase::profile_zoom ||
        phase == common_flydelta_bootstrap_zoom_phase::sign_control ||
        phase == common_flydelta_bootstrap_zoom_phase::adaptive_alpha;
}

bool valid_zoom_candidate(
        const common_flydelta_bootstrap_zoom_candidate & candidate,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1 || !valid_zoom_phase(candidate.phase) ||
            candidate.layer_indices.empty() || candidate.layer_indices.size() > 3 ||
            candidate.layer_indices.size() != candidate.layer_weights.size() ||
            !std::is_sorted(candidate.layer_indices.begin(), candidate.layer_indices.end()) ||
            candidate.layer_indices.front() == 0 ||
            std::adjacent_find(candidate.layer_indices.begin(), candidate.layer_indices.end()) !=
                candidate.layer_indices.end() || !finite(candidate.total_scale) ||
            candidate.total_scale <= 0.0f || candidate.total_scale > 1.0f) {
        error = "FlyDelta BootstrapZoom candidate is invalid";
        return false;
    }
    for (const float weight : candidate.layer_weights) {
        if (!finite(weight)) {
            error = "FlyDelta BootstrapZoom layer weight is invalid";
            return false;
        }
    }
    if (std::fabs(l2_norm(candidate.layer_weights) - 1.0f) > 0.0001f) {
        error = "FlyDelta BootstrapZoom profile must have unit L2 energy";
        return false;
    }
    if (candidate.phase == common_flydelta_bootstrap_zoom_phase::alpha_zoom &&
            (candidate.layer_indices.size() != 1 || candidate.opposite_sign_control)) {
        error = "FlyDelta BootstrapZoom alpha candidate must be a positive singleton";
        return false;
    }
    if (candidate.phase == common_flydelta_bootstrap_zoom_phase::sign_control &&
            !candidate.opposite_sign_control) {
        error = "FlyDelta BootstrapZoom sign control must be marked";
        return false;
    }
    return true;
}

bool valid_depth(const common_flydelta_evidence_depth_result & value) {
    // A search surface may be continued experimentally before any natural
    // HELPED evidence exists. That is still Bootstrap capacity; it must not
    // be mistaken for rank-backed Shallow/Deep capacity.
    const bool empty_bootstrap = value.compatible_samples == 0 &&
        value.experimental_samples > 0 &&
        value.depth == common_flydelta_search_depth::bootstrap &&
        value.effective_rank == 0;
    const bool ranked = value.compatible_samples > 0 && value.effective_rank > 0;
    return (empty_bootstrap || ranked) && finite(value.stable_rank) &&
        finite(value.median_alignment) && finite(value.condition_number);
}

bool valid_depth_value(common_flydelta_search_depth value) {
    return value == common_flydelta_search_depth::bootstrap ||
        value == common_flydelta_search_depth::shallow ||
        value == common_flydelta_search_depth::deep;
}

bool safe_geometry(
        const common_flydelta_representation_diagnostics & value,
        const common_flydelta_utility_gate_config & config) {
    return value.cosine >= config.min_cosine && value.progress > 0.0f &&
        value.leakage <= config.max_leakage && value.shift_norm <= config.max_shift_norm;
}

size_t required_observations(
        common_flydelta_experiment_phase phase,
        const common_flydelta_utility_gate_config & config) {
    switch (phase) {
        case common_flydelta_experiment_phase::bootstrap: return config.shallow_enter_observations;
        case common_flydelta_experiment_phase::shallow_controls: return config.deep_enter_observations;
        case common_flydelta_experiment_phase::deep_controls: return config.tfo_enter_observations;
    }
    return config.tfo_enter_observations;
}

float required_margin(
        common_flydelta_experiment_phase phase,
        const common_flydelta_utility_gate_config & config) {
    switch (phase) {
        case common_flydelta_experiment_phase::bootstrap: return config.shallow_enter_margin;
        case common_flydelta_experiment_phase::shallow_controls: return config.deep_enter_margin;
        case common_flydelta_experiment_phase::deep_controls: return config.tfo_enter_margin;
    }
    return config.tfo_enter_margin;
}
} // namespace

const char * common_flydelta_experiment_phase_name(
        common_flydelta_experiment_phase phase) {
    switch (phase) {
        case common_flydelta_experiment_phase::bootstrap: return "bootstrap";
        case common_flydelta_experiment_phase::shallow_controls: return "shallow_controls";
        case common_flydelta_experiment_phase::deep_controls: return "deep_controls";
    }
    return "bootstrap";
}

const char * common_flydelta_bootstrap_refinement_kind_name(
        common_flydelta_bootstrap_refinement_kind kind) {
    switch (kind) {
        case common_flydelta_bootstrap_refinement_kind::bootstrap_zoom:
            return "bootstrap_zoom";
        case common_flydelta_bootstrap_refinement_kind::adaptive_alpha:
            return "adaptive_alpha";
    }
    return "bootstrap_zoom";
}

const char * common_flydelta_utility_gate_action_name(
        common_flydelta_utility_gate_action action) {
    switch (action) {
        case common_flydelta_utility_gate_action::stop: return "stop";
        case common_flydelta_utility_gate_action::retain: return "retain";
        case common_flydelta_utility_gate_action::refine_bootstrap: return "refine_bootstrap";
        case common_flydelta_utility_gate_action::orthogonal_search: return "orthogonal_search";
        case common_flydelta_utility_gate_action::escalate_shallow: return "escalate_shallow";
        case common_flydelta_utility_gate_action::escalate_deep: return "escalate_deep";
        case common_flydelta_utility_gate_action::allow_tfo_lite: return "allow_tfo_lite";
    }
    return "retain";
}

const char * common_flydelta_next_action_name(common_flydelta_next_action action) {
    switch (action) {
        case common_flydelta_next_action::stop: return "stop";
        case common_flydelta_next_action::retain: return "retain";
        case common_flydelta_next_action::run_bootstrap: return "run_bootstrap";
        case common_flydelta_next_action::refine_bootstrap: return "refine_bootstrap";
        case common_flydelta_next_action::run_orthogonal_search: return "run_orthogonal_search";
        case common_flydelta_next_action::run_shallow_controls: return "run_shallow_controls";
        case common_flydelta_next_action::run_deep_controls: return "run_deep_controls";
        case common_flydelta_next_action::recenter_surface: return "recenter_surface";
        case common_flydelta_next_action::run_representation_augmentation:
            return "run_representation_augmentation";
        case common_flydelta_next_action::prepare_concept_material:
            return "prepare_concept_material";
        case common_flydelta_next_action::run_concept_synthesis:
            return "run_concept_synthesis";
        case common_flydelta_next_action::allow_tfo_lite: return "allow_tfo_lite";
    }
    return "retain";
}

bool common_flydelta_rank1_plateau_config_validate(
        const common_flydelta_rank1_plateau_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.minimum_bootstrap_arms < 2 ||
            config.minimum_bootstrap_arms > 64 || config.required_plateau_rounds == 0 ||
            config.required_plateau_rounds > 8 || !finite(config.minimum_useful_margin) ||
            !finite(config.maximum_recent_gain_ratio) || config.maximum_recent_gain_ratio < 0.0f ||
            config.maximum_recent_gain_ratio > 1.0f || config.maximum_region_span == 0 ||
            config.maximum_region_span > 64 || !finite(config.shallow_rank_threshold) ||
            config.shallow_rank_threshold <= 1.0f) {
        error = "FlyDelta rank-one plateau configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_evaluate_rank1_plateau(
        const common_flydelta_rank1_plateau_config & config,
        float effective_rank,
        const std::vector<common_flydelta_rank1_plateau_round> & rounds,
        common_flydelta_rank1_plateau_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_rank1_plateau_config_validate(config, error) ||
            !finite(effective_rank) || effective_rank < 0.0f || rounds.empty() ||
            rounds.size() > 32) {
        if (error.empty()) error = "FlyDelta rank-one plateau input is invalid";
        return false;
    }
    result.minimum_anchor_layer = std::numeric_limits<uint32_t>::max();
    for (const auto & round : rounds) {
        if (!round.safe_to_continue || round.evaluated_arms == 0 ||
                !finite(round.best_margin_delta) || round.anchor_layer == 0) {
            result = {};
            result.action = common_flydelta_rank1_plateau_action::continue_bootstrap;
            return true;
        }
        result.safe_arm_count += round.evaluated_arms;
        result.best_margin_delta = std::max(result.best_margin_delta, round.best_margin_delta);
        result.minimum_anchor_layer = std::min(result.minimum_anchor_layer, round.anchor_layer);
        result.maximum_anchor_layer = std::max(result.maximum_anchor_layer, round.anchor_layer);
    }
    if (result.safe_arm_count < config.minimum_bootstrap_arms ||
            effective_rank >= config.shallow_rank_threshold ||
            result.best_margin_delta <= config.minimum_useful_margin ||
            result.maximum_anchor_layer - result.minimum_anchor_layer > config.maximum_region_span) {
        result.minimum_anchor_layer = result.minimum_anchor_layer ==
            std::numeric_limits<uint32_t>::max() ? 0 : result.minimum_anchor_layer;
        result.action = common_flydelta_rank1_plateau_action::continue_bootstrap;
        return true;
    }
    result.eligible = true;
    if (rounds.size() < 2) {
        result.action = common_flydelta_rank1_plateau_action::refine_bootstrap;
        return true;
    }
    for (size_t index = rounds.size(); index > 1; --index) {
        const float previous = rounds[index - 2].best_margin_delta;
        const float current = rounds[index - 1].best_margin_delta;
        if (current < previous) break;
        const float recent_gain = current - previous;
        const float ratio = recent_gain /
            std::max(std::fabs(current), std::numeric_limits<float>::epsilon());
        if (result.plateau_streak == 0) result.recent_gain_ratio = ratio;
        if (ratio > config.maximum_recent_gain_ratio) break;
        ++result.plateau_streak;
    }
    result.plateau = result.plateau_streak >= config.required_plateau_rounds;
    result.action = result.plateau
        ? common_flydelta_rank1_plateau_action::orthogonal_search
        : common_flydelta_rank1_plateau_action::refine_bootstrap;
    return true;
}

bool common_flydelta_decide_rank1_plateau_utility(
        const common_flydelta_rank1_plateau_result & plateau,
        common_flydelta_utility_gate_decision & decision,
        std::string & error) {
    error.clear();
    decision = {};
    if (!finite(plateau.best_margin_delta) || !finite(plateau.recent_gain_ratio)) {
        error = "FlyDelta rank-one plateau result is invalid";
        return false;
    }
    decision.history = {plateau.plateau_streak, 0};
    if (!plateau.eligible) {
        decision.action = common_flydelta_utility_gate_action::retain;
        return true;
    }
    decision.utility_qualified = true;
    decision.action = plateau.action == common_flydelta_rank1_plateau_action::orthogonal_search
        ? common_flydelta_utility_gate_action::orthogonal_search
        : plateau.action == common_flydelta_rank1_plateau_action::refine_bootstrap
            ? common_flydelta_utility_gate_action::refine_bootstrap
            : common_flydelta_utility_gate_action::retain;
    return true;
}

bool common_flydelta_orthogonal_search_config_validate(
        const common_flydelta_orthogonal_search_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.minimum_arms < 3 || config.minimum_arms > 64 ||
            config.maximum_arms < config.minimum_arms || config.maximum_arms > 128 ||
            !finite(config.minimum_residual_norm) || config.minimum_residual_norm <= 0.0f ||
            !finite(config.minimum_fit_quality) || config.minimum_fit_quality < 0.0f ||
            config.minimum_fit_quality > 1.0f || !finite(config.ridge) || config.ridge <= 0.0f ||
            !finite(config.minimum_cosine) || config.minimum_cosine < -1.0f ||
            config.minimum_cosine > 1.0f || !finite(config.maximum_leakage) ||
            config.maximum_leakage < 0.0f || !finite(config.maximum_shift_norm) ||
            config.maximum_shift_norm <= 0.0f ||
            !finite(config.geometric_leakage_penalty) ||
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
        case common_flydelta_orthogonal_response_signal::decision_margin:
            return "decision_margin";
        case common_flydelta_orthogonal_response_signal::geometric_response:
            return "geometric_response";
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
    if (!finite(rank1_norm) || rank1_norm <= std::numeric_limits<float>::epsilon()) {
        error = "FlyDelta orthogonal-search rank-one direction must not be zero";
        return false;
    }
    std::vector<float> axis(rank1_direction.size());
    for (size_t i = 0; i < axis.size(); ++i) axis[i] = rank1_direction[i] / rank1_norm;

    std::vector<const common_flydelta_orthogonal_search_arm *> margin_arms;
    std::vector<const common_flydelta_orthogonal_search_arm *> geometric_arms;
    for (const auto & arm : arms) {
        if (!arm.safe_to_continue ||
                arm.outcome == common_flydelta_counterfactual_outcome::harmed) continue;
        if (arm.intervention.size() != axis.size() ||
                (arm.decision_margin_available && !finite(arm.decision_margin_delta)) ||
                (arm.geometric_response_available && !finite(arm.geometric_response))) {
            error = "FlyDelta orthogonal-search arm dimensions are invalid";
            return false;
        }
        for (const float value : arm.intervention) {
            if (!finite(value)) {
                error = "FlyDelta orthogonal-search arm contains a non-finite value";
                return false;
            }
        }
        if (arm.decision_margin_available) margin_arms.push_back(&arm);
        if (arm.geometric_response_available) geometric_arms.push_back(&arm);
    }
    const common_flydelta_orthogonal_response_signal response_signal =
        margin_arms.size() >= config.minimum_arms
            ? common_flydelta_orthogonal_response_signal::decision_margin
            : geometric_arms.size() >= config.minimum_arms
                ? common_flydelta_orthogonal_response_signal::geometric_response
                : common_flydelta_orthogonal_response_signal::none;
    const auto & usable = response_signal ==
            common_flydelta_orthogonal_response_signal::decision_margin
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

    std::vector<std::vector<float>> normal_matrix(
        axis.size(), std::vector<float>(axis.size(), 0.0f));
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
        const float along_axis = [&]() {
            float value = 0.0f;
            for (size_t i = 0; i < residual.size(); ++i) value += residual[i] * axis[i];
            return value;
        }();
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
    for (size_t index = 0; index < normal_matrix.size(); ++index) {
        normal_matrix[index][index] += config.ridge;
    }
    std::vector<float> gradient;
    if (!solve_ridge_system(normal_matrix, normal_rhs, gradient)) return true;
    const float gradient_norm = l2_norm(gradient);
    if (!finite(gradient_norm) || gradient_norm <= config.minimum_residual_norm) return true;
    const float axis_component = [&]() {
        float value = 0.0f;
        for (size_t i = 0; i < gradient.size(); ++i) value += gradient[i] * axis[i];
        return value;
    }();
    for (size_t i = 0; i < gradient.size(); ++i) gradient[i] -= axis_component * axis[i];
    const float residual_gradient_norm = l2_norm(gradient);
    if (!finite(residual_gradient_norm) || residual_gradient_norm <= config.minimum_residual_norm) return true;
    for (size_t index = 0; index < residuals.size(); ++index) {
        const auto & residual = residuals[index];
        const float predicted = [&]() {
            float value = 0.0f;
            for (size_t i = 0; i < residual.size(); ++i) value += gradient[i] * residual[i];
            return value;
        }();
        fitted_energy += predicted * predicted;
        covariance += predicted * centered_responses[index];
    }
    const float fit_denominator = std::sqrt(std::max(
        fitted_energy * response_energy + config.ridge, 0.0f));
    const float fit_quality = fit_denominator > std::numeric_limits<float>::epsilon()
        ? covariance / fit_denominator : 0.0f;
    if (!finite(fit_quality) || fit_quality < config.minimum_fit_quality) return true;
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

const char * common_flydelta_bootstrap_zoom_phase_name(
        common_flydelta_bootstrap_zoom_phase phase) {
    switch (phase) {
        case common_flydelta_bootstrap_zoom_phase::alpha_zoom: return "alpha_zoom";
        case common_flydelta_bootstrap_zoom_phase::profile_zoom: return "profile_zoom";
        case common_flydelta_bootstrap_zoom_phase::sign_control: return "sign_control";
        case common_flydelta_bootstrap_zoom_phase::adaptive_alpha: return "adaptive_alpha";
    }
    return "alpha_zoom";
}

bool common_flydelta_utility_gate_config_validate(
        const common_flydelta_utility_gate_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || !finite(config.shallow_enter_margin) ||
            !finite(config.deep_enter_margin) || !finite(config.tfo_enter_margin) ||
            config.shallow_enter_observations == 0 || config.deep_enter_observations == 0 ||
            config.tfo_enter_observations == 0 || config.exit_nonqualifying_observations == 0 ||
            !finite(config.min_cosine) || config.min_cosine < -1.0f || config.min_cosine > 1.0f ||
            !finite(config.max_leakage) || config.max_leakage < 0.0f ||
            !finite(config.max_shift_norm) || config.max_shift_norm <= 0.0f) {
        error = "FlyDelta utility gate configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_subspace_utility_observation_validate(
        const common_flydelta_subspace_utility_observation & observation,
        std::string & error) {
    error.clear();
    if (!finite(observation.decision_margin_delta) ||
            (observation.geometry_available &&
                !common_flydelta_representation_diagnostics_validate(observation.geometry, error))) {
        if (error.empty()) error = "FlyDelta subspace utility observation is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_decide_subspace_utility(
        const common_flydelta_utility_gate_config & config,
        common_flydelta_search_depth max_allowed_depth,
        common_flydelta_experiment_phase current_phase,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_utility_gate_decision & decision,
        std::string & error) {
    error.clear();
    decision = {};
    if (!common_flydelta_utility_gate_config_validate(config, error) ||
            !valid_depth_value(max_allowed_depth) || observations.empty()) {
        if (error.empty()) error = "FlyDelta utility gate input is invalid";
        return false;
    }
    bool qualified = false;
    for (const auto & observation : observations) {
        if (!common_flydelta_subspace_utility_observation_validate(observation, error)) return false;
        if (observation.alpha_response_available &&
                observation.alpha_response_status ==
                    common_flydelta_alpha_response_status::safety_limited) {
            decision.action = common_flydelta_utility_gate_action::stop;
            decision.history = {0, history.nonqualifying_streak + 1};
            return true;
        }
        if (!observation.safe_to_continue) {
            decision.action = common_flydelta_utility_gate_action::stop;
            decision.history = {0, history.nonqualifying_streak + 1};
            return true;
        }
        const bool geometry_ok = !observation.geometry_available || safe_geometry(observation.geometry, config);
        const bool alpha_can_continue = !observation.alpha_response_available ||
            observation.alpha_range_not_exhausted ||
            observation.alpha_response_status ==
                common_flydelta_alpha_response_status::helped;
        if (observation.decision_margin_available && geometry_ok &&
                alpha_can_continue &&
                observation.decision_margin_delta > required_margin(current_phase, config)) {
            qualified = true;
        }
    }
    decision.utility_qualified = qualified;
    decision.history = qualified
        ? common_flydelta_utility_history{history.qualifying_streak + 1, 0}
        : common_flydelta_utility_history{0, history.nonqualifying_streak + 1};
    if (!qualified) {
        decision.action = decision.history.nonqualifying_streak >=
                config.exit_nonqualifying_observations
            ? common_flydelta_utility_gate_action::stop
            : common_flydelta_utility_gate_action::retain;
        return true;
    }
    if (decision.history.qualifying_streak < required_observations(current_phase, config)) {
        decision.action = common_flydelta_utility_gate_action::retain;
        return true;
    }
    if (current_phase == common_flydelta_experiment_phase::bootstrap) {
        decision.action = max_allowed_depth == common_flydelta_search_depth::bootstrap
            ? common_flydelta_utility_gate_action::refine_bootstrap
            : common_flydelta_utility_gate_action::escalate_shallow;
    } else if (current_phase == common_flydelta_experiment_phase::shallow_controls &&
            max_allowed_depth == common_flydelta_search_depth::deep) {
        decision.action = common_flydelta_utility_gate_action::escalate_deep;
    } else if (current_phase == common_flydelta_experiment_phase::deep_controls &&
            max_allowed_depth == common_flydelta_search_depth::deep) {
        decision.action = common_flydelta_utility_gate_action::allow_tfo_lite;
    } else {
        decision.action = common_flydelta_utility_gate_action::retain;
    }
    return true;
}

bool common_flydelta_bootstrap_zoom_config_validate(
        const common_flydelta_bootstrap_zoom_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.max_extra_model_trials == 0 ||
            config.max_extra_model_trials > 10 || config.alpha_multipliers.empty() ||
            config.alpha_multipliers.size() > 4 || !finite(config.min_margin_improvement) ||
            config.min_margin_improvement < 0.0f) {
        error = "FlyDelta BootstrapZoom configuration is invalid";
        return false;
    }
    for (size_t index = 0; index < config.alpha_multipliers.size(); ++index) {
        const float multiplier = config.alpha_multipliers[index];
        if (!finite(multiplier) || multiplier <= 0.0f || multiplier > 4.0f ||
                (index > 0 && multiplier <= config.alpha_multipliers[index - 1])) {
            error = "FlyDelta BootstrapZoom alpha multipliers are invalid";
            return false;
        }
    }
    return true;
}

bool common_flydelta_bootstrap_zoom_candidate_validate(
        const common_flydelta_bootstrap_zoom_candidate & candidate,
        std::string & error) {
    return valid_zoom_candidate(candidate, error);
}

bool common_flydelta_bootstrap_zoom_trial_validate(
        const common_flydelta_bootstrap_zoom_trial & trial,
        std::string & error) {
    error.clear();
    // Diagnostics-first trials intentionally omit full generation and host
    // verification. They are valid when the host supplied bounded geometry;
    // only the frontier trials require host_evaluated=true.
    if (!valid_zoom_candidate(trial.candidate, error) ||
            (!trial.host_evaluated && !trial.diagnostics_available) ||
            (trial.verifier_known && !trial.host_evaluated) ||
            !finite(trial.margin_delta) ||
            (trial.diagnostics_available &&
                !common_flydelta_representation_diagnostics_validate(
                    trial.diagnostics, error))) {
        if (error.empty()) error = "FlyDelta BootstrapZoom trial is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_bootstrap_zoom_selection_validate(
        const common_flydelta_bootstrap_zoom_selection & selection,
        size_t trial_count,
        std::string & error) {
    error.clear();
    if (!finite(selection.search_score) ||
            (selection.selected && selection.trial_index >= trial_count) ||
            (!selection.selected && selection.trial_index != 0)) {
        error = "FlyDelta BootstrapZoom selection is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_select_bootstrap_zoom_trial(
        const std::vector<common_flydelta_bootstrap_zoom_trial> & trials,
        common_flydelta_bootstrap_zoom_selection & selection,
        std::string & error) {
    error.clear();
    selection = {};
    if (trials.empty() || trials.size() > 8) {
        error = "FlyDelta BootstrapZoom trial set is empty or exceeds its bound";
        return false;
    }
    for (size_t index = 0; index < trials.size(); ++index) {
        const auto & trial = trials[index];
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
        if (trial.outcome == common_flydelta_counterfactual_outcome::harmed) continue;
        const bool geometry_safe = !trial.diagnostics_available ||
            (trial.diagnostics.cosine >= 0.3f && trial.diagnostics.progress > 0.0f &&
             trial.diagnostics.leakage <= 1.0f && trial.diagnostics.shift_norm <= 1.0f);
        const bool helped = trial.outcome == common_flydelta_counterfactual_outcome::helped;
        if (!helped && !geometry_safe) continue;
        const float score = trial.margin_available ? trial.margin_delta : 0.0f;
        if (!selection.selected) {
            selection = {true, index, score};
            continue;
        }
        const auto & best = trials[selection.trial_index];
        const bool best_helped = best.outcome == common_flydelta_counterfactual_outcome::helped;
        if ((helped && !best_helped) ||
                (helped == best_helped &&
                    (score > selection.search_score ||
                     (score == selection.search_score &&
                      trial.candidate.total_scale < best.candidate.total_scale)))) {
            selection = {true, index, score};
        }
    }
    if (!selection.selected) {
        error = "FlyDelta BootstrapZoom has no safe retained trial";
        return false;
    }
    return true;
}

bool common_flydelta_bootstrap_zoom_state_validate(
        const common_flydelta_bootstrap_zoom_state & state,
        std::string & error) {
    error.clear();
    if (state.schema_version != 1 || state.state_ref.size() > 512 ||
            state.behavior_key.empty() || state.behavior_key.size() > 512 ||
            state.model_profile_fingerprint.empty() ||
            state.model_profile_fingerprint.size() > 512 ||
            state.capture_layout_revision.empty() || state.capture_layout_revision.size() > 512 ||
            !valid_zoom_phase(state.phase) || state.anchor_layer == 0 ||
            !finite(state.selected_scale) || state.selected_scale <= 0.0f ||
            state.selected_scale > 1.0f || !finite(state.best_margin_delta) ||
            !finite(state.best_search_score) || state.extra_model_trials > 10 ||
            state.next_candidate_index > state.extra_model_trials ||
            state.surface_revision == 0 ||
            state.parent_surface_revision >= state.surface_revision ||
            state.search_rank == 0 || state.search_rank > 4 ||
            !finite(state.evidence_rank) || state.evidence_rank <= 0.0f ||
            state.evidence_rank > 1024.0f || state.surface_origin.empty() ||
            state.surface_origin.size() > 128 || state.parent_surface_ref.size() > 512 ||
            state.local_layers.size() > 3 ||
            state.completed_trials.size() > 8 ||
            state.surface_trials.size() > 8 ||
            (state.phase == common_flydelta_bootstrap_zoom_phase::adaptive_alpha &&
                state.refinement_kind != common_flydelta_bootstrap_refinement_kind::adaptive_alpha) ||
            (state.alpha_response_available &&
                (!finite(state.alpha_response.scale) ||
                 !finite(state.alpha_response.utility) ||
                 !finite(state.alpha_response.last_scale) ||
                 !finite(state.alpha_response.last_utility) ||
                 !finite(state.alpha_response.utility_slope) ||
                 !finite(state.alpha_response.max_reachable_scale) ||
                 !finite(state.alpha_response.minimum_effective_scale) ||
                 !finite(state.alpha_response.best_margin_delta_total) ||
                 !finite(state.alpha_response.best_margin_delta_normalized))) ||
            (!state.local_layers.empty() &&
                (!std::is_sorted(state.local_layers.begin(), state.local_layers.end()) ||
                 state.local_layers.front() == 0 ||
                 std::adjacent_find(state.local_layers.begin(), state.local_layers.end()) !=
                     state.local_layers.end()))) {
        error = "FlyDelta BootstrapZoom state is invalid";
        return false;
    }
    if (!state.local_layers.empty() && !std::binary_search(
            state.local_layers.begin(), state.local_layers.end(), state.anchor_layer)) {
        error = "FlyDelta BootstrapZoom state anchor is not local";
        return false;
    }
    for (const auto & trial : state.completed_trials) {
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
    }
    for (const auto & trial : state.surface_trials) {
        if (!common_flydelta_bootstrap_zoom_trial_validate(trial, error)) return false;
    }
    if (!common_flydelta_bootstrap_zoom_selection_validate(
            state.selection, state.completed_trials.size(), error)) return false;
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
        if (error.empty()) {
            error = "FlyDelta orthogonal search requires persisted local BootstrapZoom trials";
        }
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
            profile[static_cast<size_t>(it - input.local_layers.begin())] =
                candidate.layer_weights[index];
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
            // The Euclidean relative dose remains available to the dose
            // controller, but is not itself a search utility: leakage must
            // not make an arm look more promising. This fallback therefore
            // uses directional progress with an explicit leakage penalty.
            arm.geometric_response = trial.diagnostics.progress *
                std::max(0.0f, trial.diagnostics.cosine) -
                config.geometric_leakage_penalty * trial.diagnostics.leakage;
        }
        arm.safe_to_continue = trial.host_evaluated &&
            trial.outcome != common_flydelta_counterfactual_outcome::harmed &&
            trial.diagnostics_available &&
            trial.diagnostics.cosine >= config.minimum_cosine &&
            trial.diagnostics.progress > 0.0f &&
            trial.diagnostics.leakage <= config.maximum_leakage &&
            trial.diagnostics.shift_norm <= config.maximum_shift_norm;
        arm.outcome = trial.outcome;
        input.arms.push_back(std::move(arm));
    }
    return true;
}

bool common_flydelta_propose_bootstrap_alpha_zoom(
        uint32_t anchor_layer,
        float base_scale,
        const common_flydelta_bootstrap_zoom_config & config,
        std::vector<common_flydelta_bootstrap_zoom_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    if (anchor_layer == 0 || !finite(base_scale) || base_scale <= 0.0f ||
            !common_flydelta_bootstrap_zoom_config_validate(config, error)) {
        if (error.empty()) error = "FlyDelta BootstrapZoom alpha input is invalid";
        return false;
    }
    for (const float multiplier : config.alpha_multipliers) {
        const float scale = base_scale * multiplier;
        if (scale > 1.0f || candidates.size() >= config.max_extra_model_trials) continue;
        common_flydelta_bootstrap_zoom_candidate candidate;
        candidate.phase = common_flydelta_bootstrap_zoom_phase::alpha_zoom;
        candidate.layer_indices = {anchor_layer};
        candidate.layer_weights = {1.0f};
        candidate.total_scale = scale;
        if (!valid_zoom_candidate(candidate, error)) return false;
        candidates.push_back(std::move(candidate));
    }
    if (candidates.empty()) {
        error = "FlyDelta BootstrapZoom alpha probes exceed the intervention bound";
        return false;
    }
    return true;
}

bool common_flydelta_propose_bootstrap_profile_zoom(
        const std::vector<uint32_t> & local_layers,
        uint32_t anchor_layer,
        float selected_scale,
        const common_flydelta_bootstrap_zoom_config & config,
        std::vector<common_flydelta_bootstrap_zoom_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    if (local_layers.empty() || local_layers.size() > 3 || !std::is_sorted(
                local_layers.begin(), local_layers.end()) || local_layers.front() == 0 ||
            std::adjacent_find(local_layers.begin(), local_layers.end()) != local_layers.end() ||
            !std::binary_search(local_layers.begin(), local_layers.end(), anchor_layer) ||
            !finite(selected_scale) || selected_scale <= 0.0f || selected_scale > 1.0f ||
            !common_flydelta_bootstrap_zoom_config_validate(config, error)) {
        if (error.empty()) error = "FlyDelta BootstrapZoom profile input is invalid";
        return false;
    }
    const auto append = [&](std::vector<uint32_t> layers, bool sign_control) {
        common_flydelta_bootstrap_zoom_candidate candidate;
        candidate.phase = sign_control ? common_flydelta_bootstrap_zoom_phase::sign_control :
            common_flydelta_bootstrap_zoom_phase::profile_zoom;
        candidate.layer_indices = std::move(layers);
        candidate.layer_weights.assign(candidate.layer_indices.size(),
            (sign_control ? -1.0f : 1.0f) / std::sqrt(static_cast<float>(candidate.layer_indices.size())));
        candidate.total_scale = selected_scale;
        candidate.opposite_sign_control = sign_control;
        return candidate;
    };
    candidates.push_back(append({anchor_layer}, false));
    if (local_layers.size() >= 2 && candidates.size() < config.max_extra_model_trials) {
        const auto anchor = std::find(local_layers.begin(), local_layers.end(), anchor_layer);
        if (anchor != local_layers.begin() && candidates.size() < config.max_extra_model_trials) {
            candidates.push_back(append({*(anchor - 1), anchor_layer}, false));
        }
        if (anchor + 1 != local_layers.end() && candidates.size() < config.max_extra_model_trials) {
            candidates.push_back(append({anchor_layer, *(anchor + 1)}, false));
        }
    }
    if (config.include_triplet_profile && local_layers.size() == 3 &&
            candidates.size() < config.max_extra_model_trials) {
        candidates.push_back(append(local_layers, false));
    }
    if (config.include_opposite_sign_control && candidates.size() < config.max_extra_model_trials) {
        candidates.push_back(append({anchor_layer}, true));
    }
    for (const auto & candidate : candidates) {
        if (!valid_zoom_candidate(candidate, error)) return false;
    }
    return true;
}

bool common_flydelta_select_search_continuation(
        const common_flydelta_search_pipeline_result & pipeline,
        common_flydelta_search_continuation & continuation,
        std::string & error) {
    error.clear();
    continuation = {};
    if (pipeline.directions.empty()) {
        error = "FlyDelta continuation requires a non-empty search pipeline result";
        return false;
    }
    bool found = false;
    float best_score = -std::numeric_limits<float>::infinity();
    for (size_t direction_index = 0; direction_index < pipeline.directions.size(); ++direction_index) {
        const auto & direction = pipeline.directions[direction_index];
        for (size_t trial_index = 0; trial_index < direction.region_trials.size(); ++trial_index) {
            const auto & trial = direction.region_trials[trial_index];
            if (!common_flydelta_intervention_region_trial_validate(trial, error)) return false;
            if (!trial.executed ||
                    trial.outcome == common_flydelta_counterfactual_outcome::harmed ||
                    !trial.safe_to_continue || !trial.promising ||
                    !finite(trial.search_score)) continue;
            const bool helped = trial.outcome == common_flydelta_counterfactual_outcome::helped &&
                trial.verifier_known;
            if (!found || (helped && !continuation.host_helped) ||
                    (helped == continuation.host_helped && trial.search_score > best_score)) {
                found = true;
                best_score = trial.search_score;
                continuation.direction_index = direction_index;
                continuation.region_trial_index = trial_index;
                continuation.region = trial.candidate;
                continuation.search_score = trial.search_score;
                continuation.host_helped = helped;
            }
        }
    }
    if (!found) {
        // A bounded search may complete normally without finding a safe,
        // promising continuation. The worker records that terminal result;
        // it is not a callback or persistence failure.
        if (common_flydelta_search_status_is_terminal_without_candidate(
                pipeline.search_status)) return true;
        error = "FlyDelta search pipeline produced no safe promising region continuation";
        return false;
    }
    return true;
}

bool common_flydelta_plan_search_continuation(
        const common_flydelta_search_continuation & continuation,
        const common_flydelta_evidence_depth_result & evidence_depth,
        common_flydelta_experiment_plan & plan,
        std::string & error) {
    error.clear();
    plan = {};
    if (continuation.schema_version != 1 || continuation.region.layer_indices.empty() ||
            continuation.region.anchor_layer_index == 0 || !finite(continuation.search_score) ||
            !valid_depth(evidence_depth)) {
        error = "FlyDelta continuation or evidence depth is invalid";
        return false;
    }
    plan.continuation = continuation;
    // Start every retained region at Bootstrap. Depth expresses capacity, not
    // permission to skip the rank-one/rank-two evidence chain.
    plan.depth = evidence_depth.depth;
    plan.phase = common_flydelta_experiment_phase::bootstrap;
    plan.bootstrap_refinement = common_flydelta_bootstrap_refinement_kind::bootstrap_zoom;
    plan.budget = common_flydelta_search_budget_for_depth(
        common_flydelta_search_depth::bootstrap);
    if (!common_flydelta_search_budget_validate(plan.budget, error)) return false;
    plan.tfo_lite_permitted_by_evidence = evidence_depth.depth ==
        common_flydelta_search_depth::deep;
    plan.tfo_lite_requires_utility_gate = plan.tfo_lite_permitted_by_evidence;
    return true;
}

bool common_flydelta_advance_experiment_plan(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_decision & utility,
        common_flydelta_experiment_plan & next,
        bool & advanced,
        std::string & error) {
    error.clear();
    next = current;
    advanced = false;
    if (current.continuation.region.layer_indices.empty() ||
            !valid_depth_value(current.depth) ||
            !common_flydelta_search_budget_validate(current.budget, error)) {
        if (error.empty()) error = "FlyDelta experiment plan transition input is invalid";
        return false;
    }
    const auto configure_phase = [&](common_flydelta_experiment_phase phase,
            common_flydelta_search_depth budget_depth) {
        next.phase = phase;
        next.budget = common_flydelta_search_budget_for_depth(budget_depth);
        next.required_compatible_directions = phase == common_flydelta_experiment_phase::bootstrap
            ? 1 : 2;
        next.require_decision_margin = phase != common_flydelta_experiment_phase::bootstrap;
        next.run_rank_two_controls_first = phase != common_flydelta_experiment_phase::bootstrap;
        next.run_tfo_lite = false;
        next.run_orthogonal_search = false;
    };
    switch (utility.action) {
        case common_flydelta_utility_gate_action::stop:
        case common_flydelta_utility_gate_action::retain:
            return true;
        case common_flydelta_utility_gate_action::refine_bootstrap:
            if (current.phase != common_flydelta_experiment_phase::bootstrap ||
                    current.depth != common_flydelta_search_depth::bootstrap) {
                error = "FlyDelta BootstrapZoom refinement is not permitted by the current plan";
                return false;
            }
            configure_phase(common_flydelta_experiment_phase::bootstrap,
                common_flydelta_search_depth::bootstrap);
            next.bootstrap_refinement = common_flydelta_bootstrap_refinement_kind::adaptive_alpha;
            advanced = true;
            break;
        case common_flydelta_utility_gate_action::orthogonal_search:
            if (current.phase != common_flydelta_experiment_phase::bootstrap) {
                error = "FlyDelta orthogonal search requires the Bootstrap phase";
                return false;
            }
            // This is deliberately a same-phase experimental operation. The
            // host must keep evidence depth unchanged and later evaluate the
            // resulting rank-two controls before allowing Deep/TFO.
            next.run_orthogonal_search = true;
            next.run_rank_two_controls_first = true;
            next.run_tfo_lite = false;
            advanced = true;
            break;
        case common_flydelta_utility_gate_action::escalate_shallow:
            if (current.phase != common_flydelta_experiment_phase::bootstrap ||
                    current.depth == common_flydelta_search_depth::bootstrap) {
                error = "FlyDelta Shallow escalation is not permitted by the current plan";
                return false;
            }
            configure_phase(common_flydelta_experiment_phase::shallow_controls,
                common_flydelta_search_depth::shallow);
            advanced = true;
            break;
        case common_flydelta_utility_gate_action::escalate_deep:
            if (current.phase != common_flydelta_experiment_phase::shallow_controls ||
                    current.depth != common_flydelta_search_depth::deep) {
                error = "FlyDelta Deep escalation requires successful Shallow controls";
                return false;
            }
            configure_phase(common_flydelta_experiment_phase::deep_controls,
                common_flydelta_search_depth::deep);
            advanced = true;
            break;
        case common_flydelta_utility_gate_action::allow_tfo_lite:
            if (current.phase != common_flydelta_experiment_phase::deep_controls ||
                    current.depth != common_flydelta_search_depth::deep ||
                    !current.tfo_lite_permitted_by_evidence ||
                    !current.tfo_lite_requires_utility_gate) {
                error = "FlyDelta TFO-lite requires Deep controls and evidence permission";
                return false;
            }
            next.run_tfo_lite = true;
            advanced = true;
            break;
    }
    return common_flydelta_search_budget_validate(next.budget, error);
}

bool common_flydelta_orchestrate_search_slice(
        const common_flydelta_experiment_plan & current,
        const common_flydelta_utility_gate_config & utility_config,
        const std::vector<common_flydelta_subspace_utility_observation> & observations,
        const common_flydelta_utility_history & history,
        common_flydelta_slice_orchestration_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_search_budget_validate(current.budget, error) ||
            observations.empty()) {
        if (error.empty()) error = "FlyDelta bounded-slice orchestration input is invalid";
        return false;
    }
    if (!common_flydelta_decide_subspace_utility(
            utility_config, current.depth, current.phase, observations, history,
            result.utility, error)) return false;
    if (result.utility.action != common_flydelta_utility_gate_action::stop &&
            result.utility.action != common_flydelta_utility_gate_action::retain &&
            !result.utility.utility_qualified) {
        error = "FlyDelta non-retain transition requires qualified utility";
        return false;
    }
    if (!common_flydelta_advance_experiment_plan(
            current, result.utility, result.plan, result.plan_advanced, error)) return false;
    switch (result.utility.action) {
        case common_flydelta_utility_gate_action::stop:
            result.next_action = common_flydelta_next_action::stop;
            break;
        case common_flydelta_utility_gate_action::retain:
            result.next_action = common_flydelta_next_action::retain;
            break;
        case common_flydelta_utility_gate_action::refine_bootstrap:
            result.next_action = common_flydelta_next_action::refine_bootstrap;
            break;
        case common_flydelta_utility_gate_action::orthogonal_search:
            result.next_action = common_flydelta_next_action::run_orthogonal_search;
            break;
        case common_flydelta_utility_gate_action::escalate_shallow:
            result.next_action = common_flydelta_next_action::run_shallow_controls;
            break;
        case common_flydelta_utility_gate_action::escalate_deep:
            result.next_action = common_flydelta_next_action::run_deep_controls;
            break;
        case common_flydelta_utility_gate_action::allow_tfo_lite:
            result.next_action = common_flydelta_next_action::allow_tfo_lite;
            break;
    }
    result.reason = common_flydelta_utility_gate_action_name(result.utility.action);
    return true;
}
