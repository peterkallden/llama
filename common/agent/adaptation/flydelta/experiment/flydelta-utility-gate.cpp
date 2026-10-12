#include "agent/adaptation/flydelta/flydelta-experiment-orchestration.h"

#include <cmath>

namespace {
bool finite_value(float value) { return std::isfinite(value); }

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
}

bool common_flydelta_utility_gate_config_validate(
        const common_flydelta_utility_gate_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || !finite_value(config.shallow_enter_margin) ||
            !finite_value(config.deep_enter_margin) || !finite_value(config.tfo_enter_margin) ||
            config.shallow_enter_observations == 0 || config.deep_enter_observations == 0 ||
            config.tfo_enter_observations == 0 || config.exit_nonqualifying_observations == 0 ||
            !finite_value(config.min_cosine) || config.min_cosine < -1.0f || config.min_cosine > 1.0f ||
            !finite_value(config.max_leakage) || config.max_leakage < 0.0f ||
            !finite_value(config.max_shift_norm) || config.max_shift_norm <= 0.0f) {
        error = "FlyDelta utility gate configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_subspace_utility_observation_validate(
        const common_flydelta_subspace_utility_observation & observation,
        std::string & error) {
    error.clear();
    if (!finite_value(observation.decision_margin_delta) ||
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
            !(max_allowed_depth == common_flydelta_search_depth::bootstrap ||
              max_allowed_depth == common_flydelta_search_depth::shallow ||
              max_allowed_depth == common_flydelta_search_depth::deep) ||
            observations.empty()) {
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
            observation.alpha_response_status == common_flydelta_alpha_response_status::helped;
        if (observation.decision_margin_available && geometry_ok && alpha_can_continue &&
                observation.decision_margin_delta > required_margin(current_phase, config)) {
            qualified = true;
        }
    }
    decision.utility_qualified = qualified;
    decision.history = qualified
        ? common_flydelta_utility_history{history.qualifying_streak + 1, 0}
        : common_flydelta_utility_history{0, history.nonqualifying_streak + 1};
    if (!qualified) {
        decision.action = decision.history.nonqualifying_streak >= config.exit_nonqualifying_observations
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
