#include "agent/adaptation/flydelta/flydelta-dose-controller.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool finite(float value) {
    return std::isfinite(value);
}

bool valid_strength(const common_flydelta_dose_policy & policy, float value) {
    return finite(value) && value >= policy.min_strength && value <= policy.max_strength;
}

void reset_decision(
        common_flydelta_dose_decision & decision,
        float requested_strength) {
    decision = {};
    decision.requested_strength = requested_strength;
}

float proposed_backoff(
        const common_flydelta_dose_policy & policy,
        const common_flydelta_dose_observation & observation,
        float relative_dose) {
    float factor = policy.backoff_safety_factor;
    if (relative_dose > std::numeric_limits<float>::epsilon()) {
        factor = std::min(factor, policy.target_relative_dose / relative_dose);
    }
    if (observation.absolute_dose > std::numeric_limits<float>::epsilon()) {
        const float target_absolute = policy.max_shift_norm * policy.target_absolute_fraction;
        factor = std::min(factor, target_absolute / observation.absolute_dose);
    }
    return observation.requested_strength * factor;
}

} // namespace

const char * common_flydelta_dose_action_name(common_flydelta_dose_action action) {
    switch (action) {
        case common_flydelta_dose_action::accept: return "accept";
        case common_flydelta_dose_action::retry_lower: return "retry_lower";
        case common_flydelta_dose_action::safety_boundary: return "safety_boundary";
        case common_flydelta_dose_action::reject: return "reject";
    }
    return "reject";
}

bool common_flydelta_dose_policy_validate(
        const common_flydelta_dose_policy & policy,
        std::string & error) {
    error.clear();
    if (policy.schema_version != 1 || !finite(policy.max_shift_norm) ||
            policy.max_shift_norm <= 0.0f || !finite(policy.max_leakage) ||
            policy.max_leakage < 0.0f || !finite(policy.target_relative_dose) ||
            policy.target_relative_dose <= 0.0f || !finite(policy.target_absolute_fraction) ||
            policy.target_absolute_fraction <= 0.0f || policy.target_absolute_fraction > 1.0f ||
            !finite(policy.min_strength) || !finite(policy.max_strength) ||
            policy.min_strength <= 0.0f || policy.max_strength < policy.min_strength ||
            !finite(policy.backoff_safety_factor) || policy.backoff_safety_factor <= 0.0f ||
            policy.backoff_safety_factor >= 1.0f) {
        error = "FlyDelta dose policy is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_dose_observation_validate(
        const common_flydelta_dose_observation & observation,
        std::string & error) {
    error.clear();
    if (!finite(observation.requested_strength) || observation.requested_strength < 0.0f ||
            !finite(observation.absolute_dose) || observation.absolute_dose < 0.0f ||
            !finite(observation.progress) || !finite(observation.leakage) ||
            observation.leakage < 0.0f) {
        error = "FlyDelta dose observation is invalid";
        return false;
    }
    return true;
}

float common_flydelta_relative_dose(float progress, float leakage) {
    if (!finite(progress) || !finite(leakage) || leakage < 0.0f) return 0.0f;
    return std::sqrt(progress * progress + leakage * leakage);
}

bool common_flydelta_dose_preflight(
        const common_flydelta_dose_policy & policy,
        const common_flydelta_dose_state & state,
        float requested_strength,
        common_flydelta_dose_decision & decision,
        std::string & error) {
    error.clear();
    reset_decision(decision, requested_strength);
    if (!common_flydelta_dose_policy_validate(policy, error) ||
            !valid_strength(policy, requested_strength)) {
        if (error.empty()) error = "FlyDelta dose preflight strength is invalid";
        decision.action = common_flydelta_dose_action::reject;
        decision.reason = error;
        return false;
    }

    const auto & unsafe_strength = state.min_unsafe_strength
        ? state.min_unsafe_strength : state.last_unsafe_strength;
    const auto & safe_strength = state.max_safe_strength
        ? state.max_safe_strength : state.last_safe_strength;
    if (unsafe_strength && requested_strength >= *unsafe_strength && safe_strength) {
        decision.action = common_flydelta_dose_action::safety_boundary;
        decision.known_safe_upper_bound = safe_strength;
        decision.safety_limited = true;
        decision.reason = "requested strength reaches a learned unsafe boundary";
        return true;
    }

    decision.action = common_flydelta_dose_action::accept;
    decision.reason = "no learned safety boundary blocks the requested strength";
    return true;
}

bool common_flydelta_dose_observe(
        const common_flydelta_dose_policy & policy,
        common_flydelta_dose_state & state,
        const common_flydelta_dose_observation & observation,
        common_flydelta_dose_decision & decision,
        std::string & error) {
    error.clear();
    reset_decision(decision, observation.requested_strength);
    if (!common_flydelta_dose_policy_validate(policy, error) ||
            !common_flydelta_dose_observation_validate(observation, error) ||
            !valid_strength(policy, observation.requested_strength)) {
        if (error.empty()) error = "FlyDelta dose observation strength is invalid";
        decision.action = common_flydelta_dose_action::reject;
        decision.reason = error;
        return false;
    }

    ++state.observation_count;
    decision.absolute_dose = observation.absolute_dose;
    decision.relative_dose = common_flydelta_relative_dose(
        observation.progress, observation.leakage);
    decision.comparable = observation.geometry_available && finite(decision.relative_dose);
    if (!observation.geometry_available) {
        decision.action = common_flydelta_dose_action::reject;
        decision.reason = "dose geometry is unavailable";
        return true;
    }

    const bool safe = observation.absolute_dose <= policy.max_shift_norm &&
        observation.leakage <= policy.max_leakage;
    if (safe) {
        if (!state.max_safe_strength ||
                observation.requested_strength > *state.max_safe_strength) {
            state.max_safe_strength = observation.requested_strength;
            state.max_safe_relative_dose = decision.relative_dose;
            state.max_safe_absolute_dose = observation.absolute_dose;
        }
        state.last_safe_strength = observation.requested_strength;
        state.last_safe_relative_dose = decision.relative_dose;
        state.last_safe_absolute_dose = observation.absolute_dose;
        state.local_relative_gain = observation.requested_strength > 0.0f
            ? decision.relative_dose / observation.requested_strength : 0.0f;
        decision.action = common_flydelta_dose_action::accept;
        decision.reason = "observed dose is inside the safety envelope";
        return true;
    }

    if (!state.min_unsafe_strength ||
            observation.requested_strength < *state.min_unsafe_strength) {
        state.min_unsafe_strength = observation.requested_strength;
        state.min_unsafe_relative_dose = decision.relative_dose;
        state.min_unsafe_absolute_dose = observation.absolute_dose;
    }
    state.last_unsafe_strength = observation.requested_strength;
    state.last_unsafe_relative_dose = decision.relative_dose;
    state.last_unsafe_absolute_dose = observation.absolute_dose;
    decision.safety_limited = true;

    const float proposal = proposed_backoff(policy, observation, decision.relative_dose);
    if (finite(proposal) && proposal >= policy.min_strength &&
            proposal < observation.requested_strength) {
        decision.action = common_flydelta_dose_action::retry_lower;
        decision.proposed_safe_strength = proposal;
        decision.reason = "observed dose exceeded the safety envelope; retry lower";
        return true;
    }

    decision.action = common_flydelta_dose_action::safety_boundary;
    decision.reason = "observed dose exceeded the safety envelope and no bounded retry is available";
    return true;
}
