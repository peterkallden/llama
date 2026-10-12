#include "agent/adaptation/flydelta/flydelta-dose-controller.h"

#include <cmath>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_flydelta_dose_policy policy;
    policy.max_shift_norm = 10.0f;
    policy.max_leakage = 1.0f;
    policy.target_relative_dose = 0.75f;
    policy.target_absolute_fraction = 0.60f;
    policy.min_strength = 0.0001f;
    policy.max_strength = 1.0f;

    CHECK(common_flydelta_dose_policy_validate(policy, error));
    CHECK(std::fabs(common_flydelta_relative_dose(0.6f, 0.8f) - 1.0f) < 0.00001f);

    common_flydelta_dose_state state;
    common_flydelta_dose_decision decision;
    CHECK(common_flydelta_dose_preflight(policy, state, 0.02f, decision, error));
    CHECK(decision.action == common_flydelta_dose_action::accept);

    CHECK(common_flydelta_dose_observe(
        policy, state,
        common_flydelta_dose_observation{true, 0.02f, 4.0f, 0.8f, 0.6f},
        decision, error));
    CHECK(decision.action == common_flydelta_dose_action::accept);
    CHECK(decision.comparable);
    CHECK(std::fabs(decision.relative_dose - 1.0f) < 0.00001f);
    CHECK(state.last_safe_strength && *state.last_safe_strength == 0.02f);
    CHECK(state.max_safe_strength && *state.max_safe_strength == 0.02f);

    CHECK(common_flydelta_dose_observe(
        policy, state,
        common_flydelta_dose_observation{true, 0.04f, 20.0f, 0.8f, 0.6f},
        decision, error));
    CHECK(decision.action == common_flydelta_dose_action::retry_lower);
    CHECK(decision.proposed_safe_strength &&
        *decision.proposed_safe_strength < 0.04f &&
        *decision.proposed_safe_strength > policy.min_strength);
    CHECK(decision.safety_limited);
    CHECK(state.min_unsafe_strength && *state.min_unsafe_strength == 0.04f);

    CHECK(common_flydelta_dose_preflight(policy, state, 0.04f, decision, error));
    CHECK(decision.action == common_flydelta_dose_action::safety_boundary);
    CHECK(decision.known_safe_upper_bound && *decision.known_safe_upper_bound == 0.02f);

    // Bounds are monotone even when observations arrive out of order. This is
    // the trust-region contract required by golden-section/TFO proposals.
    CHECK(common_flydelta_dose_observe(
        policy, state,
        common_flydelta_dose_observation{true, 0.01f, 2.0f, 0.2f, 0.1f},
        decision, error));
    CHECK(state.max_safe_strength && *state.max_safe_strength == 0.02f);
    CHECK(common_flydelta_dose_observe(
        policy, state,
        common_flydelta_dose_observation{true, 0.03f, 20.0f, 0.8f, 0.6f},
        decision, error));
    CHECK(state.min_unsafe_strength && *state.min_unsafe_strength == 0.03f);
    CHECK(common_flydelta_dose_preflight(policy, state, 0.035f, decision, error));
    CHECK(decision.action == common_flydelta_dose_action::safety_boundary);
    CHECK(decision.known_safe_upper_bound && *decision.known_safe_upper_bound == 0.02f);

    CHECK(common_flydelta_dose_observe(
        policy, state,
        common_flydelta_dose_observation{false, 0.01f, 0.0f, 0.0f, 0.0f},
        decision, error));
    CHECK(decision.action == common_flydelta_dose_action::reject);
    CHECK(!decision.comparable);
    return 0;
}
