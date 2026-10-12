#pragma once

#include <cstddef>
#include <optional>
#include <string>

// DoseController is a low-level safety/calibration primitive. It does not
// choose a direction, layer, coefficient, utility objective or host outcome.
enum class common_flydelta_dose_action {
    accept,
    retry_lower,
    safety_boundary,
    reject,
};

const char * common_flydelta_dose_action_name(common_flydelta_dose_action action);

struct common_flydelta_dose_policy {
    int schema_version = 1;
    float max_shift_norm = 1.0f;
    float max_leakage = 1.0f;
    float target_relative_dose = 0.75f;
    float target_absolute_fraction = 0.60f;
    float min_strength = 0.000001f;
    float max_strength = 1.0f;
    float backoff_safety_factor = 0.80f;
};

struct common_flydelta_dose_state {
    int schema_version = 1;
    size_t observation_count = 0;
    // Monotone trust-region bounds. The legacy last_* fields remain for
    // persisted V0 state and diagnostics; decisions use these bounds first.
    std::optional<float> max_safe_strength;
    std::optional<float> min_unsafe_strength;
    std::optional<float> max_safe_relative_dose;
    std::optional<float> min_unsafe_relative_dose;
    std::optional<float> max_safe_absolute_dose;
    std::optional<float> min_unsafe_absolute_dose;
    std::optional<float> last_safe_strength;
    std::optional<float> last_unsafe_strength;
    std::optional<float> last_safe_relative_dose;
    std::optional<float> last_unsafe_relative_dose;
    std::optional<float> last_safe_absolute_dose;
    std::optional<float> last_unsafe_absolute_dose;
    float local_relative_gain = 0.0f;
};

struct common_flydelta_dose_observation {
    bool geometry_available = false;
    float requested_strength = 0.0f;
    float absolute_dose = 0.0f;
    float progress = 0.0f;
    float leakage = 0.0f;
};

struct common_flydelta_dose_decision {
    int schema_version = 1;
    common_flydelta_dose_action action = common_flydelta_dose_action::reject;
    float requested_strength = 0.0f;
    std::optional<float> proposed_safe_strength;
    std::optional<float> known_safe_upper_bound;
    float absolute_dose = 0.0f;
    float relative_dose = 0.0f;
    bool comparable = false;
    bool safety_limited = false;
    std::string reason;
};

bool common_flydelta_dose_policy_validate(
        const common_flydelta_dose_policy & policy,
        std::string & error);

bool common_flydelta_dose_observation_validate(
        const common_flydelta_dose_observation & observation,
        std::string & error);

float common_flydelta_relative_dose(float progress, float leakage);

bool common_flydelta_dose_preflight(
        const common_flydelta_dose_policy & policy,
        const common_flydelta_dose_state & state,
        float requested_strength,
        common_flydelta_dose_decision & decision,
        std::string & error);

bool common_flydelta_dose_observe(
        const common_flydelta_dose_policy & policy,
        common_flydelta_dose_state & state,
        const common_flydelta_dose_observation & observation,
        common_flydelta_dose_decision & decision,
        std::string & error);
