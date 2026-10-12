#pragma once

#include "agent/adaptation/flydelta/flydelta-basis.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Ephemeral diagnostics for an overlay arm that could not be host-classified.
// These values describe representation geometry only; they are not learning
// evidence, a promotion signal or an activation request.
struct common_flydelta_representation_diagnostics {
    int schema_version = 1;
    uint32_t layer_index = 0;
    float cosine = 0.0f;
    float progress = 0.0f;
    float leakage = 0.0f;
    float shift_norm = 0.0f;
};

bool common_flydelta_representation_diagnostics_validate(
        const common_flydelta_representation_diagnostics & diagnostics,
        std::string & error);

// Compares one overlay arm with the no-overlay baseline against a host-
// certified behavior delta. Captures must have matching model/layout/token
// identity and the behavior delta's layer must be present in both captures.
// A zero arm shift is valid and produces zero diagnostics. This helper never
// changes counterfactual outcome, basis, DeltaMemory or promotion state.
bool common_flydelta_representation_diagnostics_from_captures(
        const common_flydelta_hidden_state_capture & baseline,
        const common_flydelta_hidden_state_capture & overlay,
        const common_flydelta_behavior_delta & behavior_delta,
        size_t max_capture_bytes,
        common_flydelta_representation_diagnostics & diagnostics,
        std::string & error);

// Reference implementation for a compact device reduction. A backend may
// calculate the same four scalars on device and return only those scalars;
// this helper is retained as the CPU parity oracle and for scalar fallback.
// The input vectors are one aligned layer, not an entire capture.
bool common_flydelta_representation_diagnostics_from_vectors(
        const float * baseline_values,
        const float * overlay_values,
        size_t dimension,
        const common_flydelta_behavior_delta & behavior_delta,
        common_flydelta_representation_diagnostics & diagnostics,
        std::string & error);

// Completes the compact reduction contract used by device backends. The
// backend computes these scalar sums while the aligned vectors are resident
// on the device; only the reductions cross the model boundary. The formula is
// identical to the vector reference implementation below.
bool common_flydelta_representation_diagnostics_from_reductions(
        uint32_t layer_index,
        double dot_shift_delta,
        double shift_squared,
        double delta_squared,
        double residual_squared,
        common_flydelta_representation_diagnostics & diagnostics,
        std::string & error);

inline bool common_flydelta_representation_diagnostics_from_vectors(
        const std::vector<float> & baseline_values,
        const std::vector<float> & overlay_values,
        const common_flydelta_behavior_delta & behavior_delta,
        common_flydelta_representation_diagnostics & diagnostics,
        std::string & error) {
    if (baseline_values.size() != overlay_values.size()) {
        error = "FlyDelta diagnostics vector sizes are not aligned";
        return false;
    }
    return common_flydelta_representation_diagnostics_from_vectors(
        baseline_values.data(), overlay_values.data(), baseline_values.size(),
        behavior_delta, diagnostics, error);
}
