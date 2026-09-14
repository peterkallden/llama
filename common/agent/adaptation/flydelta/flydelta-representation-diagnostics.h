#pragma once

#include "agent/adaptation/flydelta/flydelta-basis.h"

#include <cstddef>
#include <cstdint>
#include <string>

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
