#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"

#include <cmath>
#include <limits>

namespace {

bool finite(float value) {
    return std::isfinite(value);
}

bool find_layer(
        const common_flydelta_hidden_state_capture & capture,
        uint32_t layer,
        size_t & offset) {
    for (size_t index = 0; index < capture.layer_indices.size(); ++index) {
        if (capture.layer_indices[index] == layer) {
            offset = index * static_cast<size_t>(capture.n_embd);
            return true;
        }
    }
    return false;
}

} // namespace

bool common_flydelta_representation_diagnostics_validate(
        const common_flydelta_representation_diagnostics & diagnostics,
        std::string & error) {
    error.clear();
    if (diagnostics.schema_version != 1 ||
            !finite(diagnostics.cosine) || !finite(diagnostics.progress) ||
            !finite(diagnostics.leakage) || !finite(diagnostics.shift_norm) ||
            diagnostics.leakage < 0.0f || diagnostics.shift_norm < 0.0f) {
        error = "FlyDelta representation diagnostics are invalid";
        return false;
    }
    return true;
}

bool common_flydelta_representation_diagnostics_from_captures(
        const common_flydelta_hidden_state_capture & baseline,
        const common_flydelta_hidden_state_capture & overlay,
        const common_flydelta_behavior_delta & behavior_delta,
        size_t max_capture_bytes,
        common_flydelta_representation_diagnostics & diagnostics,
        std::string & error) {
    error.clear();
    diagnostics = {};
    if (!common_flydelta_hidden_state_capture_validate(baseline, max_capture_bytes, error) ||
            !common_flydelta_hidden_state_capture_validate(overlay, max_capture_bytes, error) ||
            !common_flydelta_behavior_delta_validate(
                behavior_delta, baseline.n_embd, max_capture_bytes, error)) {
        return false;
    }
    if (baseline.model_profile_fingerprint != overlay.model_profile_fingerprint ||
            baseline.capture_layout_revision != overlay.capture_layout_revision ||
            baseline.layer_indices != overlay.layer_indices ||
            baseline.n_embd != overlay.n_embd ||
            baseline.token_index != overlay.token_index) {
        error = "FlyDelta diagnostics captures are not aligned";
        return false;
    }

    size_t baseline_offset = 0;
    size_t overlay_offset = 0;
    if (!find_layer(baseline, static_cast<uint32_t>(behavior_delta.layer_index), baseline_offset) ||
            !find_layer(overlay, static_cast<uint32_t>(behavior_delta.layer_index), overlay_offset)) {
        error = "FlyDelta diagnostics layer is missing from capture";
        return false;
    }

    const size_t dimension = static_cast<size_t>(baseline.n_embd);
    double dot = 0.0;
    double shift_squared = 0.0;
    double delta_squared = 0.0;
    std::vector<float> shift(dimension);
    for (size_t index = 0; index < dimension; ++index) {
        const float value = overlay.values[overlay_offset + index] -
            baseline.values[baseline_offset + index];
        if (!finite(value) || !finite(behavior_delta.values[index])) {
            error = "FlyDelta diagnostics contain a non-finite value";
            return false;
        }
        shift[index] = value;
        dot += static_cast<double>(value) * behavior_delta.values[index];
        shift_squared += static_cast<double>(value) * value;
        delta_squared += static_cast<double>(behavior_delta.values[index]) * behavior_delta.values[index];
    }
    if (delta_squared <= std::numeric_limits<double>::epsilon()) {
        error = "FlyDelta diagnostics behavior delta must not be zero";
        return false;
    }

    const double shift_norm = std::sqrt(shift_squared);
    const double delta_norm = std::sqrt(delta_squared);
    const double progress = dot / delta_squared;
    double residual_squared = 0.0;
    for (size_t index = 0; index < dimension; ++index) {
        const double residual = static_cast<double>(shift[index]) -
            progress * behavior_delta.values[index];
        residual_squared += residual * residual;
    }

    diagnostics.layer_index = static_cast<uint32_t>(behavior_delta.layer_index);
    diagnostics.cosine = shift_norm > std::numeric_limits<double>::epsilon()
        ? static_cast<float>(dot / (shift_norm * delta_norm)) : 0.0f;
    diagnostics.progress = static_cast<float>(progress);
    diagnostics.leakage = static_cast<float>(std::sqrt(residual_squared) / delta_norm);
    diagnostics.shift_norm = static_cast<float>(shift_norm);
    return common_flydelta_representation_diagnostics_validate(diagnostics, error);
}
