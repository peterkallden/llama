#include "agent/adaptation/flydelta/flydelta-representation-diagnostics.h"

#include <algorithm>
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

bool common_flydelta_representation_diagnostics_from_reductions(
        const uint32_t layer_index,
        const double dot_shift_delta,
        const double shift_squared,
        const double delta_squared,
        const double residual_squared,
        common_flydelta_representation_diagnostics & diagnostics,
        std::string & error) {
    error.clear();
    diagnostics = {};
    if (!std::isfinite(dot_shift_delta) || !std::isfinite(shift_squared) ||
            !std::isfinite(delta_squared) || !std::isfinite(residual_squared) ||
            shift_squared < 0.0 || delta_squared <= std::numeric_limits<double>::epsilon() ||
            residual_squared < -1.0e-8) {
        error = "FlyDelta compact diagnostics reductions are invalid";
        return false;
    }

    const double bounded_residual_squared = std::max(0.0, residual_squared);
    const double shift_norm = std::sqrt(shift_squared);
    const double delta_norm = std::sqrt(delta_squared);
    diagnostics.layer_index = layer_index;
    diagnostics.cosine = shift_norm > std::numeric_limits<double>::epsilon()
        ? static_cast<float>(dot_shift_delta / (shift_norm * delta_norm)) : 0.0f;
    diagnostics.progress = static_cast<float>(dot_shift_delta / delta_squared);
    diagnostics.leakage = static_cast<float>(std::sqrt(bounded_residual_squared) / delta_norm);
    diagnostics.shift_norm = static_cast<float>(shift_norm);
    return common_flydelta_representation_diagnostics_validate(diagnostics, error);
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
            baseline.position != overlay.position ||
            (baseline.position == common_flydelta_capture_position::prompt_row &&
             baseline.token_index != overlay.token_index)) {
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
    return common_flydelta_representation_diagnostics_from_vectors(
        baseline.values.data() + baseline_offset,
        overlay.values.data() + overlay_offset,
        dimension, behavior_delta, diagnostics, error);
}

bool common_flydelta_representation_diagnostics_from_vectors(
        const float * baseline_values,
        const float * overlay_values,
        const size_t dimension,
        const common_flydelta_behavior_delta & behavior_delta,
        common_flydelta_representation_diagnostics & diagnostics,
        std::string & error) {
    error.clear();
    diagnostics = {};
    if (baseline_values == nullptr || overlay_values == nullptr ||
            dimension == 0 || behavior_delta.values.size() != dimension) {
        error = "FlyDelta diagnostics vector dimensions are invalid";
        return false;
    }
    double dot = 0.0;
    double shift_squared = 0.0;
    double delta_squared = 0.0;
    std::vector<float> shift(dimension);
    for (size_t index = 0; index < dimension; ++index) {
        const float value = overlay_values[index] - baseline_values[index];
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

    const double progress = dot / delta_squared;
    double residual_squared = 0.0;
    for (size_t index = 0; index < dimension; ++index) {
        const double residual = static_cast<double>(shift[index]) -
            progress * behavior_delta.values[index];
        residual_squared += residual * residual;
    }

    return common_flydelta_representation_diagnostics_from_reductions(
        static_cast<uint32_t>(behavior_delta.layer_index), dot, shift_squared,
        delta_squared, residual_squared, diagnostics, error);
}
