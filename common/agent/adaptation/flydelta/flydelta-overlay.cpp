#include "agent/adaptation/flydelta/flydelta-overlay.h"

#include <cmath>
#include <limits>

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

} // namespace

bool common_flydelta_compose_static_overlay(
        const std::string & artifact_id,
        size_t model_n_embd,
        size_t model_n_layers,
        int32_t il_start,
        int32_t il_end,
        const std::vector<common_flydelta_basis_direction> & directions,
        const std::vector<float> & coefficients,
        const common_flydelta_gate_decision & gate,
        size_t max_bytes,
        common_flydelta_static_overlay & overlay,
        std::string & error) {
    error.clear();
    overlay = {};
    if (!gate.apply) return true;
    if (!nonempty_bounded(artifact_id) || model_n_embd == 0 ||
            model_n_embd > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            model_n_layers < 2 ||
            il_start < 1 || il_end < il_start || static_cast<size_t>(il_end) >= model_n_layers ||
            !std::isfinite(gate.scale) || gate.scale <= 0.0f || gate.scale > 1.0f ||
            directions.empty() || directions.size() != coefficients.size()) {
        error = "FlyDelta static overlay composition bounds are invalid";
        return false;
    }
    if (model_n_embd > std::numeric_limits<size_t>::max() / (model_n_layers - 1)) {
        error = "FlyDelta static overlay dimensions overflow";
        return false;
    }
    const size_t value_count = model_n_embd * (model_n_layers - 1);
    if (max_bytes != 0 && value_count > max_bytes / sizeof(float)) {
        error = "FlyDelta composed overlay exceeds its byte bound";
        return false;
    }
    overlay.enabled = true;
    overlay.artifact_id = artifact_id;
    overlay.n_embd = static_cast<int32_t>(model_n_embd);
    overlay.il_start = il_start;
    overlay.il_end = il_end;
    overlay.scale = 1.0f;
    overlay.data.assign(value_count, 0.0f);
    for (size_t i = 0; i < directions.size(); ++i) {
        const auto & direction = directions[i];
        const float coefficient = coefficients[i];
        if (direction.layer_index < il_start || direction.layer_index > il_end ||
                direction.values.size() != model_n_embd || !std::isfinite(coefficient) ||
                coefficient < -1.0f || coefficient > 1.0f) {
            error = "FlyDelta basis direction is incompatible with overlay";
            overlay = {};
            return false;
        }
        const size_t offset = model_n_embd * (static_cast<size_t>(direction.layer_index) - 1);
        for (size_t value = 0; value < model_n_embd; ++value) {
            if (!std::isfinite(direction.values[value])) {
                error = "FlyDelta basis direction contains a non-finite value";
                overlay = {};
                return false;
            }
            overlay.data[offset + value] += direction.values[value] * coefficient * gate.scale;
            if (!std::isfinite(overlay.data[offset + value])) {
                error = "FlyDelta composed overlay contains a non-finite value";
                overlay = {};
                return false;
            }
        }
    }
    return common_flydelta_static_overlay_validate(
        overlay, model_n_embd, model_n_layers, max_bytes, error);
}
