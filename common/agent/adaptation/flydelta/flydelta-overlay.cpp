#include "agent/adaptation/flydelta/flydelta-overlay.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

} // namespace

bool common_flydelta_sparse_overlay_validate(
        const common_flydelta_sparse_overlay & overlay,
        const size_t model_n_embd,
        const size_t model_n_layers,
        const size_t max_bytes,
        std::string & error) {
    error.clear();
    if (!overlay.enabled) return true;
    if (overlay.artifact_id.empty() || overlay.n_embd <= 0 || overlay.il_start < 1 ||
            overlay.il_end < overlay.il_start || model_n_embd == 0 || model_n_layers < 2 ||
            static_cast<size_t>(overlay.n_embd) != model_n_embd ||
            static_cast<size_t>(overlay.il_end) >= model_n_layers ||
            overlay.layer_indices.empty() ||
            overlay.data.size() != overlay.layer_indices.size() * model_n_embd) {
        error = "FlyDelta sparse overlay identity or dimensions are invalid";
        return false;
    }
    if (overlay.layer_indices.size() >
            std::numeric_limits<size_t>::max() / model_n_embd) {
        error = "FlyDelta sparse overlay dimensions overflow";
        return false;
    }
    if (max_bytes != 0 && overlay.data.size() > max_bytes / sizeof(float)) {
        error = "FlyDelta sparse overlay exceeds its byte bound";
        return false;
    }
    uint32_t previous = 0;
    bool first = true;
    for (const uint32_t layer : overlay.layer_indices) {
        if (layer < static_cast<uint32_t>(overlay.il_start) ||
                layer > static_cast<uint32_t>(overlay.il_end) ||
                (!first && layer <= previous)) {
            error = "FlyDelta sparse overlay layer indices are invalid";
            return false;
        }
        previous = layer;
        first = false;
    }
    for (const float value : overlay.data) {
        if (!std::isfinite(value)) {
            error = "FlyDelta sparse overlay contains a non-finite value";
            return false;
        }
    }
    return true;
}

bool common_flydelta_sparse_overlay_batch_validate(
        const common_flydelta_sparse_overlay_batch & batch,
        const size_t model_n_embd,
        const size_t model_n_layers,
        const size_t max_bytes_per_overlay,
        std::string & error) {
    error.clear();
    if (batch.schema_version != 1 || batch.overlays.empty() || batch.overlays.size() > 256) {
        error = "FlyDelta sparse overlay batch is invalid";
        return false;
    }
    std::unordered_set<std::string> identities;
    for (const auto & overlay : batch.overlays) {
        if (!common_flydelta_sparse_overlay_validate(
                overlay, model_n_embd, model_n_layers, max_bytes_per_overlay, error)) {
            return false;
        }
        if (overlay.enabled && !identities.insert(overlay.artifact_id).second) {
            error = "FlyDelta sparse overlay batch reuses an artifact identity";
            return false;
        }
    }
    return true;
}

bool common_flydelta_expand_sparse_overlay_batch(
        const common_flydelta_sparse_overlay_batch & batch,
        const size_t model_n_embd,
        const size_t model_n_layers,
        const size_t max_bytes_per_overlay,
        std::vector<common_flydelta_static_overlay> & dense,
        std::string & error) {
    error.clear();
    dense.clear();
    if (!common_flydelta_sparse_overlay_batch_validate(
            batch, model_n_embd, model_n_layers, max_bytes_per_overlay, error)) {
        return false;
    }
    dense.reserve(batch.overlays.size());
    for (const auto & sparse : batch.overlays) {
        common_flydelta_static_overlay expanded;
        if (!common_flydelta_expand_sparse_overlay(
                sparse, model_n_embd, model_n_layers, max_bytes_per_overlay,
                expanded, error)) {
            dense.clear();
            return false;
        }
        dense.push_back(std::move(expanded));
    }
    return true;
}

bool common_flydelta_compose_sparse_overlay(
        const std::string & artifact_id,
        const size_t model_n_embd,
        const size_t model_n_layers,
        const int32_t il_start,
        const int32_t il_end,
        const std::vector<common_flydelta_basis_direction> & directions,
        const std::vector<float> & coefficients,
        const common_flydelta_gate_decision & gate,
        const size_t max_bytes,
        common_flydelta_sparse_overlay & overlay,
        std::string & error) {
    error.clear();
    overlay = {};
    if (!gate.apply) return true;
    if (artifact_id.empty() || model_n_embd == 0 || model_n_layers < 2 ||
            il_start < 1 || il_end < il_start ||
            static_cast<size_t>(il_end) >= model_n_layers ||
            !std::isfinite(gate.scale) || gate.scale <= 0.0f || gate.scale > 1.0f ||
            directions.empty() || directions.size() != coefficients.size()) {
        error = "FlyDelta sparse overlay composition bounds are invalid";
        return false;
    }
    if (model_n_embd > std::numeric_limits<size_t>::max() / directions.size()) {
        error = "FlyDelta sparse overlay dimensions overflow";
        return false;
    }

    std::vector<uint32_t> layers;
    layers.reserve(directions.size());
    for (size_t index = 0; index < directions.size(); ++index) {
        const auto & direction = directions[index];
        const float coefficient = coefficients[index];
        if (direction.layer_index < il_start || direction.layer_index > il_end ||
                direction.values.size() != model_n_embd ||
                !std::isfinite(coefficient) || coefficient < -1.0f || coefficient > 1.0f) {
            error = "FlyDelta sparse basis direction is incompatible with overlay";
            return false;
        }
        layers.push_back(static_cast<uint32_t>(direction.layer_index));
    }
    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
    if (layers.empty() || layers.size() > std::numeric_limits<size_t>::max() / model_n_embd ||
            (max_bytes != 0 && layers.size() * model_n_embd > max_bytes / sizeof(float))) {
        error = "FlyDelta sparse overlay exceeds its byte bound";
        return false;
    }

    overlay.enabled = true;
    overlay.artifact_id = artifact_id;
    overlay.n_embd = static_cast<int32_t>(model_n_embd);
    overlay.il_start = il_start;
    overlay.il_end = il_end;
    overlay.layer_indices = layers;
    overlay.data.assign(layers.size() * model_n_embd, 0.0f);

    for (size_t index = 0; index < directions.size(); ++index) {
        const auto & direction = directions[index];
        const float coefficient = coefficients[index] * gate.scale;
        const auto layer_it = std::lower_bound(
            overlay.layer_indices.begin(), overlay.layer_indices.end(),
            static_cast<uint32_t>(direction.layer_index));
        const size_t layer_offset = static_cast<size_t>(layer_it - overlay.layer_indices.begin()) * model_n_embd;
        for (size_t value = 0; value < model_n_embd; ++value) {
            if (!std::isfinite(direction.values[value])) {
                error = "FlyDelta sparse basis direction contains a non-finite value";
                overlay = {};
                return false;
            }
            overlay.data[layer_offset + value] += direction.values[value] * coefficient;
            if (!std::isfinite(overlay.data[layer_offset + value])) {
                error = "FlyDelta sparse overlay contains a non-finite value";
                overlay = {};
                return false;
            }
        }
    }
    return common_flydelta_sparse_overlay_validate(
        overlay, model_n_embd, model_n_layers, max_bytes, error);
}

bool common_flydelta_expand_sparse_overlay(
        const common_flydelta_sparse_overlay & sparse,
        const size_t model_n_embd,
        const size_t model_n_layers,
        const size_t max_bytes,
        common_flydelta_static_overlay & dense,
        std::string & error) {
    error.clear();
    dense = {};
    if (!common_flydelta_sparse_overlay_validate(
            sparse, model_n_embd, model_n_layers, max_bytes, error)) {
        return false;
    }
    if (!sparse.enabled) return true;
    if (model_n_embd > std::numeric_limits<size_t>::max() / (model_n_layers - 1)) {
        error = "FlyDelta dense overlay dimensions overflow";
        return false;
    }
    const size_t dense_values = model_n_embd * (model_n_layers - 1);
    if (max_bytes != 0 && dense_values > max_bytes / sizeof(float)) {
        error = "FlyDelta dense overlay exceeds its byte bound";
        return false;
    }
    dense.enabled = true;
    dense.artifact_id = sparse.artifact_id;
    dense.n_embd = sparse.n_embd;
    dense.il_start = sparse.il_start;
    dense.il_end = sparse.il_end;
    dense.scale = 1.0f;
    dense.data.assign(dense_values, 0.0f);
    for (size_t index = 0; index < sparse.layer_indices.size(); ++index) {
        const size_t dense_offset = model_n_embd *
            (static_cast<size_t>(sparse.layer_indices[index]) - 1);
        const size_t sparse_offset = index * model_n_embd;
        std::copy_n(sparse.data.begin() + sparse_offset, model_n_embd,
            dense.data.begin() + dense_offset);
    }
    return common_flydelta_static_overlay_validate(
        dense, model_n_embd, model_n_layers, max_bytes, error);
}

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
