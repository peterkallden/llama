#include "agent/adaptation/flydelta/flydelta-basis.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

bool compatible_capture_position(
        const common_flydelta_hidden_state_capture & failed,
        const common_flydelta_hidden_state_capture & repaired) {
    if (failed.position != repaired.position) return false;
    return failed.position == common_flydelta_capture_position::generation_boundary ||
        failed.token_index == repaired.token_index;
}

float norm(const std::vector<float> & values) {
    float squared = 0.0f;
    for (const float value : values) squared += value * value;
    return std::sqrt(squared);
}

float similarity(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) return -1.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < left.size(); ++i) dot += left[i] * right[i];
    return dot;
}

} // namespace

bool common_flydelta_behavior_delta_validate(
        const common_flydelta_behavior_delta & delta,
        size_t expected_dimension,
        size_t max_bytes,
        std::string & error) {
    error.clear();
    if (delta.schema_version != 1 || !nonempty_bounded(delta.id) ||
            !nonempty_bounded(delta.behavior_key) ||
            !nonempty_bounded(delta.capture_manifest_id) ||
            !nonempty_bounded(delta.host_evidence_ref) || expected_dimension == 0 ||
            !nonempty_bounded(delta.model_profile_fingerprint) ||
            !nonempty_bounded(delta.execution_context_fingerprint) ||
            !nonempty_bounded(delta.capture_layout_revision) || delta.layer_index < 0 ||
            delta.values.size() != expected_dimension ||
            (max_bytes != 0 && delta.values.size() * sizeof(float) > max_bytes)) {
        error = "FlyDelta behavior delta identity or bounds are invalid";
        return false;
    }
    float squared = 0.0f;
    for (const float value : delta.values) {
        if (!std::isfinite(value)) {
            error = "FlyDelta behavior delta contains a non-finite value";
            return false;
        }
        squared += value * value;
    }
    if (squared <= std::numeric_limits<float>::epsilon()) {
        error = "FlyDelta behavior delta must not be zero";
        return false;
    }
    return true;
}

bool common_flydelta_behavior_deltas_from_captures(
        const common_flydelta_capture_manifest & manifest,
        const common_flydelta_hidden_state_capture & failed,
        const common_flydelta_hidden_state_capture & repaired,
        const std::string & host_evidence_ref,
        size_t max_capture_bytes,
        size_t max_delta_bytes,
        std::vector<common_flydelta_behavior_delta> & deltas,
        std::string & error) {
    error.clear();
    deltas.clear();
    if (!common_flydelta_capture_manifest_validate(
            manifest, max_capture_bytes, error) ||
            !common_flydelta_hidden_state_capture_validate(
                failed, max_capture_bytes, error) ||
            !common_flydelta_hidden_state_capture_validate(
                repaired, max_capture_bytes, error) ||
            !nonempty_bounded(host_evidence_ref)) {
        if (error.empty()) error = "FlyDelta behavior captures have invalid identity";
        return false;
    }
    if (manifest.negative_execution_ref.empty() ||
            manifest.positive_execution_ref.empty() ||
            failed.model_profile_fingerprint != manifest.model_profile_fingerprint ||
            repaired.model_profile_fingerprint != manifest.model_profile_fingerprint ||
            failed.capture_layout_revision != manifest.capture_layout_revision ||
            repaired.capture_layout_revision != manifest.capture_layout_revision ||
            failed.layer_indices != repaired.layer_indices ||
            failed.n_embd != repaired.n_embd ||
            !compatible_capture_position(failed, repaired) ||
            failed.values.size() != repaired.values.size()) {
        error = "FlyDelta baseline and candidate captures are not aligned";
        return false;
    }

    const size_t values_per_layer = failed.n_embd;
    if (values_per_layer == 0 || failed.values.size() !=
            failed.layer_indices.size() * values_per_layer) {
        error = "FlyDelta captures have an invalid layer layout";
        return false;
    }
    if (failed.values.size() > (std::numeric_limits<size_t>::max() -
            repaired.values.size()) / sizeof(float) ||
            manifest.captured_bytes != (failed.values.size() + repaired.values.size()) * sizeof(float)) {
        error = "FlyDelta capture manifest byte count does not match captures";
        return false;
    }

    deltas.reserve(failed.layer_indices.size());
    for (size_t layer = 0; layer < failed.layer_indices.size(); ++layer) {
        common_flydelta_behavior_delta delta;
        delta.id = manifest.id + "/behavior-delta/layer-" +
            std::to_string(failed.layer_indices[layer]);
        delta.source = manifest.source;
        delta.behavior_key = manifest.behavior_key;
        delta.capture_manifest_id = manifest.id;
        delta.host_evidence_ref = host_evidence_ref;
        delta.model_profile_fingerprint = manifest.model_profile_fingerprint;
        delta.execution_context_fingerprint = manifest.execution_context_fingerprint;
        delta.capture_layout_revision = manifest.capture_layout_revision;
        delta.layer_index = static_cast<int32_t>(failed.layer_indices[layer]);
        delta.values.resize(values_per_layer);
        const size_t offset = layer * values_per_layer;
        for (size_t i = 0; i < values_per_layer; ++i) {
            delta.values[i] = repaired.values[offset + i] - failed.values[offset + i];
        }
        if (!common_flydelta_behavior_delta_validate(
                delta, values_per_layer, max_delta_bytes, error)) {
            deltas.clear();
            return false;
        }
        deltas.push_back(std::move(delta));
    }
    return true;
}

bool common_flydelta_basis_config_validate(
        const common_flydelta_basis_config & config,
        std::string & error) {
    error.clear();
    if (config.dimension == 0 || config.dimension > (1U << 20) || config.max_directions == 0 ||
            config.max_directions > 256 || !std::isfinite(config.cluster_similarity) ||
            config.cluster_similarity < 0.0f || config.cluster_similarity > 1.0f ||
            !nonempty_bounded(config.behavior_key) ||
            !nonempty_bounded(config.model_profile_fingerprint) ||
            !nonempty_bounded(config.execution_context_fingerprint) ||
            !nonempty_bounded(config.capture_layout_revision)) {
        error = "FlyDelta basis configuration is invalid";
        return false;
    }
    return true;
}

common_flydelta_basis_builder::common_flydelta_basis_builder(
        common_flydelta_basis_config config)
    : config_(config) {}

bool common_flydelta_basis_builder::add(
        const common_flydelta_behavior_delta & delta,
        const common_flydelta_intervention_credit & credit,
        std::string & error) {
    error.clear();
    if (!common_flydelta_basis_config_validate(config_, error) ||
            !common_flydelta_behavior_delta_validate(delta, config_.dimension, 64U * 1024U * 1024U, error) ||
            !common_flydelta_intervention_credit_validate(credit, error)) {
        return false;
    }
    if (delta.source != config_.source || delta.behavior_key != config_.behavior_key ||
            delta.model_profile_fingerprint != config_.model_profile_fingerprint ||
            delta.execution_context_fingerprint != config_.execution_context_fingerprint ||
            delta.capture_layout_revision != config_.capture_layout_revision) {
        error = "FlyDelta behavior delta is incompatible with basis configuration";
        return false;
    }
    if (credit.outcome == common_flydelta_counterfactual_outcome::unknown) return true;

    const float delta_norm = norm(delta.values);
    std::vector<float> normalized(delta.values.size());
    for (size_t i = 0; i < delta.values.size(); ++i) normalized[i] = delta.values[i] / delta_norm;

    size_t best_index = directions_.size();
    float best_similarity = -1.0f;
    for (size_t i = 0; i < directions_.size(); ++i) {
        if (directions_[i].layer_index != delta.layer_index) continue;
        const float candidate_similarity = similarity(normalized, directions_[i].values);
        if (candidate_similarity > best_similarity) {
            best_similarity = candidate_similarity;
            best_index = i;
        }
    }

    if (credit.outcome == common_flydelta_counterfactual_outcome::helped &&
            (best_index == directions_.size() || best_similarity < config_.cluster_similarity)) {
        if (directions_.size() >= config_.max_directions) {
            error = "FlyDelta basis direction bound is exhausted";
            return false;
        }
        directions_.push_back({delta.layer_index, normalized, 1, 0, 0});
        return true;
    }
    if (best_index == directions_.size() || best_similarity < config_.cluster_similarity) return true;

    auto & direction = directions_[best_index];
    if (credit.outcome == common_flydelta_counterfactual_outcome::helped) {
        const float count = static_cast<float>(direction.helped_observations);
        for (size_t i = 0; i < direction.values.size(); ++i) {
            direction.values[i] = (direction.values[i] * count + normalized[i]) / (count + 1.0f);
        }
        const float updated_norm = norm(direction.values);
        for (float & value : direction.values) value /= updated_norm;
        ++direction.helped_observations;
    } else if (credit.outcome == common_flydelta_counterfactual_outcome::harmed) {
        ++direction.harmed_observations;
    } else {
        ++direction.neutral_observations;
    }
    return true;
}
