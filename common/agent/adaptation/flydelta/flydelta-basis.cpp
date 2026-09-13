#include "agent/adaptation/flydelta/flydelta-basis.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
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

bool common_flydelta_repair_delta_validate(
        const common_flydelta_repair_delta & delta,
        size_t expected_dimension,
        size_t max_bytes,
        std::string & error) {
    error.clear();
    if (delta.schema_version != 1 || !nonempty_bounded(delta.id) ||
            !nonempty_bounded(delta.capture_manifest_id) ||
            !nonempty_bounded(delta.host_evidence_ref) || expected_dimension == 0 ||
            delta.values.size() != expected_dimension ||
            (max_bytes != 0 && delta.values.size() * sizeof(float) > max_bytes)) {
        error = "FlyDelta repair delta identity or bounds are invalid";
        return false;
    }
    float squared = 0.0f;
    for (const float value : delta.values) {
        if (!std::isfinite(value)) {
            error = "FlyDelta repair delta contains a non-finite value";
            return false;
        }
        squared += value * value;
    }
    if (squared <= std::numeric_limits<float>::epsilon()) {
        error = "FlyDelta repair delta must not be zero";
        return false;
    }
    return true;
}

bool common_flydelta_basis_config_validate(
        const common_flydelta_basis_config & config,
        std::string & error) {
    error.clear();
    if (config.dimension == 0 || config.dimension > (1U << 20) || config.max_directions == 0 ||
            config.max_directions > 256 || !std::isfinite(config.cluster_similarity) ||
            config.cluster_similarity < 0.0f || config.cluster_similarity > 1.0f) {
        error = "FlyDelta basis configuration is invalid";
        return false;
    }
    return true;
}

common_flydelta_basis_builder::common_flydelta_basis_builder(
        common_flydelta_basis_config config)
    : config_(config) {}

bool common_flydelta_basis_builder::add(
        const common_flydelta_repair_delta & delta,
        const common_flydelta_intervention_credit & credit,
        std::string & error) {
    error.clear();
    if (!common_flydelta_basis_config_validate(config_, error) ||
            !common_flydelta_repair_delta_validate(delta, config_.dimension, 64U * 1024U * 1024U, error) ||
            !common_flydelta_intervention_credit_validate(credit, error)) {
        return false;
    }
    if (credit.outcome == common_flydelta_counterfactual_outcome::unknown) return true;

    const float delta_norm = norm(delta.values);
    std::vector<float> normalized(delta.values.size());
    for (size_t i = 0; i < delta.values.size(); ++i) normalized[i] = delta.values[i] / delta_norm;

    size_t best_index = directions_.size();
    float best_similarity = -1.0f;
    for (size_t i = 0; i < directions_.size(); ++i) {
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
        directions_.push_back({normalized, 1, 0, 0});
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
