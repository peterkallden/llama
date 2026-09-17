#include "agent/adaptation/flydelta/flydelta-aggregation.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

float norm(const std::vector<float> & values) {
    float squared = 0.0f;
    for (const float value : values) squared += value * value;
    return std::sqrt(squared);
}

bool compatible(
        const common_flydelta_behavior_delta & delta,
        const common_flydelta_aggregation_config & config) {
    const auto matches_if_set = [](const std::string & expected,
            const std::string & actual) {
        return expected.empty() || expected == actual;
    };
    const auto & identity = config.identity;
    return delta.source == identity.source && delta.behavior_key == identity.behavior_key &&
        delta.model_profile_fingerprint == identity.model_profile_fingerprint &&
        delta.execution_context_fingerprint == identity.execution_context_fingerprint &&
        delta.capture_layout_revision == identity.capture_layout_revision &&
        delta.layer_index == identity.layer_index &&
        matches_if_set(config.scope_fingerprint, delta.scope_fingerprint) &&
        matches_if_set(config.tokenizer_fingerprint, delta.tokenizer_fingerprint) &&
        matches_if_set(config.template_fingerprint, delta.template_fingerprint) &&
        matches_if_set(config.generation_semantics_fingerprint,
            delta.generation_semantics_fingerprint);
}

} // namespace

bool common_flydelta_aggregation_config_validate(
        const common_flydelta_aggregation_config & config,
        std::string & error) {
    error.clear();
    if (!common_flydelta_direction_search_config_validate(config.identity, error) ||
            !common_flydelta_evidence_depth_config_validate(config.depth, error) ||
            config.max_retained_samples < config.depth.min_deep_samples ||
            config.max_retained_samples > config.depth.max_samples) {
        if (error.empty()) error = "FlyDelta aggregation configuration is invalid";
        return false;
    }
    return true;
}

common_flydelta_incremental_aggregation::common_flydelta_incremental_aggregation(
        common_flydelta_aggregation_config config)
    : config_(std::move(config)),
      mean_direction_(config_.identity.dimension, 0.0f),
      m2_(config_.identity.dimension, 0.0f) {}

bool common_flydelta_incremental_aggregation::ingest(
        const common_flydelta_contrast_sample & sample,
        std::string & error) {
    error.clear();
    if (!common_flydelta_aggregation_config_validate(config_, error) ||
            !common_flydelta_behavior_delta_validate(
                sample.delta, config_.identity.dimension, 64U * 1024U * 1024U, error) ||
            !common_flydelta_intervention_credit_validate(sample.credit, error)) {
        return false;
    }
    if (seen_sample_ids_.find(sample.delta.id) != seen_sample_ids_.end()) {
        // Snapshot replay and duplicate append-only references are idempotent.
        return true;
    }
    seen_sample_ids_.insert(sample.delta.id);
    ++observations_seen_;
    if (!compatible(sample.delta, config_) ||
            sample.credit.outcome == common_flydelta_counterfactual_outcome::harmed) {
        ++rejected_samples_;
        return true;
    }

    const float value_norm = norm(sample.delta.values);
    if (!std::isfinite(value_norm) || value_norm <= std::numeric_limits<float>::epsilon()) {
        error = "FlyDelta aggregation sample must not be zero";
        return false;
    }
    ++compatible_samples_;
    std::vector<float> normalized(sample.delta.values.size());
    for (size_t i = 0; i < normalized.size(); ++i) {
        normalized[i] = sample.delta.values[i] / value_norm;
    }

    const float count = static_cast<float>(compatible_samples_);
    for (size_t i = 0; i < normalized.size(); ++i) {
        const float difference = normalized[i] - mean_direction_[i];
        mean_direction_[i] += difference / count;
        const float updated_difference = normalized[i] - mean_direction_[i];
        m2_[i] += difference * updated_difference;
    }
    if (retained_samples_.size() < config_.max_retained_samples) {
        retained_sample_ids_.push_back(sample.delta.id);
        retained_samples_.push_back(sample);
    }
    return true;
}

bool common_flydelta_incremental_aggregation::assess_depth(
        common_flydelta_evidence_depth_result & result,
        std::string & error) const {
    return common_flydelta_assess_evidence_depth(
        config_.identity, config_.depth, retained_samples_, result, error);
}

common_flydelta_aggregation_snapshot common_flydelta_incremental_aggregation::snapshot() const {
    common_flydelta_aggregation_snapshot result;
    result.observations_seen = observations_seen_;
    result.compatible_samples = compatible_samples_;
    result.rejected_samples = rejected_samples_;
    result.mean_direction = mean_direction_;
    result.seen_sample_ids.assign(seen_sample_ids_.begin(), seen_sample_ids_.end());
    std::sort(result.seen_sample_ids.begin(), result.seen_sample_ids.end());
    result.retained_sample_ids = retained_sample_ids_;
    result.retained_samples = retained_samples_;
    if (compatible_samples_ > 1) {
        result.variance.resize(m2_.size(), 0.0f);
        const float divisor = static_cast<float>(compatible_samples_ - 1);
        for (size_t i = 0; i < m2_.size(); ++i) result.variance[i] = m2_[i] / divisor;
    }
    return result;
}
