#include "agent/adaptation/flydelta/flydelta-direction-search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

float dot(const std::vector<float> & left, const std::vector<float> & right) {
    float result = 0.0f;
    for (size_t i = 0; i < left.size(); ++i) result += left[i] * right[i];
    return result;
}

float norm(const std::vector<float> & values) {
    float squared = 0.0f;
    for (const float value : values) squared += value * value;
    return std::sqrt(squared);
}

bool normalize(const std::vector<float> & input, std::vector<float> & output) {
    const float input_norm = norm(input);
    if (!std::isfinite(input_norm) || input_norm <= std::numeric_limits<float>::epsilon()) return false;
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) output[i] = input[i] / input_norm;
    return true;
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) return (values[middle - 1] + values[middle]) * 0.5f;
    return values[middle];
}

bool supported_kind(common_flydelta_direction_kind kind) {
    switch (kind) {
        case common_flydelta_direction_kind::raw_repair:
        case common_flydelta_direction_kind::normalized_trimmed_mean:
        case common_flydelta_direction_kind::diagonal_whitened_mean:
        case common_flydelta_direction_kind::token_margin_direction:
        case common_flydelta_direction_kind::execution_boundary_prototype:
            return true;
    }
    return false;
}

common_flydelta_direction_candidate make_candidate(
        common_flydelta_direction_kind kind, int32_t layer_index,
        std::vector<float> values, size_t source_samples, size_t retained_samples,
        float median_alignment, bool experimental_only = false) {
    common_flydelta_direction_candidate result;
    result.kind = kind;
    result.layer_index = layer_index;
    result.values = std::move(values);
    result.source_samples = source_samples;
    result.retained_samples = retained_samples;
    result.median_alignment = median_alignment;
    result.experimental_only = experimental_only;
    return result;
}

} // namespace

const char * common_flydelta_direction_kind_name(
        common_flydelta_direction_kind kind) {
    switch (kind) {
        case common_flydelta_direction_kind::raw_repair: return "raw_repair";
        case common_flydelta_direction_kind::normalized_trimmed_mean: return "normalized_trimmed_mean";
        case common_flydelta_direction_kind::diagonal_whitened_mean: return "diagonal_whitened_mean";
        case common_flydelta_direction_kind::token_margin_direction: return "token_margin_direction";
        case common_flydelta_direction_kind::execution_boundary_prototype: return "execution_boundary_prototype";
    }
    return "unknown";
}

const char * common_flydelta_direction_search_mode_name(
        common_flydelta_direction_search_mode mode) {
    switch (mode) {
        case common_flydelta_direction_search_mode::learning: return "learning";
        case common_flydelta_direction_search_mode::experimental: return "experimental";
    }
    return "unknown";
}

bool common_flydelta_direction_search_config_validate(
        const common_flydelta_direction_search_config & config,
        std::string & error) {
    error.clear();
    const bool valid_mode =
        config.mode == common_flydelta_direction_search_mode::learning ||
        config.mode == common_flydelta_direction_search_mode::experimental;
    if (config.schema_version != 1 || config.dimension == 0 || config.dimension > (1U << 20) ||
            config.layer_index < 0 || config.min_samples == 0 ||
            config.max_samples < config.min_samples || config.max_samples > 256 ||
            !std::isfinite(config.min_median_alignment) || config.min_median_alignment < -1.0f ||
            config.min_median_alignment > 1.0f || !std::isfinite(config.trim_fraction) ||
            config.trim_fraction < 0.0f || config.trim_fraction >= 0.5f ||
            !std::isfinite(config.variance_ridge) || config.variance_ridge <= 0.0f ||
            !valid_mode ||
            !nonempty_bounded(config.behavior_key) ||
            !nonempty_bounded(config.model_profile_fingerprint) ||
            !nonempty_bounded(config.execution_context_fingerprint) ||
            !nonempty_bounded(config.capture_layout_revision)) {
        error = "FlyDelta direction search configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_build_token_margin_candidate(
        const common_flydelta_direction_search_config & config,
        const common_flydelta_token_margin_material & material,
        common_flydelta_direction_candidate & candidate,
        std::string & error) {
    error.clear();
    candidate = {};
    if (!common_flydelta_direction_search_config_validate(config, error) ||
            material.positive_output_row.size() != config.dimension ||
            material.negative_output_row.size() != config.dimension) {
        if (error.empty()) error = "FlyDelta token-margin material dimension is invalid";
        return false;
    }
    std::vector<float> difference(config.dimension);
    for (size_t i = 0; i < config.dimension; ++i) {
        difference[i] = material.positive_output_row[i] - material.negative_output_row[i];
    }
    std::vector<float> normalized;
    if (!normalize(difference, normalized)) {
        error = "FlyDelta token-margin direction must not be zero";
        return false;
    }
    candidate = make_candidate(
        common_flydelta_direction_kind::token_margin_direction,
        config.layer_index, std::move(normalized), 1, 1, 1.0f,
        config.mode == common_flydelta_direction_search_mode::experimental);
    return common_flydelta_direction_candidate_validate(candidate, config.dimension, error);
}

bool common_flydelta_build_boundary_prototype_candidate(
        const common_flydelta_direction_search_config & config,
        const std::vector<common_flydelta_boundary_sample> & samples,
        common_flydelta_direction_candidate & candidate,
        std::string & error) {
    error.clear();
    candidate = {};
    if (!common_flydelta_direction_search_config_validate(config, error) ||
            samples.empty() || samples.size() > config.max_samples) {
        if (error.empty()) error = "FlyDelta boundary-prototype sample bounds are invalid";
        return false;
    }
    size_t positives = 0;
    size_t negatives = 0;
    std::vector<float> positive_mean(config.dimension, 0.0f);
    std::vector<float> negative_mean(config.dimension, 0.0f);
    for (const auto & sample : samples) {
        if (sample.values.size() != config.dimension) {
            error = "FlyDelta boundary-prototype sample dimension is invalid";
            return false;
        }
        for (const float value : sample.values) {
            if (!std::isfinite(value)) {
                error = "FlyDelta boundary-prototype sample contains a non-finite value";
                return false;
            }
        }
        auto & mean = sample.positive ? positive_mean : negative_mean;
        for (size_t i = 0; i < config.dimension; ++i) mean[i] += sample.values[i];
        if (sample.positive) ++positives; else ++negatives;
    }
    if (positives == 0 || negatives == 0) {
        error = "FlyDelta boundary prototype requires positive and negative samples";
        return false;
    }
    for (float & value : positive_mean) value /= static_cast<float>(positives);
    for (float & value : negative_mean) value /= static_cast<float>(negatives);
    std::vector<float> difference(config.dimension);
    for (size_t i = 0; i < config.dimension; ++i) {
        difference[i] = positive_mean[i] - negative_mean[i];
    }
    std::vector<float> normalized;
    if (!normalize(difference, normalized)) {
        error = "FlyDelta boundary prototype must not be zero";
        return false;
    }
    candidate = make_candidate(
        common_flydelta_direction_kind::execution_boundary_prototype,
        config.layer_index, std::move(normalized), samples.size(),
        samples.size(), 1.0f,
        config.mode == common_flydelta_direction_search_mode::experimental);
    return common_flydelta_direction_candidate_validate(candidate, config.dimension, error);
}

bool common_flydelta_direction_candidate_validate(
        const common_flydelta_direction_candidate & candidate,
        size_t expected_dimension,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1 || !supported_kind(candidate.kind) ||
            candidate.layer_index < 0 || expected_dimension == 0 ||
            candidate.values.size() != expected_dimension || candidate.source_samples == 0 ||
            candidate.retained_samples == 0 || candidate.retained_samples > candidate.source_samples ||
            !std::isfinite(candidate.median_alignment) || candidate.median_alignment < -1.0f ||
            candidate.median_alignment > 1.0f) {
        error = "FlyDelta direction candidate identity or bounds are invalid";
        return false;
    }
    float squared = 0.0f;
    for (const float value : candidate.values) {
        if (!std::isfinite(value)) {
            error = "FlyDelta direction candidate contains a non-finite value";
            return false;
        }
        squared += value * value;
    }
    if (!std::isfinite(squared) || squared <= std::numeric_limits<float>::epsilon()) {
        error = "FlyDelta direction candidate must not be zero";
        return false;
    }
    return true;
}

bool common_flydelta_build_direction_candidates(
        const common_flydelta_direction_search_config & config,
        const std::vector<common_flydelta_contrast_sample> & samples,
        std::vector<common_flydelta_direction_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    if (!common_flydelta_direction_search_config_validate(config, error) ||
            samples.empty() || samples.size() > config.max_samples) {
        if (error.empty()) error = "FlyDelta direction search sample bounds are invalid";
        return false;
    }

    struct normalized_sample {
        std::vector<float> values;
        float median_alignment = 0.0f;
    };
    std::vector<normalized_sample> normalized;
    normalized.reserve(samples.size());
    std::unordered_set<std::string> ids;
    const bool experimental_mode =
        config.mode == common_flydelta_direction_search_mode::experimental;
    for (const auto & sample : samples) {
        if (!common_flydelta_behavior_delta_validate(
                    sample.delta, config.dimension, 64U * 1024U * 1024U, error) ||
                !common_flydelta_intervention_credit_validate(sample.credit, error)) {
            return false;
        }
        const bool outcome_allowed = experimental_mode
            ? sample.credit.outcome != common_flydelta_counterfactual_outcome::harmed
            : sample.credit.outcome == common_flydelta_counterfactual_outcome::helped &&
                sample.credit.eligible_for_learning;
        if (!outcome_allowed ||
                sample.delta.source != config.source ||
                sample.delta.behavior_key != config.behavior_key ||
                sample.delta.model_profile_fingerprint != config.model_profile_fingerprint ||
                sample.delta.execution_context_fingerprint != config.execution_context_fingerprint ||
                sample.delta.capture_layout_revision != config.capture_layout_revision ||
                sample.delta.layer_index != config.layer_index ||
                !ids.insert(sample.delta.id).second) {
            error = experimental_mode
                ? "FlyDelta experimental direction search samples are incompatible or harmed"
                : "FlyDelta direction search samples are incompatible or not host-helped";
            return false;
        }
        normalized_sample value;
        if (!normalize(sample.delta.values, value.values)) {
            error = "FlyDelta direction search sample is zero";
            return false;
        }
        normalized.push_back(std::move(value));
    }

    for (size_t i = 0; i < normalized.size(); ++i) {
        std::vector<float> similarities;
        similarities.reserve(normalized.size() - 1);
        for (size_t j = 0; j < normalized.size(); ++j) {
            if (i != j) similarities.push_back(dot(normalized[i].values, normalized[j].values));
        }
        normalized[i].median_alignment = similarities.empty() ? 1.0f : median(std::move(similarities));
    }

    candidates.push_back(make_candidate(
        common_flydelta_direction_kind::raw_repair, config.layer_index,
        normalized.front().values, samples.size(), 1, normalized.front().median_alignment,
        experimental_mode));
    if (samples.size() < config.min_samples) return true;

    std::vector<size_t> retained;
    for (size_t i = 0; i < normalized.size(); ++i) {
        if (normalized[i].median_alignment >= config.min_median_alignment) retained.push_back(i);
    }
    if (retained.size() < config.min_samples) return true;

    std::sort(retained.begin(), retained.end(), [&](size_t left, size_t right) {
        return normalized[left].median_alignment > normalized[right].median_alignment;
    });
    const size_t trimmed_count = static_cast<size_t>(std::ceil(
        static_cast<float>(retained.size()) * (1.0f - config.trim_fraction)));
    retained.resize(std::min(retained.size(), std::max(config.min_samples, trimmed_count)));
    const auto retained_alignment = [&]() {
        std::vector<float> values;
        values.reserve(retained.size());
        for (const size_t index : retained) values.push_back(normalized[index].median_alignment);
        return median(std::move(values));
    }();

    std::vector<float> mean(config.dimension, 0.0f);
    for (const size_t index : retained) {
        for (size_t dimension = 0; dimension < config.dimension; ++dimension) {
            mean[dimension] += normalized[index].values[dimension];
        }
    }
    for (float & value : mean) value /= static_cast<float>(retained.size());
    std::vector<float> normalized_mean;
    if (!normalize(mean, normalized_mean)) return true;
    candidates.push_back(make_candidate(
        common_flydelta_direction_kind::normalized_trimmed_mean, config.layer_index,
        normalized_mean, samples.size(), retained.size(), retained_alignment,
        experimental_mode));

    std::vector<float> whitened(config.dimension, 0.0f);
    for (size_t dimension = 0; dimension < config.dimension; ++dimension) {
        float variance = 0.0f;
        for (const size_t index : retained) {
            const float difference = normalized[index].values[dimension] - mean[dimension];
            variance += difference * difference;
        }
        variance /= static_cast<float>(retained.size());
        whitened[dimension] = mean[dimension] / (variance + config.variance_ridge);
    }
    std::vector<float> normalized_whitened;
    if (normalize(whitened, normalized_whitened)) {
        candidates.push_back(make_candidate(
            common_flydelta_direction_kind::diagonal_whitened_mean, config.layer_index,
            normalized_whitened, samples.size(), retained.size(), retained_alignment,
            experimental_mode));
    }
    for (const auto & value : candidates) {
        if (!common_flydelta_direction_candidate_validate(value, config.dimension, error)) {
            candidates.clear();
            return false;
        }
    }
    return true;
}
