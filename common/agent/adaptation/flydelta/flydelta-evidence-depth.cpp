#include "agent/adaptation/flydelta/flydelta-evidence-depth.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

float norm(const std::vector<float> & values) {
    float squared = 0.0f;
    for (const float value : values) squared += value * value;
    return std::sqrt(squared);
}

float dot(const std::vector<float> & left, const std::vector<float> & right) {
    float value = 0.0f;
    for (size_t i = 0; i < left.size(); ++i) value += left[i] * right[i];
    return value;
}

float median(std::vector<float> values) {
    if (values.empty()) return 0.0f;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) return (values[middle - 1] + values[middle]) * 0.5f;
    return values[middle];
}

void jacobi_eigenvalues(std::vector<std::vector<float>> matrix, std::vector<float> & values) {
    const size_t size = matrix.size();
    for (size_t iteration = 0; iteration < 32U * std::max<size_t>(1, size * size); ++iteration) {
        size_t row = 0;
        size_t column = 0;
        float largest = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = i + 1; j < size; ++j) {
                if (std::fabs(matrix[i][j]) > largest) {
                    largest = std::fabs(matrix[i][j]);
                    row = i;
                    column = j;
                }
            }
        }
        if (largest <= 1.0e-6f) break;
        const float angle = 0.5f * std::atan2(
            2.0f * matrix[row][column], matrix[column][column] - matrix[row][row]);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        for (size_t i = 0; i < size; ++i) {
            if (i == row || i == column) continue;
            const float old_row = matrix[i][row];
            const float old_column = matrix[i][column];
            matrix[i][row] = matrix[row][i] = cosine * old_row - sine * old_column;
            matrix[i][column] = matrix[column][i] = sine * old_row + cosine * old_column;
        }
        const float old_row_value = matrix[row][row];
        const float old_column_value = matrix[column][column];
        const float old_off_diagonal = matrix[row][column];
        matrix[row][row] = cosine * cosine * old_row_value -
            2.0f * sine * cosine * old_off_diagonal + sine * sine * old_column_value;
        matrix[column][column] = sine * sine * old_row_value +
            2.0f * sine * cosine * old_off_diagonal + cosine * cosine * old_column_value;
        matrix[row][column] = matrix[column][row] = 0.0f;
    }
    values.clear();
    values.reserve(size);
    for (size_t i = 0; i < size; ++i) values.push_back(std::max(0.0f, matrix[i][i]));
    std::sort(values.begin(), values.end(), std::greater<float>());
}

bool same_identity(
        const common_flydelta_behavior_delta & delta,
        const common_flydelta_direction_search_config & identity) {
    return delta.source == identity.source && delta.behavior_key == identity.behavior_key &&
        delta.model_profile_fingerprint == identity.model_profile_fingerprint &&
        delta.execution_context_fingerprint == identity.execution_context_fingerprint &&
        delta.capture_layout_revision == identity.capture_layout_revision &&
        delta.layer_index == identity.layer_index;
}

} // namespace

const char * common_flydelta_search_depth_name(common_flydelta_search_depth depth) {
    switch (depth) {
        case common_flydelta_search_depth::bootstrap: return "bootstrap";
        case common_flydelta_search_depth::shallow: return "shallow";
        case common_flydelta_search_depth::deep: return "deep";
    }
    return "unknown";
}

bool common_flydelta_evidence_depth_config_validate(
        const common_flydelta_evidence_depth_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.min_shallow_samples == 0 ||
            config.min_deep_samples < config.min_shallow_samples ||
            config.max_samples < config.min_deep_samples || config.max_samples > 256 ||
            !std::isfinite(config.rank_relative_tolerance) ||
            config.rank_relative_tolerance <= 0.0f || config.rank_relative_tolerance > 0.5f ||
            !std::isfinite(config.min_median_alignment) ||
            config.min_median_alignment < -1.0f || config.min_median_alignment > 1.0f ||
            !std::isfinite(config.max_condition_number) || config.max_condition_number < 1.0f) {
        error = "FlyDelta evidence depth configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_assess_evidence_depth(
        const common_flydelta_direction_search_config & identity,
        const common_flydelta_evidence_depth_config & config,
        const std::vector<common_flydelta_contrast_sample> & samples,
        common_flydelta_evidence_depth_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_direction_search_config_validate(identity, error) ||
            !common_flydelta_evidence_depth_config_validate(config, error) ||
            samples.size() > config.max_samples) {
        if (error.empty()) error = "FlyDelta evidence depth input is invalid";
        return false;
    }

    std::vector<std::vector<float>> normalized;
    normalized.reserve(samples.size());
    for (const auto & sample : samples) {
        if (!common_flydelta_behavior_delta_validate(
                sample.delta, identity.dimension, 64U * 1024U * 1024U, error) ||
                !common_flydelta_intervention_credit_validate(sample.credit, error)) {
            return false;
        }
        if (!same_identity(sample.delta, identity) ||
                sample.credit.outcome == common_flydelta_counterfactual_outcome::harmed) {
            ++result.incompatible_samples;
            continue;
        }
        const float value_norm = norm(sample.delta.values);
        std::vector<float> values(identity.dimension);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = sample.delta.values[i] / value_norm;
        }
        normalized.push_back(std::move(values));
    }
    result.compatible_samples = normalized.size();
    if (normalized.empty()) return true;

    std::vector<float> pairwise_alignment;
    for (size_t i = 0; i < normalized.size(); ++i) {
        for (size_t j = i + 1; j < normalized.size(); ++j) {
            pairwise_alignment.push_back(dot(normalized[i], normalized[j]));
        }
    }
    result.median_alignment = pairwise_alignment.empty() ? 1.0f : median(pairwise_alignment);
    result.geometry_stable = result.median_alignment >= config.min_median_alignment;

    std::vector<std::vector<float>> gram(
        normalized.size(), std::vector<float>(normalized.size(), 0.0f));
    for (size_t i = 0; i < normalized.size(); ++i) {
        for (size_t j = 0; j < normalized.size(); ++j) {
            gram[i][j] = dot(normalized[i], normalized[j]);
        }
    }
    std::vector<float> eigenvalues;
    jacobi_eigenvalues(std::move(gram), eigenvalues);
    const float largest = eigenvalues.empty() ? 0.0f : eigenvalues.front();
    const float threshold = largest * config.rank_relative_tolerance;
    float eigenvalue_sum = 0.0f;
    float eigenvalue_squared_sum = 0.0f;
    float smallest_retained = std::numeric_limits<float>::max();
    for (const float eigenvalue : eigenvalues) {
        if (eigenvalue <= threshold) continue;
        ++result.effective_rank;
        eigenvalue_sum += eigenvalue;
        eigenvalue_squared_sum += eigenvalue * eigenvalue;
        smallest_retained = std::min(smallest_retained, eigenvalue);
    }
    if (eigenvalue_squared_sum > std::numeric_limits<float>::epsilon()) {
        result.stable_rank = eigenvalue_sum * eigenvalue_sum / eigenvalue_squared_sum;
    }
    if (result.effective_rank > 0 && smallest_retained > 0.0f) {
        result.condition_number = std::sqrt(largest / smallest_retained);
    }
    result.basis_condition_ok = result.effective_rank <= 1 ||
        result.condition_number <= config.max_condition_number;
    result.shallow_ready = result.compatible_samples >= config.min_shallow_samples &&
        result.effective_rank >= 2 && result.basis_condition_ok;
    result.deep_ready = result.compatible_samples >= config.min_deep_samples &&
        result.effective_rank >= 2 && result.geometry_stable && result.basis_condition_ok;
    result.depth = result.deep_ready ? common_flydelta_search_depth::deep :
        result.shallow_ready ? common_flydelta_search_depth::shallow :
        common_flydelta_search_depth::bootstrap;
    return true;
}
