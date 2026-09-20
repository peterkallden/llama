#include "agent/adaptation/flydelta/flydelta-concept.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace {

constexpr size_t k_max_reference = 512;

bool bounded(const std::string & value) {
    return !value.empty() && value.size() <= k_max_reference;
}

bool finite_vector(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

float norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) sum += static_cast<double>(value) * value;
    return static_cast<float>(std::sqrt(sum));
}

float cosine(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) return 0.0f;
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        dot += static_cast<double>(left[index]) * right[index];
        left_norm += static_cast<double>(left[index]) * left[index];
        right_norm += static_cast<double>(right[index]) * right[index];
    }
    if (left_norm <= 0.0 || right_norm <= 0.0) return 0.0f;
    return static_cast<float>(dot / std::sqrt(left_norm * right_norm));
}

std::vector<float> normalized(const std::vector<float> & values) {
    const float length = norm(values);
    if (!std::isfinite(length) || length <= 1.0e-8f) return {};
    std::vector<float> result = values;
    for (float & value : result) value /= length;
    return result;
}

std::vector<float> mean_vectors(
        const std::vector<std::vector<float>> & vectors,
        const std::vector<size_t> & indices,
        bool normalize_each) {
    if (vectors.empty() || indices.empty()) return {};
    std::vector<float> result(vectors.front().size(), 0.0f);
    for (size_t index : indices) {
        const std::vector<float> value = normalize_each ? normalized(vectors[index]) : vectors[index];
        if (value.empty()) return {};
        for (size_t dimension = 0; dimension < result.size(); ++dimension) {
            result[dimension] += value[dimension];
        }
    }
    const float divisor = static_cast<float>(indices.size());
    for (float & value : result) value /= divisor;
    return result;
}

std::vector<float> subtract(
        const std::vector<float> & left,
        const std::vector<float> & right) {
    if (left.size() != right.size()) return {};
    std::vector<float> result(left.size());
    for (size_t index = 0; index < left.size(); ++index) result[index] = left[index] - right[index];
    return result;
}

std::vector<float> add(
        const std::vector<float> & left,
        const std::vector<float> & right) {
    if (left.size() != right.size()) return {};
    std::vector<float> result(left.size());
    for (size_t index = 0; index < left.size(); ++index) result[index] = left[index] + right[index];
    return result;
}

bool make_candidate(
        const common_flydelta_concept_spec & spec,
        common_flydelta_concept_candidate_kind kind,
        int32_t layer,
        const std::vector<std::vector<float>> & residuals,
        const std::vector<size_t> & retained,
        const std::vector<float> & values,
        float median_alignment,
        common_flydelta_concept_candidate & candidate,
        std::string & error) {
    candidate = {};
    candidate.kind = kind;
    candidate.concept_key = spec.concept_key;
    candidate.extraction_id = spec.extraction_id;
    candidate.behavior_key = spec.behavior_key;
    candidate.model_profile_fingerprint = spec.model_profile_fingerprint;
    candidate.capture_layout_revision = spec.capture_layout_revision;
    candidate.layer_index = layer;
    candidate.values = normalized(values);
    candidate.source_trajectories = residuals.size();
    candidate.retained_trajectories = retained.size();
    candidate.median_alignment = median_alignment;
    candidate.control_residualized = true;
    candidate.experimental_only = true;
    candidate.learning_eligible = false;
    return common_flydelta_concept_candidate_validate(
        candidate, values.size(), error);
}

} // namespace

bool common_flydelta_concept_spec_validate(
        const common_flydelta_concept_spec & spec,
        std::string & error) {
    error.clear();
    if (spec.schema_version != 1 || !bounded(spec.concept_key) ||
            !bounded(spec.extraction_id) ||
            !bounded(spec.behavior_key) || !bounded(spec.source_ref) ||
            !bounded(spec.grounding_ref) ||
            !bounded(spec.verifier_ref) || !bounded(spec.model_profile_fingerprint) ||
            !bounded(spec.tokenizer_fingerprint) || !bounded(spec.template_fingerprint) ||
            !bounded(spec.capture_layout_revision) || !bounded(spec.scope_fingerprint) ||
            !spec.host_approved || !spec.redaction_attested) {
        error = "FlyDelta concept specification is incomplete or not host approved";
        return false;
    }
    return true;
}

bool common_flydelta_concept_trajectory_validate(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_trajectory & trajectory,
        size_t expected_dimension,
        std::string & error) {
    error.clear();
    if (!common_flydelta_concept_spec_validate(spec, error)) return false;
    if (trajectory.schema_version != 1 || !bounded(trajectory.id) ||
            !bounded(trajectory.fixture_ref) || !bounded(trajectory.baseline_capture_ref) ||
            !bounded(trajectory.conditioned_capture_ref) ||
            (spec.require_control && !bounded(trajectory.control_capture_ref)) ||
            !bounded(trajectory.semantic_anchor) || trajectory.layer_index <= 0 ||
            !trajectory.aligned || !trajectory.conditioned_host_verified ||
            trajectory.baseline.size() != expected_dimension ||
            trajectory.conditioned.size() != expected_dimension ||
            (spec.require_control && trajectory.control.size() != expected_dimension) ||
            !finite_vector(trajectory.baseline) || !finite_vector(trajectory.conditioned) ||
            !finite_vector(trajectory.control)) {
        error = "FlyDelta concept trajectory is incomplete, unaligned or unverified";
        return false;
    }
    return true;
}

bool common_flydelta_concept_build_config_validate(
        const common_flydelta_concept_build_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.dimension == 0 || !config.require_control ||
            config.min_trajectories < 2 || config.max_trajectories < config.min_trajectories ||
            config.max_trajectories > 64 || config.trim_fraction < 0.0f ||
            config.trim_fraction >= 0.5f || !std::isfinite(config.variance_ridge) ||
            config.variance_ridge <= 0.0f) {
        error = "FlyDelta concept build configuration is invalid";
        return false;
    }
    return true;
}

const char * common_flydelta_concept_candidate_kind_name(
        common_flydelta_concept_candidate_kind kind) {
    switch (kind) {
        case common_flydelta_concept_candidate_kind::raw_mean: return "raw_mean";
        case common_flydelta_concept_candidate_kind::trimmed_mean: return "trimmed_mean";
        case common_flydelta_concept_candidate_kind::diagonal_whitened_mean:
            return "diagonal_whitened_mean";
    }
    return "unknown";
}

bool common_flydelta_concept_candidate_validate(
        const common_flydelta_concept_candidate & candidate,
        size_t expected_dimension,
        std::string & error) {
    error.clear();
    if (candidate.schema_version != 1 || !bounded(candidate.concept_key) ||
            !bounded(candidate.extraction_id) ||
            !bounded(candidate.behavior_key) || candidate.origin != "host_taught_extracted" ||
            !bounded(candidate.model_profile_fingerprint) ||
            !bounded(candidate.capture_layout_revision) || candidate.layer_index <= 0 ||
            candidate.values.size() != expected_dimension || !finite_vector(candidate.values) ||
            norm(candidate.values) < 0.99f || norm(candidate.values) > 1.01f ||
            candidate.source_trajectories < 2 ||
            candidate.retained_trajectories < 2 || !candidate.control_residualized ||
            !candidate.experimental_only || candidate.learning_eligible) {
        error = "FlyDelta concept candidate is invalid or not experimental-only";
        return false;
    }
    return true;
}

bool common_flydelta_concept_candidate_to_direction(
        const common_flydelta_concept_candidate & candidate,
        common_flydelta_direction_candidate & direction,
        std::string & error) {
    error.clear();
    if (!common_flydelta_concept_candidate_validate(
                candidate, candidate.values.size(), error)) {
        return false;
    }

    direction = {};
    switch (candidate.kind) {
        case common_flydelta_concept_candidate_kind::raw_mean:
            direction.kind = common_flydelta_direction_kind::raw_repair;
            break;
        case common_flydelta_concept_candidate_kind::trimmed_mean:
            direction.kind = common_flydelta_direction_kind::normalized_trimmed_mean;
            break;
        case common_flydelta_concept_candidate_kind::diagonal_whitened_mean:
            direction.kind = common_flydelta_direction_kind::diagonal_whitened_mean;
            break;
    }
    direction.layer_index = candidate.layer_index;
    direction.values = candidate.values;
    direction.origin = candidate.origin;
    direction.extraction_id = candidate.extraction_id;
    direction.source_samples = candidate.source_trajectories;
    direction.retained_samples = candidate.retained_trajectories;
    direction.median_alignment = candidate.median_alignment;
    direction.experimental_only = true;
    return common_flydelta_direction_candidate_validate(
        direction, candidate.values.size(), error);
}

bool common_flydelta_build_concept_candidates(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_build_config & config,
        const std::vector<common_flydelta_concept_trajectory> & trajectories,
        std::vector<common_flydelta_concept_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    if (!common_flydelta_concept_build_config_validate(config, error) ||
            !common_flydelta_concept_spec_validate(spec, error)) return false;
    if (trajectories.size() < config.min_trajectories ||
            trajectories.size() > config.max_trajectories) {
        error = "FlyDelta concept has insufficient or excessive trajectories";
        return false;
    }

    std::vector<std::vector<float>> residuals;
    residuals.reserve(trajectories.size());
    for (const auto & trajectory : trajectories) {
        if (!common_flydelta_concept_trajectory_validate(
                spec, trajectory, config.dimension, error)) return false;
        if (!residuals.empty() && trajectory.layer_index != trajectories.front().layer_index) {
            error = "FlyDelta concept trajectories must share a layer for one candidate set";
            return false;
        }
        // conditioned - baseline minus the matched control displacement.
        const auto conditioned_delta = subtract(trajectory.conditioned, trajectory.baseline);
        const auto control_delta = subtract(trajectory.control, trajectory.baseline);
        auto residual = subtract(conditioned_delta, control_delta);
        if (residual.empty() || norm(residual) <= 1.0e-8f) {
            error = "FlyDelta concept trajectory has no usable residual signal";
            return false;
        }
        residuals.push_back(std::move(residual));
    }

    std::vector<size_t> all_indices(residuals.size(), 0);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    const auto initial_mean = normalized(mean_vectors(residuals, all_indices, true));
    if (initial_mean.empty()) {
        error = "FlyDelta concept trajectories have no common direction";
        return false;
    }

    std::vector<std::pair<float, size_t>> alignments;
    alignments.reserve(residuals.size());
    for (size_t index = 0; index < residuals.size(); ++index) {
        alignments.emplace_back(cosine(normalized(residuals[index]), initial_mean), index);
    }
    std::sort(alignments.begin(), alignments.end(),
        [](const auto & left, const auto & right) { return left.first > right.first; });
    const size_t requested_trim = static_cast<size_t>(
        std::floor(config.trim_fraction * residuals.size()));
    const size_t max_trim = residuals.size() - config.min_trajectories;
    const size_t trim = std::min(requested_trim, max_trim);
    std::vector<size_t> retained;
    retained.reserve(residuals.size() - trim);
    for (size_t index = trim; index < alignments.size(); ++index) {
        retained.push_back(alignments[index].second);
    }
    std::vector<float> retained_alignments;
    for (size_t index : retained) {
        retained_alignments.push_back(cosine(normalized(residuals[index]), initial_mean));
    }
    std::sort(retained_alignments.begin(), retained_alignments.end());
    const float median_alignment = retained_alignments[retained_alignments.size() / 2];

    const auto raw_mean = mean_vectors(residuals, all_indices, true);
    const auto trimmed_mean = mean_vectors(residuals, retained, true);
    std::vector<float> mean_raw = mean_vectors(residuals, retained, false);
    std::vector<float> variance(config.dimension, 0.0f);
    for (size_t index : retained) {
        for (size_t dimension = 0; dimension < config.dimension; ++dimension) {
            const float difference = residuals[index][dimension] - mean_raw[dimension];
            variance[dimension] += difference * difference;
        }
    }
    const float divisor = static_cast<float>(std::max<size_t>(1, retained.size()));
    for (float & value : variance) value /= divisor;
    std::vector<float> whitened = mean_raw;
    for (size_t dimension = 0; dimension < whitened.size(); ++dimension) {
        whitened[dimension] /= std::sqrt(variance[dimension] + config.variance_ridge);
    }

    for (const auto & kind_values : {
            std::pair<common_flydelta_concept_candidate_kind, std::vector<float>>{
                common_flydelta_concept_candidate_kind::raw_mean, raw_mean},
            {common_flydelta_concept_candidate_kind::trimmed_mean, trimmed_mean},
            {common_flydelta_concept_candidate_kind::diagonal_whitened_mean, whitened}}) {
        common_flydelta_concept_candidate candidate;
        if (!make_candidate(spec, kind_values.first, trajectories.front().layer_index,
                residuals, retained, kind_values.second, median_alignment, candidate, error)) {
            candidates.clear();
            return false;
        }
        candidates.push_back(std::move(candidate));
    }
    return true;
}
