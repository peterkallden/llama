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
    candidate.synthesis_semantics =
        common_flydelta_concept_synthesis_semantics::control_residualized;
    candidate.concept_key = spec.concept_key;
    candidate.extraction_id = spec.extraction_id;
    candidate.behavior_key = spec.behavior_key;
    candidate.model_profile_fingerprint = spec.model_profile_fingerprint;
    candidate.capture_layout_revision = spec.capture_layout_revision;
    candidate.layer_index = layer;
    candidate.values = normalized(values);
    candidate.source_trajectories = residuals.size();
    candidate.retained_trajectories = retained.size();
    candidate.control_trajectories = residuals.size();
    candidate.retained_control_trajectories = retained.size();
    candidate.median_alignment = median_alignment;
    candidate.control_residualized = true;
    candidate.experimental_only = true;
    candidate.learning_eligible = false;
    return common_flydelta_concept_candidate_validate(
        candidate, values.size(), error);
}

std::vector<float> prototype_mean(
        const std::vector<common_flydelta_concept_prototype_sample> & samples,
        const std::vector<size_t> & indices) {
    std::vector<std::vector<float>> values;
    values.reserve(indices.size());
    for (const size_t index : indices) values.push_back(samples[index].values);
    return mean_vectors(values, [&]() {
        std::vector<size_t> local(indices.size());
        std::iota(local.begin(), local.end(), 0);
        return local;
    }(), true);
}

std::vector<size_t> prototype_retained_indices(
        const std::vector<common_flydelta_concept_prototype_sample> & samples,
        float trim_fraction) {
    std::vector<size_t> all(samples.size());
    std::iota(all.begin(), all.end(), 0);
    if (samples.size() < 2 || trim_fraction <= 0.0f) return all;

    const auto centroid = prototype_mean(samples, all);
    std::vector<std::pair<float, size_t>> ranked;
    ranked.reserve(samples.size());
    for (const size_t index : all) {
        ranked.emplace_back(cosine(normalized(samples[index].values), centroid), index);
    }
    std::sort(ranked.begin(), ranked.end(),
        [](const auto & left, const auto & right) { return left.first > right.first; });
    const size_t requested_trim = static_cast<size_t>(
        std::floor(trim_fraction * samples.size()));
    const size_t trim = std::min(requested_trim, samples.size() - 2);
    std::vector<size_t> retained;
    retained.reserve(samples.size() - trim);
    for (size_t index = 0; index < ranked.size() - trim; ++index) {
        retained.push_back(ranked[index].second);
    }
    return retained;
}

bool same_prototype_identity(
        const common_flydelta_concept_prototype_sample & left,
        const common_flydelta_concept_prototype_sample & right) {
    // A positive prototype intentionally combines independent manifestations.
    // Their task anchors and fixture/verifier refs may differ; model, capture
    // site and scope identity must not.
    return left.model_profile_fingerprint == right.model_profile_fingerprint &&
        left.tokenizer_fingerprint == right.tokenizer_fingerprint &&
        left.template_fingerprint == right.template_fingerprint &&
        left.capture_layout_revision == right.capture_layout_revision &&
        left.scope_fingerprint == right.scope_fingerprint &&
        left.layer_index == right.layer_index;
}

bool make_prototype_candidate(
        const common_flydelta_concept_spec & spec,
        common_flydelta_concept_candidate_kind kind,
        int32_t layer,
        const std::vector<float> & values,
        common_flydelta_concept_synthesis_semantics semantics,
        const char * origin,
        size_t primary_count,
        size_t retained_primary_count,
        size_t control_count,
        size_t retained_control_count,
        size_t negative_count,
        size_t retained_negative_count,
        common_flydelta_concept_candidate & candidate,
        std::string & error) {
    candidate = {};
    candidate.kind = kind;
    candidate.synthesis_semantics = semantics;
    candidate.concept_key = spec.concept_key;
    candidate.extraction_id = spec.extraction_id;
    candidate.behavior_key = spec.behavior_key;
    candidate.origin = origin;
    candidate.model_profile_fingerprint = spec.model_profile_fingerprint;
    candidate.capture_layout_revision = spec.capture_layout_revision;
    candidate.layer_index = layer;
    candidate.values = normalized(values);
    candidate.source_trajectories = primary_count;
    candidate.retained_trajectories = retained_primary_count;
    candidate.control_trajectories = control_count;
    candidate.retained_control_trajectories = retained_control_count;
    candidate.negative_trajectories = negative_count;
    candidate.retained_negative_trajectories = retained_negative_count;
    candidate.control_residualized = false;
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
    const bool has_negative_ref = !trajectory.negative_capture_ref.empty();
    const bool has_negative_values = !trajectory.negative.empty();
    if (trajectory.schema_version != 1 || !bounded(trajectory.id) ||
            !bounded(trajectory.fixture_ref) || !bounded(trajectory.baseline_capture_ref) ||
            !bounded(trajectory.conditioned_capture_ref) ||
            (spec.require_control && !bounded(trajectory.control_capture_ref)) ||
            (has_negative_ref != has_negative_values) ||
            (has_negative_ref &&
                (!bounded(trajectory.negative_capture_ref) ||
                 !trajectory.negative_host_verified ||
                 trajectory.negative.size() != expected_dimension)) ||
            !bounded(trajectory.semantic_anchor) || trajectory.layer_index <= 0 ||
            !trajectory.aligned || !trajectory.conditioned_host_verified ||
            trajectory.baseline.size() != expected_dimension ||
            trajectory.conditioned.size() != expected_dimension ||
            (spec.require_control && trajectory.control.size() != expected_dimension) ||
            !finite_vector(trajectory.baseline) || !finite_vector(trajectory.conditioned) ||
            !finite_vector(trajectory.control) ||
            (has_negative_values && !finite_vector(trajectory.negative))) {
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

const char * common_flydelta_concept_synthesis_semantics_name(
        common_flydelta_concept_synthesis_semantics semantics) {
    switch (semantics) {
        case common_flydelta_concept_synthesis_semantics::control_residualized:
            return "control_residualized";
        case common_flydelta_concept_synthesis_semantics::positive_prototype:
            return "positive_prototype";
        case common_flydelta_concept_synthesis_semantics::negative_repulsion:
            return "negative_repulsion";
    }
    return "unknown";
}

std::vector<size_t> common_flydelta_select_concept_synthesis_frontier(
        const std::vector<common_flydelta_concept_candidate> & candidates,
        size_t max_candidates) {
    std::vector<size_t> selected;
    if (max_candidates == 0) return selected;
    selected.reserve(std::min(max_candidates, candidates.size()));

    // Keep the semantic comparison deterministic and bounded. The raw
    // estimator is preferred because it is the least transformed candidate;
    // a source's first available estimator remains a safe fallback.
    const common_flydelta_concept_synthesis_semantics semantics[] = {
        common_flydelta_concept_synthesis_semantics::control_residualized,
        common_flydelta_concept_synthesis_semantics::positive_prototype,
        common_flydelta_concept_synthesis_semantics::negative_repulsion,
    };
    const auto has_primary = std::any_of(candidates.begin(), candidates.end(), [](const auto & candidate) {
        return candidate.synthesis_semantics ==
                common_flydelta_concept_synthesis_semantics::control_residualized ||
            candidate.synthesis_semantics ==
                common_flydelta_concept_synthesis_semantics::positive_prototype;
    });
    for (const auto semantic : semantics) {
        if (selected.size() >= max_candidates) break;
        if (semantic == common_flydelta_concept_synthesis_semantics::negative_repulsion &&
                !has_primary) continue;
        size_t fallback = candidates.size();
        size_t preferred = candidates.size();
        for (size_t index = 0; index < candidates.size(); ++index) {
            if (candidates[index].synthesis_semantics != semantic) continue;
            if (fallback == candidates.size()) fallback = index;
            if (candidates[index].kind == common_flydelta_concept_candidate_kind::raw_mean) {
                preferred = index;
                break;
            }
        }
        const size_t chosen = preferred != candidates.size() ? preferred : fallback;
        if (chosen != candidates.size() &&
                std::find(selected.begin(), selected.end(), chosen) == selected.end()) {
            selected.push_back(chosen);
        }
    }
    for (size_t index = 0; index < candidates.size() && selected.size() < max_candidates; ++index) {
        if (candidates[index].synthesis_semantics ==
                common_flydelta_concept_synthesis_semantics::negative_repulsion &&
                !has_primary) continue;
        if (std::find(selected.begin(), selected.end(), index) == selected.end()) {
            selected.push_back(index);
        }
    }
    return selected;
}

bool common_flydelta_concept_candidate_validate(
        const common_flydelta_concept_candidate & candidate,
        size_t expected_dimension,
        std::string & error) {
    error.clear();
    const bool residualized = candidate.synthesis_semantics ==
        common_flydelta_concept_synthesis_semantics::control_residualized;
    const bool prototype = candidate.synthesis_semantics ==
        common_flydelta_concept_synthesis_semantics::positive_prototype;
    const bool negative = candidate.synthesis_semantics ==
        common_flydelta_concept_synthesis_semantics::negative_repulsion;
    if (candidate.schema_version != 1 || !bounded(candidate.concept_key) ||
            !bounded(candidate.extraction_id) ||
            !bounded(candidate.behavior_key) ||
            (candidate.origin != "host_taught_extracted" &&
             candidate.origin != "host_taught_positive_prototype" &&
             candidate.origin != "host_taught_negative_repulsion") ||
            !bounded(candidate.model_profile_fingerprint) ||
            !bounded(candidate.capture_layout_revision) || candidate.layer_index <= 0 ||
            candidate.values.size() != expected_dimension || !finite_vector(candidate.values) ||
            norm(candidate.values) < 0.99f || norm(candidate.values) > 1.01f ||
            candidate.source_trajectories < 2 ||
            candidate.retained_trajectories < 2 ||
            !candidate.experimental_only || candidate.learning_eligible ||
            (!residualized && !prototype && !negative) ||
            (residualized && candidate.origin != "host_taught_extracted") ||
            (prototype && candidate.origin != "host_taught_positive_prototype") ||
            (negative && candidate.origin != "host_taught_negative_repulsion") ||
            (residualized && (!candidate.control_residualized ||
                candidate.control_trajectories < 2 ||
                candidate.retained_control_trajectories < 2 ||
                candidate.negative_trajectories != 0 ||
                candidate.retained_negative_trajectories != 0)) ||
            (prototype && (candidate.control_residualized ||
                candidate.control_trajectories < 2 ||
                candidate.retained_control_trajectories < 2 ||
                candidate.negative_trajectories != 0 ||
                candidate.retained_negative_trajectories != 0)) ||
            (negative && (candidate.control_residualized ||
                candidate.control_trajectories < 2 ||
                candidate.retained_control_trajectories < 2 ||
                candidate.negative_trajectories < 2 ||
                candidate.retained_negative_trajectories < 2 ||
                candidate.negative_trajectories != candidate.source_trajectories ||
                candidate.retained_negative_trajectories != candidate.retained_trajectories))) {
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
            direction.kind = candidate.synthesis_semantics ==
                    common_flydelta_concept_synthesis_semantics::positive_prototype
                ? common_flydelta_direction_kind::positive_prototype
                : candidate.synthesis_semantics ==
                    common_flydelta_concept_synthesis_semantics::negative_repulsion
                    ? common_flydelta_direction_kind::negative_repulsion
                    : common_flydelta_direction_kind::raw_repair;
            break;
        case common_flydelta_concept_candidate_kind::trimmed_mean:
            direction.kind = candidate.synthesis_semantics ==
                    common_flydelta_concept_synthesis_semantics::positive_prototype
                ? common_flydelta_direction_kind::positive_prototype_trimmed_mean
                : candidate.synthesis_semantics ==
                    common_flydelta_concept_synthesis_semantics::negative_repulsion
                    ? common_flydelta_direction_kind::negative_repulsion_trimmed_mean
                    : common_flydelta_direction_kind::normalized_trimmed_mean;
            break;
        case common_flydelta_concept_candidate_kind::diagonal_whitened_mean:
            direction.kind = candidate.synthesis_semantics ==
                    common_flydelta_concept_synthesis_semantics::positive_prototype
                ? common_flydelta_direction_kind::positive_prototype_diagonal_whitened_mean
                : candidate.synthesis_semantics ==
                    common_flydelta_concept_synthesis_semantics::negative_repulsion
                    ? common_flydelta_direction_kind::negative_repulsion_diagonal_whitened_mean
                    : common_flydelta_direction_kind::diagonal_whitened_mean;
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
    const size_t keep = alignments.size() - trim;
    for (size_t index = 0; index < keep; ++index) {
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

bool common_flydelta_concept_prototype_build_config_validate(
        const common_flydelta_concept_prototype_build_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.dimension == 0 ||
            config.min_positive_samples < 2 || config.min_control_samples < 2 ||
            config.max_samples < config.min_positive_samples ||
            config.max_samples < config.min_control_samples ||
            config.max_samples > 64 || config.trim_fraction < 0.0f ||
            config.trim_fraction >= 0.5f || !std::isfinite(config.variance_ridge) ||
            config.variance_ridge <= 0.0f) {
        error = "FlyDelta positive prototype build configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_concept_prototype_sample_validate(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_prototype_sample & sample,
        size_t expected_dimension,
        std::string & error) {
    error.clear();
    if (!common_flydelta_concept_spec_validate(spec, error)) return false;
    if (sample.schema_version != 1 || !bounded(sample.id) ||
            !bounded(sample.capture_ref) || !bounded(sample.semantic_anchor) ||
            sample.model_profile_fingerprint != spec.model_profile_fingerprint ||
            sample.tokenizer_fingerprint != spec.tokenizer_fingerprint ||
            sample.template_fingerprint != spec.template_fingerprint ||
            sample.capture_layout_revision != spec.capture_layout_revision ||
            sample.scope_fingerprint != spec.scope_fingerprint ||
            !bounded(sample.verifier_ref) || sample.layer_index <= 0 ||
            sample.values.size() != expected_dimension || !finite_vector(sample.values) ||
            norm(sample.values) <= 1.0e-8f || !sample.host_verified ||
            !sample.independent) {
        error = "FlyDelta prototype capture is incompatible or not host verified";
        return false;
    }
    return true;
}

bool common_flydelta_build_positive_prototype_candidates(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_prototype_build_config & config,
        const std::vector<common_flydelta_concept_prototype_sample> & positive,
        const std::vector<common_flydelta_concept_prototype_sample> & controls,
        std::vector<common_flydelta_concept_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();
    if (!common_flydelta_concept_spec_validate(spec, error) ||
            !common_flydelta_concept_prototype_build_config_validate(config, error)) {
        return false;
    }
    if (positive.size() < config.min_positive_samples ||
            controls.size() < config.min_control_samples ||
            positive.size() > config.max_samples || controls.size() > config.max_samples) {
        error = "FlyDelta positive prototype has insufficient or excessive samples";
        return false;
    }

    std::vector<std::string> seen_ids;
    seen_ids.reserve(positive.size() + controls.size());
    auto validate_set = [&](const auto & samples) {
        for (const auto & sample : samples) {
            if (!common_flydelta_concept_prototype_sample_validate(
                    spec, sample, config.dimension, error)) return false;
            if (!seen_ids.empty() && std::find(seen_ids.begin(), seen_ids.end(), sample.id) !=
                    seen_ids.end()) {
                error = "FlyDelta positive prototype reuses a capture sample";
                return false;
            }
            seen_ids.push_back(sample.id);
        }
        return true;
    };
    if (!validate_set(positive) || !validate_set(controls)) return false;

    for (const auto & sample : positive) {
        if (!same_prototype_identity(sample, positive.front())) {
            error = "FlyDelta positive prototype samples are not compatible";
            return false;
        }
    }
    for (const auto & sample : controls) {
        if (!same_prototype_identity(sample, positive.front())) {
            error = "FlyDelta positive prototype controls are not compatible";
            return false;
        }
    }

    const auto positive_indices = prototype_retained_indices(positive, config.trim_fraction);
    const auto control_indices = prototype_retained_indices(controls, config.trim_fraction);
    std::vector<size_t> all_positive(positive.size());
    std::vector<size_t> all_controls(controls.size());
    std::iota(all_positive.begin(), all_positive.end(), 0);
    std::iota(all_controls.begin(), all_controls.end(), 0);
    const auto raw_positive_mean = prototype_mean(positive, all_positive);
    const auto raw_control_mean = prototype_mean(controls, all_controls);
    const auto raw_difference = subtract(raw_positive_mean, raw_control_mean);
    const auto positive_mean = prototype_mean(positive, positive_indices);
    const auto control_mean = prototype_mean(controls, control_indices);
    const auto trimmed_difference = subtract(positive_mean, control_mean);
    if (raw_difference.empty() || trimmed_difference.empty() ||
            norm(raw_difference) <= 1.0e-8f || norm(trimmed_difference) <= 1.0e-8f) {
        error = "FlyDelta positive prototype has no usable positive-control signal";
        return false;
    }

    std::vector<float> variance(config.dimension, 0.0f);
    for (const size_t index : positive_indices) {
        const auto value = normalized(positive[index].values);
        for (size_t dimension = 0; dimension < config.dimension; ++dimension) {
            const float difference = value[dimension] - positive_mean[dimension];
            variance[dimension] += difference * difference;
        }
    }
    for (const size_t index : control_indices) {
        const auto value = normalized(controls[index].values);
        for (size_t dimension = 0; dimension < config.dimension; ++dimension) {
            const float difference = value[dimension] - control_mean[dimension];
            variance[dimension] += difference * difference;
        }
    }
    const float divisor = static_cast<float>(positive_indices.size() + control_indices.size());
    for (float & value : variance) value /= divisor;
    std::vector<float> whitened = trimmed_difference;
    for (size_t dimension = 0; dimension < whitened.size(); ++dimension) {
        whitened[dimension] /= std::sqrt(variance[dimension] + config.variance_ridge);
    }

    for (const auto & kind_values : {
            std::pair<common_flydelta_concept_candidate_kind, std::vector<float>>{
                common_flydelta_concept_candidate_kind::raw_mean, raw_difference},
            {common_flydelta_concept_candidate_kind::trimmed_mean, trimmed_difference},
            {common_flydelta_concept_candidate_kind::diagonal_whitened_mean, whitened}}) {
        common_flydelta_concept_candidate candidate;
        if (!make_prototype_candidate(
                spec, kind_values.first, positive.front().layer_index,
                kind_values.second,
                common_flydelta_concept_synthesis_semantics::positive_prototype,
                "host_taught_positive_prototype",
                positive.size(), positive_indices.size(), controls.size(),
                control_indices.size(), 0, 0, candidate, error)) {
            candidates.clear();
            return false;
        }
        candidates.push_back(std::move(candidate));
    }
    return true;
}

bool common_flydelta_build_negative_repulsion_candidates(
        const common_flydelta_concept_spec & spec,
        const common_flydelta_concept_prototype_build_config & config,
        const std::vector<common_flydelta_concept_prototype_sample> & negative,
        const std::vector<common_flydelta_concept_prototype_sample> & controls,
        std::vector<common_flydelta_concept_candidate> & candidates,
        std::string & error) {
    error.clear();
    candidates.clear();

    // Reuse the existing prototype estimator, trimming and diagonal
    // whitening.  Its positive-minus-control output is negated here so the
    // resulting semantic direction is control-minus-negative.  This keeps
    // whitening an estimator choice rather than a second semantic path.
    std::vector<common_flydelta_concept_candidate> prototype_candidates;
    if (!common_flydelta_build_positive_prototype_candidates(
            spec, config, negative, controls, prototype_candidates, error)) {
        if (!error.empty()) {
            error = "FlyDelta negative repulsion material is invalid: " + error;
        }
        return false;
    }
    for (auto & candidate : prototype_candidates) {
        for (float & value : candidate.values) value = -value;
        candidate.synthesis_semantics =
            common_flydelta_concept_synthesis_semantics::negative_repulsion;
        candidate.origin = "host_taught_negative_repulsion";
        candidate.negative_trajectories = candidate.source_trajectories;
        candidate.retained_negative_trajectories = candidate.retained_trajectories;
        candidate.control_residualized = false;
        if (!common_flydelta_concept_candidate_validate(
                candidate, config.dimension, error)) {
            candidates.clear();
            return false;
        }
        candidates.push_back(std::move(candidate));
    }
    return true;
}
