#include "agent/adaptation/flydelta/flydelta-coefficient-search.h"

#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace {

bool finite(float value) {
    return std::isfinite(value);
}

float norm(const std::vector<float> & values) {
    float sum = 0.0f;
    for (const float value : values) sum += value * value;
    return std::sqrt(sum);
}

float dot(const std::vector<float> & left, const std::vector<float> & right) {
    float value = 0.0f;
    for (size_t i = 0; i < left.size(); ++i) value += left[i] * right[i];
    return value;
}

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

} // namespace

bool common_flydelta_low_rank_basis_validate(
        const common_flydelta_low_rank_basis & basis,
        size_t max_rank,
        std::string & error) {
    error.clear();
    if (basis.schema_version != 1 || basis.dimension == 0 ||
            basis.dimension > (1U << 20) || basis.layer_index < 0 ||
            basis.vectors.empty() || basis.vectors.size() > max_rank) {
        error = "FlyDelta low-rank basis identity or bounds are invalid";
        return false;
    }
    for (const auto & vector : basis.vectors) {
        if (vector.size() != basis.dimension || norm(vector) <= std::numeric_limits<float>::epsilon()) {
            error = "FlyDelta low-rank basis vector is invalid";
            return false;
        }
        for (const float value : vector) {
            if (!finite(value)) {
                error = "FlyDelta low-rank basis contains a non-finite value";
                return false;
            }
        }
    }
    return true;
}

bool common_flydelta_build_low_rank_basis(
        size_t dimension,
        size_t max_rank,
        const std::vector<common_flydelta_direction_candidate> & candidates,
        common_flydelta_low_rank_basis & basis,
        std::string & error) {
    error.clear();
    basis = {};
    if (dimension == 0 || dimension > (1U << 20) || max_rank == 0 || max_rank > 16 ||
            candidates.empty() || candidates.size() > 256) {
        error = "FlyDelta low-rank basis input bounds are invalid";
        return false;
    }
    basis.dimension = dimension;
    basis.layer_index = candidates.front().layer_index;
    for (const auto & candidate : candidates) {
        if (!common_flydelta_direction_candidate_validate(candidate, dimension, error) ||
                candidate.layer_index != basis.layer_index) {
            if (error.empty()) error = "FlyDelta low-rank candidates are incompatible";
            return false;
        }
        if (basis.vectors.size() == max_rank) break;
        std::vector<float> vector = candidate.values;
        for (const auto & existing : basis.vectors) {
            const float projection = dot(vector, existing);
            for (size_t i = 0; i < vector.size(); ++i) vector[i] -= projection * existing[i];
        }
        const float vector_norm = norm(vector);
        if (!finite(vector_norm)) {
            error = "FlyDelta low-rank basis normalization failed";
            return false;
        }
        if (vector_norm <= 0.000001f) continue;
        for (float & value : vector) value /= vector_norm;
        basis.vectors.push_back(std::move(vector));
    }
    if (basis.vectors.empty()) {
        error = "FlyDelta low-rank candidates do not span a nonzero basis";
        return false;
    }
    return common_flydelta_low_rank_basis_validate(basis, max_rank, error);
}

bool common_flydelta_build_paired_intervention_basis(
        const common_flydelta_paired_intervention_proposal & proposal,
        const common_flydelta_direction_candidate & prefer,
        const common_flydelta_direction_candidate & avoid,
        common_flydelta_low_rank_basis & basis,
        std::string & error) {
    error.clear();
    basis = {};
    if (!common_flydelta_paired_intervention_proposal_validate(proposal, error) ||
            prefer.layer_index != proposal.layer_index ||
            avoid.layer_index != proposal.layer_index ||
            prefer.kind != proposal.prefer_direction_kind ||
            avoid.kind != proposal.avoid_direction_kind ||
            prefer.values.size() != avoid.values.size() ||
            prefer.values.empty()) {
        if (error.empty()) error = "FlyDelta paired intervention basis identity is invalid";
        return false;
    }
    if (!common_flydelta_direction_candidate_validate(
                prefer, prefer.values.size(), error) ||
            !common_flydelta_direction_candidate_validate(
                avoid, avoid.values.size(), error)) {
        return false;
    }
    const std::vector<common_flydelta_direction_candidate> components = {prefer, avoid};
    if (!common_flydelta_build_low_rank_basis(
                prefer.values.size(), 2, components, basis, error)) return false;
    if (basis.vectors.size() != 2) {
        error = "FlyDelta paired intervention components do not span rank two";
        basis = {};
        return false;
    }
    return true;
}

bool common_flydelta_semantic_basis_query_validate(
        const common_flydelta_semantic_basis_query & query,
        std::string & error) {
    error.clear();
    if (query.schema_version != 1 ||
            !common_adaptation_evidence_source_name(query.source) ||
            std::string(common_adaptation_evidence_source_name(query.source)) == "unknown" ||
            !nonempty_bounded(query.concept_key) ||
            !nonempty_bounded(query.behavior_key) ||
            !nonempty_bounded(query.model_profile_fingerprint) ||
            !nonempty_bounded(query.tokenizer_fingerprint) ||
            !nonempty_bounded(query.template_fingerprint) ||
            !nonempty_bounded(query.capture_layout_revision) ||
            !nonempty_bounded(query.scope_fingerprint) ||
            !nonempty_bounded(query.oracle_ref) ||
            !nonempty_bounded(query.oracle_revision) ||
            query.max_rank == 0 || query.max_rank > 16) {
        error = "FlyDelta semantic basis query identity or rank is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_resolve_semantic_basis(
        const common_flydelta_semantic_basis_query & query,
        const std::vector<common_flydelta_semantic_direction_candidate> & candidates,
        common_flydelta_low_rank_basis & basis,
        std::vector<size_t> & selected_candidate_indices,
        std::string & error) {
    error.clear();
    basis = {};
    selected_candidate_indices.clear();
    if (!common_flydelta_semantic_basis_query_validate(query, error) ||
            candidates.empty() || candidates.size() > 256) {
        if (error.empty()) error = "FlyDelta semantic basis candidates are out of bounds";
        return false;
    }

    std::vector<common_flydelta_direction_candidate> selected;
    selected.reserve(query.max_rank);
    for (size_t index = 0; index < candidates.size() && selected.size() < query.max_rank; ++index) {
        const auto & candidate = candidates[index];
        if (!common_flydelta_synthesis_candidate_descriptor_validate(
                    candidate.descriptor, error) ||
                !common_flydelta_direction_candidate_validate(
                    candidate.direction, candidate.direction.values.size(), error)) {
            return false;
        }
        const auto & descriptor = candidate.descriptor;
        const bool identity_match =
            descriptor.source == query.source &&
            descriptor.concept_key == query.concept_key &&
            descriptor.behavior_key == query.behavior_key &&
            descriptor.model_profile_fingerprint == query.model_profile_fingerprint &&
            descriptor.tokenizer_fingerprint == query.tokenizer_fingerprint &&
            descriptor.template_fingerprint == query.template_fingerprint &&
            descriptor.capture_layout_revision == query.capture_layout_revision &&
            descriptor.scope_fingerprint == query.scope_fingerprint &&
            descriptor.oracle_ref == query.oracle_ref &&
            descriptor.oracle_revision == query.oracle_revision;
        if (!identity_match || !descriptor.host_verified || descriptor.experimental_only ||
                candidate.direction.experimental_only) continue;

        std::vector<common_flydelta_direction_candidate> trial = selected;
        trial.push_back(candidate.direction);
        common_flydelta_low_rank_basis trial_basis;
        if (!common_flydelta_build_low_rank_basis(
                    candidate.direction.values.size(), query.max_rank, trial,
                    trial_basis, error)) return false;
        // Gram-Schmidt silently drops collinear directions. Do not report a
        // descriptor as selected when it contributed no new basis dimension.
        if (trial_basis.vectors.size() == selected.size()) continue;
        selected.push_back(candidate.direction);
        selected_candidate_indices.push_back(index);
        basis = std::move(trial_basis);
    }
    if (selected.empty()) {
        error = "FlyDelta semantic basis has no compatible verified directions";
        return false;
    }
    return common_flydelta_low_rank_basis_validate(basis, query.max_rank, error);
}
