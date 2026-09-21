#include "agent/adaptation/flydelta/flydelta-activation.h"

#include <cmath>

namespace {

bool nonempty_bounded(const std::string & value, size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

} // namespace

bool common_flydelta_prepare_activation(
        const common_flydelta_gate_config & gate_config,
        const common_flydelta_activation_request & request,
        size_t max_bytes,
        common_flydelta_activation_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_gate_decide(
            gate_config, request.gate_request, result.gate, error)) {
        return false;
    }
    if (!result.gate.apply) return true;
    if (!nonempty_bounded(request.candidate_id) ||
            !nonempty_bounded(request.model_profile_fingerprint) ||
            !nonempty_bounded(request.capture_layout_revision)) {
        error = "FlyDelta activation metadata is incomplete";
        result = {};
        return false;
    }
    common_flydelta_sparse_overlay sparse_overlay;
    if (!common_flydelta_compose_sparse_overlay(
            request.artifact_id,
            request.model_n_embd,
            request.model_n_layers,
            request.il_start,
            request.il_end,
            request.directions,
            request.coefficients,
            result.gate,
            max_bytes,
            sparse_overlay,
            error)) {
        result = {};
        return false;
    }
    if (!common_flydelta_expand_sparse_overlay(
            sparse_overlay,
            request.model_n_embd,
            request.model_n_layers,
            max_bytes,
            result.overlay,
            error)) {
        result = {};
        return false;
    }
    result.sparse_overlay = std::move(sparse_overlay);
    return true;
}

bool common_flydelta_prepare_activation_from_artifact(
        const common_flydelta_artifact & artifact,
        const common_flydelta_compatibility & expected,
        const std::string & model_profile_fingerprint,
        const common_flydelta_sparse_code & code,
        const std::string & candidate_id,
        const common_flydelta_gate_config & gate_config,
        const common_flydelta_gate_request & gate_request,
        size_t max_artifact_weights,
        size_t max_artifact_bytes,
        size_t max_overlay_bytes,
        common_flydelta_activation_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (artifact.schema_version != 2 || artifact.content_hash.empty() ||
            artifact.content_hash != common_flydelta_artifact_hash(artifact) ||
            !common_flydelta_artifact_validate(
                artifact, max_artifact_weights, max_artifact_bytes, error) ||
            !common_flydelta_artifact_matches(artifact, expected, error)) {
        if (error.empty()) error = "FlyDelta activation requires a verified compatible v2 artifact";
        return false;
    }
    if (code.expansion_dim != artifact.encoder.expansion_dim ||
            code.indices.size() != code.values.size() ||
            !nonempty_bounded(model_profile_fingerprint) ||
            !nonempty_bounded(candidate_id)) {
        error = "FlyDelta artifact activation context is invalid";
        return false;
    }

    common_flydelta_delta_memory memory(artifact.memory);
    if (!memory.set_weights(artifact.weights, error)) return false;
    std::vector<float> coefficients;
    if (!memory.predict(code, coefficients, error)) return false;

    std::vector<common_flydelta_basis_direction> directions;
    directions.reserve(artifact.steering_basis.size());
    for (const auto & source : artifact.steering_basis) {
        common_flydelta_basis_direction direction;
        direction.layer_index = source.layer_index;
        direction.values = source.values;
        directions.push_back(std::move(direction));
    }
    common_flydelta_activation_request request;
    request.candidate_id = candidate_id;
    request.artifact_id = artifact.id;
    request.model_profile_fingerprint = model_profile_fingerprint;
    request.capture_layout_revision = expected.inference_layout_revision;
    request.model_n_embd = artifact.model_n_embd;
    request.model_n_layers = artifact.model_n_layers;
    request.il_start = artifact.il_start;
    request.il_end = artifact.il_end;
    request.directions = std::move(directions);
    request.coefficients = std::move(coefficients);
    request.gate_request = gate_request;
    return common_flydelta_prepare_activation(
        gate_config, request, max_overlay_bytes, result, error);
}

bool common_flydelta_activation_result_validate(
        const common_flydelta_activation_result & result,
        size_t model_n_embd,
        size_t model_n_layers,
        size_t max_bytes,
        std::string & error) {
    error.clear();
    if (result.gate.apply != result.overlay.enabled) {
        error = "FlyDelta activation gate and overlay state differ";
        return false;
    }
    if (!result.gate.apply) {
        if (!result.sparse_overlay.artifact_id.empty() ||
                !result.sparse_overlay.layer_indices.empty() ||
                !result.sparse_overlay.data.empty() ||
                !result.overlay.artifact_id.empty() || !result.overlay.data.empty()) {
            error = "FlyDelta no-op activation contains overlay data";
            return false;
        }
        return true;
    }
    if (!common_flydelta_static_overlay_validate(
            result.overlay, model_n_embd, model_n_layers, max_bytes, error)) {
        return false;
    }
    if (!result.sparse_overlay.enabled) {
        // Accept legacy dense-only activations while model hosts migrate to
        // the per-sequence sparse material contract.
        return true;
    }
    if (!common_flydelta_sparse_overlay_validate(
            result.sparse_overlay, model_n_embd, model_n_layers, max_bytes, error)) {
        return false;
    }
    if (result.sparse_overlay.artifact_id != result.overlay.artifact_id ||
            result.sparse_overlay.n_embd != result.overlay.n_embd ||
            result.sparse_overlay.il_start != result.overlay.il_start ||
            result.sparse_overlay.il_end != result.overlay.il_end) {
        error = "FlyDelta sparse and dense activation identities differ";
        return false;
    }
    common_flydelta_static_overlay expanded;
    if (!common_flydelta_expand_sparse_overlay(
            result.sparse_overlay, model_n_embd, model_n_layers, max_bytes,
            expanded, error)) {
        return false;
    }
    if (expanded.data != result.overlay.data) {
        error = "FlyDelta sparse and dense activation data differ";
        return false;
    }
    return true;
}
