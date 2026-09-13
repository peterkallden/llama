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
    if (!common_flydelta_compose_static_overlay(
            request.artifact_id,
            request.model_n_embd,
            request.model_n_layers,
            request.il_start,
            request.il_end,
            request.directions,
            request.coefficients,
            result.gate,
            max_bytes,
            result.overlay,
            error)) {
        result = {};
        return false;
    }
    return true;
}
