#include "agent/adaptation/flydelta/flydelta-gate.h"

#include <cmath>

namespace {

bool unit(float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

void no_op(common_flydelta_gate_decision & decision, const char * reason) {
    decision.apply = false;
    decision.scale = 0.0f;
    decision.reason = reason;
}

} // namespace

bool common_flydelta_gate_config_validate(
        const common_flydelta_gate_config & config,
        std::string & error) {
    error.clear();
    if (!unit(config.min_familiarity) || !unit(config.max_novelty) ||
            !std::isfinite(config.max_scale) || config.max_scale <= 0.0f ||
            config.max_scale > 1.0f) {
        error = "FlyDelta gate configuration is invalid";
        return false;
    }
    if (config.min_familiarity + config.max_novelty < 1.0f) {
        error = "FlyDelta gate familiarity and novelty bounds are inconsistent";
        return false;
    }
    return true;
}

bool common_flydelta_gate_decide(
        const common_flydelta_gate_config & config,
        const common_flydelta_gate_request & request,
        common_flydelta_gate_decision & decision,
        std::string & error) {
    error.clear();
    decision = {};
    if (!common_flydelta_gate_config_validate(config, error) ||
            !unit(request.familiarity) || !unit(request.novelty) ||
            !std::isfinite(request.requested_scale) || request.requested_scale < 0.0f ||
            request.requested_scale > config.max_scale) {
        if (error.empty()) error = "FlyDelta gate request is invalid";
        return false;
    }
    if (!config.enabled) { no_op(decision, "feature_disabled"); return true; }
    if (!request.explicit_opt_in) { no_op(decision, "explicit_opt_in_required"); return true; }
    if (request.candidate_status != common_flydelta_candidate_status::approved) {
        no_op(decision, "candidate_not_approved"); return true;
    }
    if (!request.basis_available) { no_op(decision, "basis_unavailable"); return true; }
    if (request.familiarity < config.min_familiarity || request.novelty > config.max_novelty) {
        no_op(decision, "context_not_familiar"); return true;
    }
    if (request.requested_scale == 0.0f) { no_op(decision, "requested_scale_zero"); return true; }
    decision.apply = true;
    decision.scale = request.requested_scale;
    decision.reason = "approved_familiar_context";
    return true;
}

bool common_flydelta_gate_request_from_context(
        const common_flydelta_recognition_memory & recognition,
        const common_flydelta_sparse_code & code,
        bool explicit_opt_in,
        common_flydelta_candidate_status candidate_status,
        bool basis_available,
        float requested_scale,
        common_flydelta_gate_request & request,
        std::string & error) {
    error.clear();
    if (code.expansion_dim == 0 || code.indices.size() != code.values.size()) {
        error = "FlyDelta recognition context is invalid";
        return false;
    }
    request = {};
    request.explicit_opt_in = explicit_opt_in;
    request.candidate_status = candidate_status;
    request.basis_available = basis_available;
    request.familiarity = recognition.familiarity(code);
    request.novelty = recognition.novelty(code);
    request.requested_scale = requested_scale;
    return true;
}
