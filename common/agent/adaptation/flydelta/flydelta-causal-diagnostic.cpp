#include "flydelta-causal-diagnostic.h"

#include <cmath>

namespace {

bool bounded(const std::string & value, const size_t max_size = 512) {
    return !value.empty() && value.size() <= max_size;
}

} // namespace

const char * common_flydelta_causal_patch_kind_name(
        const common_flydelta_causal_patch_kind kind) {
    switch (kind) {
        case common_flydelta_causal_patch_kind::self_replacement:
            return "self_replacement";
        case common_flydelta_causal_patch_kind::exact_replacement:
            return "exact_replacement";
        case common_flydelta_causal_patch_kind::scaled_exact_delta:
            return "scaled_exact_delta";
        case common_flydelta_causal_patch_kind::existing_direction:
            return "existing_direction";
        case common_flydelta_causal_patch_kind::norm_matched_control:
            return "norm_matched_control";
        case common_flydelta_causal_patch_kind::wrong_concept:
            return "wrong_concept";
    }
    return "self_replacement";
}

bool common_flydelta_causal_patch_kind_from_name(
        const std::string & value,
        common_flydelta_causal_patch_kind & kind) {
    if (value == "self_replacement") {
        kind = common_flydelta_causal_patch_kind::self_replacement;
    } else if (value == "exact_replacement") {
        kind = common_flydelta_causal_patch_kind::exact_replacement;
    } else if (value == "scaled_exact_delta") {
        kind = common_flydelta_causal_patch_kind::scaled_exact_delta;
    } else if (value == "existing_direction") {
        kind = common_flydelta_causal_patch_kind::existing_direction;
    } else if (value == "norm_matched_control") {
        kind = common_flydelta_causal_patch_kind::norm_matched_control;
    } else if (value == "wrong_concept") {
        kind = common_flydelta_causal_patch_kind::wrong_concept;
    } else {
        return false;
    }
    return true;
}
bool common_flydelta_causal_diagnostic_arm_validate(
        const common_flydelta_causal_diagnostic_arm & arm,
        std::string & error) {
    error.clear();
    if (arm.schema_version != 1 || !bounded(arm.arm_id) || arm.absolute_position < 0 ||
            (arm.patch_attempted && !arm.patch_applied && arm.executed) ||
            !std::isfinite(arm.baseline_margin) || !std::isfinite(arm.candidate_margin) ||
            !std::isfinite(arm.margin_delta) ||
            (arm.margin_available && std::fabs(
                (arm.candidate_margin - arm.baseline_margin) - arm.margin_delta) > 1.0e-3f) ||
            (arm.host_evaluated && !arm.executed) ||
            (arm.verifier_known && !arm.host_evaluated)) {
        error = "FlyDelta causal diagnostic arm is invalid";
        return false;
    }
    if (!arm.evidence_ref.empty() && !bounded(arm.evidence_ref)) {
        error = "FlyDelta causal diagnostic arm evidence reference is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_causal_diagnostic_report_validate(
        const common_flydelta_causal_diagnostic_report & report,
        std::string & error) {
    error.clear();
    if (report.schema_version != 1 || !bounded(report.experiment_id) ||
            !bounded(report.manifest_ref) || report.arms.empty() || report.arms.size() > 32 ||
            report.learning_eligible || report.promotion_eligible || report.activation_authority) {
        error = "FlyDelta causal diagnostic report is invalid or crosses lifecycle boundary";
        return false;
    }
    for (const auto & arm : report.arms) {
        if (!common_flydelta_causal_diagnostic_arm_validate(arm, error)) return false;
    }
    return true;
}
