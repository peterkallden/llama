#include "astc-vulkan-artifact-policy.h"

#include <cmath>

bool astc_vulkan_parse_quality_policy(const std::string & name,
                                      astc_vulkan_quality_policy & policy) {
    if (name == "quality") policy = astc_vulkan_quality_policy::quality;
    else if (name == "balanced") policy = astc_vulkan_quality_policy::balanced;
    else if (name == "compact" || name == "size") policy = astc_vulkan_quality_policy::size;
    else if (name == "speed") policy = astc_vulkan_quality_policy::speed;
    else if (name == "auto") policy = astc_vulkan_quality_policy::automatic;
    else return false;
    return true;
}

const char * astc_vulkan_quality_policy_name(astc_vulkan_quality_policy policy) {
    switch (policy) {
        case astc_vulkan_quality_policy::quality: return "quality";
        case astc_vulkan_quality_policy::balanced: return "balanced";
        case astc_vulkan_quality_policy::size: return "compact";
        case astc_vulkan_quality_policy::speed: return "speed";
        case astc_vulkan_quality_policy::automatic: return "auto";
    }
    return "unknown";
}

bool astc_vulkan_resolve_user_profile(const std::string & name,
                                      astc_vulkan_user_profile_defaults & defaults) {
    if (name == "quality") {
        defaults.policy = astc_vulkan_quality_policy::quality;
        defaults.footprint = astc_vulkan_footprint::k4x4;
        defaults.representation = astc_vulkan_representation::kScalar;
    } else if (name == "balanced") {
        defaults.policy = astc_vulkan_quality_policy::balanced;
        defaults.footprint = astc_vulkan_footprint::k6x6;
        defaults.representation = astc_vulkan_representation::kScalar;
    } else if (name == "compact") {
        defaults.policy = astc_vulkan_quality_policy::size;
        defaults.footprint = astc_vulkan_footprint::k8x5;
        defaults.representation = astc_vulkan_representation::kPairedD2;
    } else if (name == "speed") {
        defaults.policy = astc_vulkan_quality_policy::speed;
        defaults.footprint = astc_vulkan_footprint::k6x6;
        defaults.representation = astc_vulkan_representation::kScalar;
    } else if (name == "auto") {
        defaults.policy = astc_vulkan_quality_policy::automatic;
        defaults.footprint = astc_vulkan_footprint::k6x6;
        defaults.representation = astc_vulkan_representation::kScalar;
    } else {
        return false;
    }
    return true;
}

bool astc_vulkan_artifact_has_robust_replay_evidence(
        const astc_vulkan_artifact_evidence & evidence) {
    return evidence.replay_case_count > 1 &&
           std::isfinite(evidence.median_loss_delta) &&
           std::isfinite(evidence.worst_loss_delta) &&
           std::isfinite(evidence.worst_top1_agreement);
}

float astc_vulkan_artifact_median_loss_delta(
        const astc_vulkan_artifact_evidence & evidence) {
    return astc_vulkan_artifact_has_robust_replay_evidence(evidence) ?
        evidence.median_loss_delta : evidence.loss_delta;
}

float astc_vulkan_artifact_worst_loss_delta(
        const astc_vulkan_artifact_evidence & evidence) {
    return astc_vulkan_artifact_has_robust_replay_evidence(evidence) ?
        evidence.worst_loss_delta : evidence.loss_delta;
}

float astc_vulkan_artifact_worst_top1_agreement(
        const astc_vulkan_artifact_evidence & evidence) {
    return astc_vulkan_artifact_has_robust_replay_evidence(evidence) ?
        evidence.worst_top1_agreement : evidence.top1_agreement;
}

bool astc_vulkan_artifact_is_eligible(const astc_vulkan_artifact_candidate & candidate,
                                      bool device_supports_format, bool fits_memory_budget) {
    return astc_vulkan_artifact_is_eligible(candidate, device_supports_format,
                                             fits_memory_budget, {});
}

bool astc_vulkan_artifact_is_eligible(const astc_vulkan_artifact_candidate & candidate,
                                      bool device_supports_format, bool fits_memory_budget,
                                      const astc_vulkan_artifact_selection_rules & rules) {
    const auto & evidence = candidate.evidence;
    const float worst_loss = astc_vulkan_artifact_worst_loss_delta(evidence);
    const float worst_top1 = astc_vulkan_artifact_worst_top1_agreement(evidence);
    return candidate.tensor != nullptr && device_supports_format && fits_memory_budget &&
           evidence.model_gate_passed && evidence.vulkan_gate_passed &&
           std::isfinite(evidence.loss_delta) && std::isfinite(evidence.logits_relative_mse) &&
           std::isfinite(worst_loss) && std::isfinite(worst_top1) &&
           worst_loss <= rules.max_worst_loss_delta &&
           worst_top1 >= rules.min_worst_top1_agreement &&
           std::isfinite(candidate.rate_bpw) && candidate.rate_bpw > 0.0;
}

bool astc_vulkan_artifact_policy_precedes(const astc_vulkan_artifact_candidate & left,
                                          const astc_vulkan_artifact_candidate & right,
                                          astc_vulkan_quality_policy policy) {
    return astc_vulkan_artifact_policy_precedes(left, right, policy, {});
}

bool astc_vulkan_artifact_policy_precedes(const astc_vulkan_artifact_candidate & left,
                                          const astc_vulkan_artifact_candidate & right,
                                          astc_vulkan_quality_policy policy,
                                          const astc_vulkan_artifact_selection_rules & rules) {
    if (policy == astc_vulkan_quality_policy::size && left.rate_bpw != right.rate_bpw) {
        return left.rate_bpw < right.rate_bpw;
    }
    const float left_loss = astc_vulkan_artifact_median_loss_delta(left.evidence);
    const float right_loss = astc_vulkan_artifact_median_loss_delta(right.evidence);
    if (std::fabs(left_loss - right_loss) > rules.simplicity_loss_epsilon) {
        return left_loss < right_loss;
    }
    // A near-tie should favour the less transformation- and trace-dependent
    // artifact before applying proxy metrics.  This keeps neutral and
    // unscaled streams as the default unless selection/normalization has a
    // measurable model-level advantage.
    if (left.normalization != right.normalization) {
        return left.normalization == astc_vulkan_normalization::none;
    }
    if (left.variant != right.variant) {
        return left.variant == astc_vulkan_artifact_variant::neutral;
    }
    if (left.evidence.logits_relative_mse != right.evidence.logits_relative_mse) {
        return left.evidence.logits_relative_mse < right.evidence.logits_relative_mse;
    }
    const float left_top1 = astc_vulkan_artifact_worst_top1_agreement(left.evidence);
    const float right_top1 = astc_vulkan_artifact_worst_top1_agreement(right.evidence);
    if (left_top1 != right_top1) {
        return left_top1 > right_top1;
    }
    return left_loss < right_loss;
}
