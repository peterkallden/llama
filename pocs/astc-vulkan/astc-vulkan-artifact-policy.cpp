#include "astc-vulkan-artifact-policy.h"

#include <cmath>

bool astc_vulkan_parse_quality_policy(const std::string & name,
                                      astc_vulkan_quality_policy & policy) {
    if (name == "quality") policy = astc_vulkan_quality_policy::quality;
    else if (name == "balanced") policy = astc_vulkan_quality_policy::balanced;
    else if (name == "size" || name == "compact") policy = astc_vulkan_quality_policy::size;
    else if (name == "speed") policy = astc_vulkan_quality_policy::speed;
    else if (name == "auto") policy = astc_vulkan_quality_policy::automatic;
    else return false;
    return true;
}

const char * astc_vulkan_quality_policy_name(astc_vulkan_quality_policy policy) {
    switch (policy) {
        case astc_vulkan_quality_policy::quality: return "quality";
        case astc_vulkan_quality_policy::balanced: return "balanced";
        case astc_vulkan_quality_policy::size: return "size";
        case astc_vulkan_quality_policy::speed: return "speed";
        case astc_vulkan_quality_policy::automatic: return "auto";
    }
    return "unknown";
}

bool astc_vulkan_artifact_is_eligible(const astc_vulkan_artifact_candidate & candidate,
                                      bool device_supports_format, bool fits_memory_budget) {
    const auto & evidence = candidate.evidence;
    return candidate.tensor != nullptr && device_supports_format && fits_memory_budget &&
           evidence.model_gate_passed && evidence.vulkan_gate_passed &&
           std::isfinite(evidence.loss_delta) && std::isfinite(evidence.logits_relative_mse) &&
           std::isfinite(candidate.rate_bpw) && candidate.rate_bpw > 0.0;
}

bool astc_vulkan_artifact_policy_precedes(const astc_vulkan_artifact_candidate & left,
                                          const astc_vulkan_artifact_candidate & right,
                                          astc_vulkan_quality_policy policy) {
    if (policy == astc_vulkan_quality_policy::size && left.rate_bpw != right.rate_bpw) {
        return left.rate_bpw < right.rate_bpw;
    }
    if (left.evidence.loss_delta != right.evidence.loss_delta) {
        return left.evidence.loss_delta < right.evidence.loss_delta;
    }
    if (left.evidence.logits_relative_mse != right.evidence.logits_relative_mse) {
        return left.evidence.logits_relative_mse < right.evidence.logits_relative_mse;
    }
    if (left.evidence.top1_agreement != right.evidence.top1_agreement) {
        return left.evidence.top1_agreement > right.evidence.top1_agreement;
    }
    return left.variant < right.variant;
}
