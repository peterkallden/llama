#include "astc-vulkan-artifact-policy.h"

#include <cassert>
#include <string>

int main() {
    astc_vulkan_quality_policy parsed;
    assert(astc_vulkan_parse_quality_policy("quality", parsed) &&
           parsed == astc_vulkan_quality_policy::quality);
    assert(astc_vulkan_parse_quality_policy("compact", parsed) &&
           parsed == astc_vulkan_quality_policy::size);
    assert(astc_vulkan_parse_quality_policy("auto", parsed) &&
           parsed == astc_vulkan_quality_policy::automatic);
    assert(!astc_vulkan_parse_quality_policy("unknown", parsed));
    assert(std::string(astc_vulkan_quality_policy_name(astc_vulkan_quality_policy::speed)) == "speed");

    astc_vulkan_user_profile_defaults defaults;
    assert(astc_vulkan_resolve_user_profile("quality", defaults));
    assert(defaults.policy == astc_vulkan_quality_policy::quality &&
           defaults.footprint == astc_vulkan_footprint::k4x4 &&
           defaults.representation == astc_vulkan_representation::kScalar);
    assert(astc_vulkan_resolve_user_profile("compact", defaults));
    assert(defaults.policy == astc_vulkan_quality_policy::size &&
           defaults.footprint == astc_vulkan_footprint::k8x5 &&
           defaults.representation == astc_vulkan_representation::kPairedD2);
    assert(astc_vulkan_resolve_user_profile("auto", defaults));
    assert(defaults.policy == astc_vulkan_quality_policy::automatic);
    assert(!astc_vulkan_resolve_user_profile("d2-8x5", defaults));

    astc_vulkan_tensor_record tensor;
    tensor.name = "blk.0.ffn_down.weight";
    astc_vulkan_artifact_candidate unscaled{&tensor, astc_vulkan_artifact_variant::validation_selected,
        astc_vulkan_normalization::none, {true, true, 0.0f, 0.015f, 0.024f, 1.0f}, 1.6};
    astc_vulkan_artifact_candidate scaled{&tensor, astc_vulkan_artifact_variant::validation_selected,
        astc_vulkan_normalization::per_row_absmax, {true, true, 0.0f, 0.007f, 0.055f, 1.0f}, 1.602};
    assert(astc_vulkan_artifact_is_eligible(unscaled, true, true));
    assert(astc_vulkan_artifact_policy_precedes(unscaled, scaled, astc_vulkan_quality_policy::quality));
    assert(astc_vulkan_artifact_policy_precedes(unscaled, scaled, astc_vulkan_quality_policy::balanced));
    assert(astc_vulkan_artifact_policy_precedes(unscaled, scaled, astc_vulkan_quality_policy::size));
    scaled.rate_bpw = 1.5;
    assert(astc_vulkan_artifact_policy_precedes(scaled, unscaled, astc_vulkan_quality_policy::compact));
    scaled.evidence.model_gate_passed = false;
    assert(!astc_vulkan_artifact_is_eligible(scaled, true, true));
    return 0;
}
