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
    assert(astc_vulkan_resolve_user_profile("balanced", defaults));
    assert(defaults.policy == astc_vulkan_quality_policy::balanced &&
           defaults.footprint == astc_vulkan_footprint::k6x6 &&
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

    // Near-equal model evidence should prefer the simpler artifact. This
    // prevents a tiny selected/absmax proxy win from becoming a global
    // runtime rule without a meaningful model-level margin.
    astc_vulkan_artifact_candidate neutral = unscaled;
    neutral.variant = astc_vulkan_artifact_variant::neutral;
    neutral.evidence.loss_delta = 0.0200f;
    astc_vulkan_artifact_candidate selected = neutral;
    selected.variant = astc_vulkan_artifact_variant::validation_selected;
    selected.evidence.loss_delta = 0.0195f;
    assert(astc_vulkan_artifact_policy_precedes(
        neutral, selected, astc_vulkan_quality_policy::quality));
    astc_vulkan_artifact_candidate absmax = selected;
    absmax.normalization = astc_vulkan_normalization::per_row_absmax;
    absmax.evidence.loss_delta = 0.0192f;
    assert(astc_vulkan_artifact_policy_precedes(
        neutral, absmax, astc_vulkan_quality_policy::quality));

    // Multi-prompt evidence gates on the bad tail and ranks by median loss.
    astc_vulkan_artifact_selection_rules robust_rules;
    robust_rules.max_worst_loss_delta = 0.05f;
    robust_rules.min_worst_top1_agreement = 0.90f;
    selected.evidence.model_gate_passed = true;
    selected.evidence.replay_case_count = 3;
    selected.evidence.median_loss_delta = 0.010f;
    selected.evidence.worst_loss_delta = 0.080f;
    selected.evidence.worst_top1_agreement = 0.95f;
    assert(!astc_vulkan_artifact_is_eligible(selected, true, true, robust_rules));
    selected.evidence.worst_loss_delta = 0.030f;
    selected.evidence.p90_loss_delta = 0.020f;
    assert(astc_vulkan_artifact_is_eligible(selected, true, true, robust_rules));
    neutral.evidence.replay_case_count = 3;
    neutral.evidence.median_loss_delta = 0.012f;
    neutral.evidence.p90_loss_delta = 0.015f;
    neutral.evidence.worst_loss_delta = 0.020f;
    neutral.evidence.worst_top1_agreement = 1.0f;
    assert(astc_vulkan_artifact_policy_precedes(
        selected, neutral, astc_vulkan_quality_policy::quality, robust_rules));

    // P90 ranks ordinary prompt-tail behaviour; max remains a separate hard
    // catastrophe gate. A slightly better median is not enough when its P90
    // is materially worse.
    astc_vulkan_artifact_candidate p90_safe = neutral;
    p90_safe.artifact_id = "d2-la-neutral";
    p90_safe.evidence.median_loss_delta = .011f;
    p90_safe.evidence.p90_loss_delta = .014f;
    astc_vulkan_artifact_candidate p90_risky = selected;
    p90_risky.artifact_id = "d2-la-selected";
    p90_risky.evidence.median_loss_delta = .009f;
    p90_risky.evidence.p90_loss_delta = .030f;
    assert(astc_vulkan_artifact_policy_precedes(
        p90_safe, p90_risky, astc_vulkan_quality_policy::quality, robust_rules));

    // The per-tensor bank is representation-neutral. It keeps the two best
    // evidence-backed ASTC alternatives; native remains an implicit fallback.
    astc_vulkan_artifact_candidate d1 = p90_safe;
    d1.artifact_id = "d1-10x8";
    d1.rate_bpw = 1.6;
    d1.evidence.median_loss_delta = .013f;
    d1.evidence.p90_loss_delta = .016f;
    std::vector<astc_vulkan_artifact_candidate> shortlist;
    std::string error;
    assert(astc_vulkan_rank_tensor_artifact_shortlist(
        {p90_risky, d1, p90_safe}, astc_vulkan_quality_policy::quality,
        robust_rules, 2, shortlist, error));
    assert(shortlist.size() == 2);
    assert(shortlist[0].artifact_id == "d2-la-neutral");
    assert(shortlist[1].artifact_id == "d1-10x8");
    return 0;
}
