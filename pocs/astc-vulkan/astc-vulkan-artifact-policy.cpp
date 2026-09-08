#include "astc-vulkan-artifact-policy.h"

#include <algorithm>
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

float astc_vulkan_artifact_p90_loss_delta(
        const astc_vulkan_artifact_evidence & evidence) {
    // v6 artifacts do not have a percentile field. Their existing worst
    // value remains the conservative compatibility fallback until replay is
    // regenerated with v7 provenance.
    if (evidence.replay_case_count > 1 && std::isfinite(evidence.p90_loss_delta)) {
        return evidence.p90_loss_delta;
    }
    return astc_vulkan_artifact_worst_loss_delta(evidence);
}

float astc_vulkan_artifact_robust_loss_score(
        const astc_vulkan_artifact_evidence & evidence,
        const astc_vulkan_artifact_selection_rules & rules) {
    return astc_vulkan_artifact_median_loss_delta(evidence) +
           rules.p90_loss_weight * astc_vulkan_artifact_p90_loss_delta(evidence);
}

double astc_vulkan_artifact_storage_bpw(const astc_vulkan_artifact_record & artifact) {
    const auto & storage = artifact.storage;
    const uint64_t weights = static_cast<uint64_t>(storage.width) * storage.height;
    if (weights == 0) return 0.0;
    const long double bytes = static_cast<long double>(storage.byte_size) +
        storage.layout_byte_size + artifact.row_scale_byte_size + artifact.pair_map_byte_size;
    return static_cast<double>(bytes * 8.0L / static_cast<long double>(weights));
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
    const float p90_loss = astc_vulkan_artifact_p90_loss_delta(evidence);
    const float worst_top1 = astc_vulkan_artifact_worst_top1_agreement(evidence);
    return candidate.tensor != nullptr && device_supports_format && fits_memory_budget &&
           evidence.model_gate_passed && evidence.vulkan_gate_passed &&
           std::isfinite(evidence.loss_delta) && std::isfinite(evidence.logits_relative_mse) &&
           std::isfinite(worst_loss) && std::isfinite(worst_top1) &&
           worst_loss <= rules.max_worst_loss_delta &&
           p90_loss <= rules.max_p90_loss_delta &&
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
    const float left_score = astc_vulkan_artifact_robust_loss_score(left.evidence, rules);
    const float right_score = astc_vulkan_artifact_robust_loss_score(right.evidence, rules);
    if (std::fabs(left_score - right_score) > rules.simplicity_loss_epsilon) {
        return left_score < right_score;
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
    return left_score < right_score;
}

bool astc_vulkan_rank_tensor_artifact_shortlist(
        const std::vector<astc_vulkan_artifact_candidate> & candidates,
        astc_vulkan_quality_policy policy,
        const astc_vulkan_artifact_selection_rules & rules,
        size_t max_entries,
        std::vector<astc_vulkan_artifact_candidate> & shortlist,
        std::string & error) {
    shortlist.clear();
    if (max_entries == 0) {
        error = "artifact shortlist max_entries must be non-zero";
        return false;
    }
    const astc_vulkan_tensor_record * tensor = nullptr;
    for (const auto & candidate : candidates) {
        if (candidate.tensor == nullptr) {
            error = "artifact shortlist contains a null tensor";
            return false;
        }
        if (tensor == nullptr) tensor = candidate.tensor;
        if (candidate.tensor->name != tensor->name) {
            error = "artifact shortlist candidates must belong to one tensor";
            return false;
        }
        if (astc_vulkan_artifact_is_eligible(candidate, true, true, rules)) {
            shortlist.push_back(candidate);
        }
    }
    std::stable_sort(shortlist.begin(), shortlist.end(),
        [&](const auto & left, const auto & right) {
            if (astc_vulkan_artifact_policy_precedes(left, right, policy, rules)) return true;
            if (astc_vulkan_artifact_policy_precedes(right, left, policy, rules)) return false;
            return left.artifact_id < right.artifact_id;
        });
    if (shortlist.size() > max_entries) shortlist.resize(max_entries);
    error.clear();
    return true;
}
