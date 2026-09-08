#pragma once

#include <cstdint>
#include <limits>
#include <string>

// Artifact identity is deliberately separate from storage identity. An
// artifact is one immutable, offline-validated encoding candidate for a
// tensor; several artifacts may share the same tensor name and footprint.
// These types are serialized by manifest v4 and used by the scheduler policy.

enum class astc_vulkan_artifact_variant : uint8_t {
    neutral = 0,
    validation_selected = 1,
};

enum class astc_vulkan_normalization : uint8_t {
    none = 0,
    per_row_absmax = 1,
};

enum class astc_vulkan_paired_semantic : uint8_t {
    direct_rgb = 0,
    luminance_alpha = 1,
};

struct astc_vulkan_artifact_evidence {
    bool model_gate_passed = false;
    bool vulkan_gate_passed = false;
    float activation_mse = 0.0f;
    float logits_relative_mse = 0.0f;
    float loss_delta = 0.0f;
    float top1_agreement = 0.0f;
    std::string calibration_validation_hash;
    std::string replay_corpus_hash;

    // v6 robust model-replay summary. Legacy v4/v5 artifacts leave
    // replay_case_count at zero and use loss_delta/top1_agreement exactly as
    // before.  Once several disjoint prompts/traces have been replayed, the
    // offline selector gates on the tail and ranks by the median instead of
    // allowing one favourable average to hide a bad prompt.
    uint32_t replay_case_count = 0;
    float median_loss_delta = 0.0f;
    // P90 is the normal robustness term used for ranking. The maximum stays
    // separate as a catastrophe detector; it must not dominate a small replay
    // corpus by itself. NaN means that a legacy artifact has no percentile.
    float p90_loss_delta = std::numeric_limits<float>::quiet_NaN();
    float worst_loss_delta = 0.0f;
    float worst_top1_agreement = 0.0f;

    // Split provenance is deliberately separate. In particular, a pair map
    // and a validation-selected commit prefix must not be promoted using the
    // same evidence split that generated them. Legacy artifacts retain the
    // combined calibration_validation_hash and replay_corpus_hash above.
    std::string calibration_hash;
    std::string validation_hash;
    std::string artifact_holdout_hash;
    std::string final_model_holdout_hash;
};
