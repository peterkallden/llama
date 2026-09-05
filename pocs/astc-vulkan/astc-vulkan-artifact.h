#pragma once

#include <cstdint>
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
};

