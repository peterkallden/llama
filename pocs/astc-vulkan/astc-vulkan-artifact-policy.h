#pragma once

#include "astc-vulkan-artifact.h"
#include "astc-vulkan-manifest.h"

#include <limits>
#include <string>

// Offline evidence and runtime policy contract for one concrete ASTC artifact.
// The scheduler selects only already validated artifacts; it never runs an
// encoder or infers neutral/selected at runtime. This module is intentionally
// independent from Vulkan resource ownership and cache serialization.

enum class astc_vulkan_quality_policy : unsigned char {
    quality,
    balanced,
    size,
    compact = size, // Backward-compatible name used by existing callers.
    speed,
    automatic,
};

// High-level user intent used by cache discovery and planning.  The resolver
// supplies a conservative first-pass physical candidate; it does not claim
// that this candidate is model-approved.  Artifact evidence remains the
// authority for production selection.
struct astc_vulkan_user_profile_defaults {
    astc_vulkan_quality_policy policy = astc_vulkan_quality_policy::balanced;
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k6x6;
    astc_vulkan_representation representation = astc_vulkan_representation::kScalar;
};

// User-facing profile spelling. Runtime selection still operates only on
// evidence-backed artifacts; this parser does not inspect model weights.
bool astc_vulkan_parse_quality_policy(const std::string & name,
                                      astc_vulkan_quality_policy & policy);
const char * astc_vulkan_quality_policy_name(astc_vulkan_quality_policy policy);

// Resolve the user-facing profile without requiring a D1/D2 spelling.  The
// explicit --footprint/--representation options can override these defaults
// for reproducible research runs.
bool astc_vulkan_resolve_user_profile(const std::string & name,
                                      astc_vulkan_user_profile_defaults & defaults);

struct astc_vulkan_artifact_candidate {
    const astc_vulkan_tensor_record * tensor = nullptr;
    astc_vulkan_artifact_variant variant = astc_vulkan_artifact_variant::neutral;
    astc_vulkan_normalization normalization = astc_vulkan_normalization::none;
    astc_vulkan_artifact_evidence evidence{};
    double rate_bpw = 0.0;
};

// Offline-only selection rules. They deliberately describe evidence, not
// codec mechanics: runtime receives one already-approved artifact and never
// guesses neutral/selected or a normalization from model weights.
struct astc_vulkan_artifact_selection_rules {
    // Disabled by default. Set these when an artifact carries multi-prompt
    // replay evidence and a caller has a concrete quality budget.
    float max_worst_loss_delta = std::numeric_limits<float>::infinity();
    float min_worst_top1_agreement = 0.0f;

    // Prefer the less trace-dependent artifact when robust loss is within
    // this tolerance. This means unscaled before absmax, and neutral before
    // validation-selected. It is intentionally small and can be set to zero
    // for strict numerical ranking.
    float simplicity_loss_epsilon = 0.001f;
};

bool astc_vulkan_artifact_has_robust_replay_evidence(
    const astc_vulkan_artifact_evidence & evidence);
float astc_vulkan_artifact_median_loss_delta(
    const astc_vulkan_artifact_evidence & evidence);
float astc_vulkan_artifact_worst_loss_delta(
    const astc_vulkan_artifact_evidence & evidence);
float astc_vulkan_artifact_worst_top1_agreement(
    const astc_vulkan_artifact_evidence & evidence);

bool astc_vulkan_artifact_is_eligible(const astc_vulkan_artifact_candidate & candidate,
                                      bool device_supports_format, bool fits_memory_budget);
bool astc_vulkan_artifact_is_eligible(const astc_vulkan_artifact_candidate & candidate,
                                      bool device_supports_format, bool fits_memory_budget,
                                      const astc_vulkan_artifact_selection_rules & rules);

// Returns true if `left` ranks ahead of `right` after eligibility filtering.
// Loss is primary for quality/balanced/speed/automatic. Size (and the
// backward-compatible compact spelling) chooses lower rate first. Speed and
// automatic are deterministic placeholders until measured device timings are
// added to artifact evidence; they intentionally do not guess that lower rate
// means faster inference.
bool astc_vulkan_artifact_policy_precedes(const astc_vulkan_artifact_candidate & left,
                                          const astc_vulkan_artifact_candidate & right,
                                          astc_vulkan_quality_policy policy);
bool astc_vulkan_artifact_policy_precedes(const astc_vulkan_artifact_candidate & left,
                                          const astc_vulkan_artifact_candidate & right,
                                          astc_vulkan_quality_policy policy,
                                          const astc_vulkan_artifact_selection_rules & rules);
