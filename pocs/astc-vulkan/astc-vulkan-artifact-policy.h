#pragma once

#include "astc-vulkan-artifact.h"
#include "astc-vulkan-manifest.h"

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

// User-facing profile spelling. Runtime selection still operates only on
// evidence-backed artifacts; this parser does not inspect model weights.
bool astc_vulkan_parse_quality_policy(const std::string & name,
                                      astc_vulkan_quality_policy & policy);
const char * astc_vulkan_quality_policy_name(astc_vulkan_quality_policy policy);

struct astc_vulkan_artifact_candidate {
    const astc_vulkan_tensor_record * tensor = nullptr;
    astc_vulkan_artifact_variant variant = astc_vulkan_artifact_variant::neutral;
    astc_vulkan_normalization normalization = astc_vulkan_normalization::none;
    astc_vulkan_artifact_evidence evidence{};
    double rate_bpw = 0.0;
};

bool astc_vulkan_artifact_is_eligible(const astc_vulkan_artifact_candidate & candidate,
                                      bool device_supports_format, bool fits_memory_budget);

// Returns true if `left` ranks ahead of `right` after eligibility filtering.
// Loss is primary for quality/balanced/speed/automatic. Size (and the
// backward-compatible compact spelling) chooses lower rate first. Speed and
// automatic are deterministic placeholders until measured device timings are
// added to artifact evidence; they intentionally do not guess that lower rate
// means faster inference.
bool astc_vulkan_artifact_policy_precedes(const astc_vulkan_artifact_candidate & left,
                                          const astc_vulkan_artifact_candidate & right,
                                          astc_vulkan_quality_policy policy);
