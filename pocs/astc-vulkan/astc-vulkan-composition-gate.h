#pragma once

#include "astc-vulkan-artifact-policy.h"

#include <string>
#include <vector>

// Offline global composition gate.
//
// Per-tensor artifact selection answers which candidates are locally worthy of
// replay. This module consumes *global* model-replay observations for an
// ordered sequence of those candidates. Every observation must represent the
// model with all earlier accepted artifacts plus the trial artifact. It never
// runs a model or makes a runtime scheduler decision.

struct astc_vulkan_composition_trial {
    std::string artifact_id;
    astc_vulkan_artifact_candidate candidate{};
};

struct astc_vulkan_composition_stage {
    std::string tensor_name;
    // Ordered offline alternatives: usually local winner, local runner-up,
    // then a simpler neutral artifact. Each carries the global replay result
    // for this exact prefix state.
    std::vector<astc_vulkan_composition_trial> trials;
};

struct astc_vulkan_composition_options {
    astc_vulkan_artifact_selection_rules quality_rules{};
};

struct astc_vulkan_composition_decision {
    std::string tensor_name;
    std::string artifact_id;
    bool use_native_fallback = false;
    std::string reason;
};

struct astc_vulkan_composition_result {
    std::vector<astc_vulkan_composition_decision> decisions;
};

// Applies the ordered, coordinate-descent-like admission gate. A failed trial
// does not terminate the sweep: the next precomputed alternative is tried for
// the same tensor. If none satisfy the global replay budget, the tensor stays
// native. The caller owns the expensive replay loop that produces `trials`.
bool astc_vulkan_select_composed_artifacts(
    const std::vector<astc_vulkan_composition_stage> & stages,
    const astc_vulkan_composition_options & options,
    astc_vulkan_composition_result & result,
    std::string & error);
