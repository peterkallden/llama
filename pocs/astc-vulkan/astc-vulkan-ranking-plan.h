#pragma once

#include "astc-vulkan-objective.h"

// Offline candidate-ranking backend plan.
//
// ASTC payload generation is deliberately CPU-only: normal Vulkan exposes
// sampled ASTC decode, not ASTC encode. All later stages have compatible CPU
// and GPU contracts, allowing a future policy to select a device-specific
// default without changing the payload or artifact format.

enum class astc_vulkan_ranking_backend : unsigned char {
    kCpu,
    kGpu,
};

struct astc_vulkan_ranking_plan {
    // CPU astcenc generates legal standard ASTC payloads in every plan.
    astc_vulkan_ranking_backend candidate_encode = astc_vulkan_ranking_backend::kCpu;

    // Exact decode + D2 reconstruction relative to candidate zero.
    astc_vulkan_ranking_backend delta_score = astc_vulkan_ranking_backend::kCpu;

    // Activation/YAQA candidate objective over exact decoded candidates.
    // This remains CPU until a full model-objective kernel is validated.
    astc_vulkan_ranking_backend objective = astc_vulkan_ranking_backend::kCpu;
    astc_vulkan_objective objective_kind = astc_vulkan_objective::activation;

    // Parallel proposal-gain calculation against the current strip residual.
    astc_vulkan_ranking_backend proposal_gain = astc_vulkan_ranking_backend::kCpu;

    // Conflict-aware residual commit. The current GPU plan leaves this
    // ordered host stage; only proposal gains are device-side today.
    astc_vulkan_ranking_backend conflict_commit = astc_vulkan_ranking_backend::kCpu;
};

astc_vulkan_ranking_plan astc_vulkan_make_cpu_ranking_plan();
astc_vulkan_ranking_plan astc_vulkan_make_gpu_ranking_plan();
// Select the verified fast path after the caller's device/SPIR-V preflight.
// This does not perform Vulkan probing itself; unsupported devices stay CPU.
astc_vulkan_ranking_plan astc_vulkan_make_default_ranking_plan(bool gpu_preflight_passed);
// YAQA is selected on GPU only when the separate YAQA batch preflight passed.
// Other objectives continue to use the ordinary semantic ranking path.
astc_vulkan_ranking_plan astc_vulkan_make_default_ranking_plan(
        bool gpu_preflight_passed, bool yaqa_gpu_preflight_passed,
        astc_vulkan_objective objective);
bool astc_vulkan_validate_ranking_plan(const astc_vulkan_ranking_plan & plan);
const char * astc_vulkan_ranking_backend_name(astc_vulkan_ranking_backend backend);
