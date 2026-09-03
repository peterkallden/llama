#pragma once

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
    astc_vulkan_ranking_backend objective = astc_vulkan_ranking_backend::kCpu;

    // Parallel proposal-gain calculation against the current strip residual.
    astc_vulkan_ranking_backend proposal_gain = astc_vulkan_ranking_backend::kCpu;

    // Conflict-aware residual commit. This remains ordered within a strip,
    // whether its implementation lives on CPU or GPU.
    astc_vulkan_ranking_backend conflict_commit = astc_vulkan_ranking_backend::kCpu;
};

astc_vulkan_ranking_plan astc_vulkan_make_cpu_ranking_plan();
astc_vulkan_ranking_plan astc_vulkan_make_gpu_ranking_plan();
bool astc_vulkan_validate_ranking_plan(const astc_vulkan_ranking_plan & plan);
const char * astc_vulkan_ranking_backend_name(astc_vulkan_ranking_backend backend);

