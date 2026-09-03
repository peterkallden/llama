#include "astc-vulkan-ranking-plan.h"

astc_vulkan_ranking_plan astc_vulkan_make_cpu_ranking_plan() {
    return {};
}

astc_vulkan_ranking_plan astc_vulkan_make_gpu_ranking_plan() {
    astc_vulkan_ranking_plan result;
    result.delta_score = astc_vulkan_ranking_backend::kGpu;
    result.objective = astc_vulkan_ranking_backend::kGpu;
    result.proposal_gain = astc_vulkan_ranking_backend::kGpu;
    result.conflict_commit = astc_vulkan_ranking_backend::kGpu;
    return result;
}

bool astc_vulkan_validate_ranking_plan(const astc_vulkan_ranking_plan & plan) {
    // Standard Vulkan has no general ASTC encoding operation.
    if (plan.candidate_encode != astc_vulkan_ranking_backend::kCpu) return false;
    // A GPU proposal/commit phase retains its deltas and residual on device;
    // allowing a CPU delta producer here would silently introduce readbacks.
    if (plan.proposal_gain == astc_vulkan_ranking_backend::kGpu &&
        plan.delta_score != astc_vulkan_ranking_backend::kGpu) return false;
    if (plan.conflict_commit == astc_vulkan_ranking_backend::kGpu &&
        plan.proposal_gain != astc_vulkan_ranking_backend::kGpu) return false;
    return true;
}

const char * astc_vulkan_ranking_backend_name(astc_vulkan_ranking_backend backend) {
    switch (backend) {
        case astc_vulkan_ranking_backend::kCpu: return "cpu";
        case astc_vulkan_ranking_backend::kGpu: return "gpu";
    }
    return "unknown";
}
