#include "astc-vulkan-ranking-plan.h"

#include <cassert>
#include <cstdio>
#include <string>

int main() {
    const auto cpu = astc_vulkan_make_cpu_ranking_plan();
    const auto gpu = astc_vulkan_make_gpu_ranking_plan();
    const auto preferred_gpu = astc_vulkan_make_default_ranking_plan(true);
    const auto preferred_cpu = astc_vulkan_make_default_ranking_plan(false);
    const auto yaqa_gpu = astc_vulkan_make_default_ranking_plan(
        true, true, astc_vulkan_objective::two_sided_trace);
    const auto yaqa_cpu = astc_vulkan_make_default_ranking_plan(
        true, false, astc_vulkan_objective::two_sided_trace);
    assert(astc_vulkan_validate_ranking_plan(cpu));
    assert(astc_vulkan_validate_ranking_plan(gpu));
    assert(astc_vulkan_validate_ranking_plan(preferred_gpu));
    assert(astc_vulkan_validate_ranking_plan(preferred_cpu));
    assert(astc_vulkan_validate_ranking_plan(yaqa_gpu));
    assert(astc_vulkan_validate_ranking_plan(yaqa_cpu));
    assert(cpu.candidate_encode == astc_vulkan_ranking_backend::kCpu);
    assert(gpu.candidate_encode == astc_vulkan_ranking_backend::kCpu);
    assert(gpu.delta_score == astc_vulkan_ranking_backend::kGpu);
    assert(gpu.proposal_gain == astc_vulkan_ranking_backend::kGpu);
    assert(gpu.objective == astc_vulkan_ranking_backend::kCpu);
    assert(gpu.conflict_commit == astc_vulkan_ranking_backend::kCpu);
    assert(preferred_gpu.delta_score == astc_vulkan_ranking_backend::kGpu);
    assert(preferred_cpu.delta_score == astc_vulkan_ranking_backend::kCpu);
    assert(yaqa_gpu.objective == astc_vulkan_ranking_backend::kGpu);
    assert(yaqa_cpu.objective == astc_vulkan_ranking_backend::kCpu);
    auto invalid = gpu;
    invalid.candidate_encode = astc_vulkan_ranking_backend::kGpu;
    assert(!astc_vulkan_validate_ranking_plan(invalid));
    invalid = gpu;
    invalid.delta_score = astc_vulkan_ranking_backend::kCpu;
    assert(!astc_vulkan_validate_ranking_plan(invalid));
    invalid = gpu;
    invalid.objective = astc_vulkan_ranking_backend::kGpu;
    assert(!astc_vulkan_validate_ranking_plan(invalid));
    invalid = gpu;
    invalid.conflict_commit = astc_vulkan_ranking_backend::kGpu;
    assert(!astc_vulkan_validate_ranking_plan(invalid));
    invalid = gpu;
    invalid.objective = astc_vulkan_ranking_backend::kGpu;
    invalid.objective_kind = astc_vulkan_objective::activation;
    assert(!astc_vulkan_validate_ranking_plan(invalid));
    assert(std::string(astc_vulkan_ranking_backend_name(astc_vulkan_ranking_backend::kCpu)) == "cpu");
    std::puts("ASTC Vulkan ranking backend-plan contract passed");
    return 0;
}
