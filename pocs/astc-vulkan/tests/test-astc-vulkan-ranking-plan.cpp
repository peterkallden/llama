#include "astc-vulkan-ranking-plan.h"

#include <cassert>
#include <cstdio>
#include <string>

int main() {
    const auto cpu = astc_vulkan_make_cpu_ranking_plan();
    const auto gpu = astc_vulkan_make_gpu_ranking_plan();
    assert(astc_vulkan_validate_ranking_plan(cpu));
    assert(astc_vulkan_validate_ranking_plan(gpu));
    assert(cpu.candidate_encode == astc_vulkan_ranking_backend::kCpu);
    assert(gpu.candidate_encode == astc_vulkan_ranking_backend::kCpu);
    assert(gpu.delta_score == astc_vulkan_ranking_backend::kGpu);
    assert(gpu.conflict_commit == astc_vulkan_ranking_backend::kGpu);
    auto invalid = gpu;
    invalid.candidate_encode = astc_vulkan_ranking_backend::kGpu;
    assert(!astc_vulkan_validate_ranking_plan(invalid));
    invalid = gpu;
    invalid.delta_score = astc_vulkan_ranking_backend::kCpu;
    assert(!astc_vulkan_validate_ranking_plan(invalid));
    assert(std::string(astc_vulkan_ranking_backend_name(astc_vulkan_ranking_backend::kCpu)) == "cpu");
    std::puts("ASTC Vulkan ranking backend-plan contract passed");
    return 0;
}
