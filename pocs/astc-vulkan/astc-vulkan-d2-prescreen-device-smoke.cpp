#include "astc-vulkan-d2-prescreen-dispatch.h"

#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::vector<float> weights{
        1.0f, 0.5f, -0.2f, 0.1f, 0.7f, -0.4f, 0.3f, 0.2f,
        0.9f, 0.4f, -0.1f, 0.0f, 0.6f, -0.3f, 0.2f, 0.1f,
        -0.5f, 0.2f, 0.1f, 0.8f, -0.2f, 0.3f, 0.4f, -0.1f,
        -0.4f, 0.1f, 0.2f, 0.7f, -0.1f, 0.2f, 0.3f, 0.0f,
    };
    const std::vector<float> energy(8, 1.0f);
    const std::vector<astc_vulkan_d2_prescreen_candidate> candidates{
        {astc_vulkan_footprint::k6x5, astc_vulkan_d2_prescreen_semantic::luminance_alpha,
         astc_vulkan_d2_prescreen_normalization::none, false, 16},
        {astc_vulkan_footprint::k8x5, astc_vulkan_d2_prescreen_semantic::luminance_alpha,
         astc_vulkan_d2_prescreen_normalization::row_absmax, true, 16},
        {astc_vulkan_footprint::k10x5, astc_vulkan_d2_prescreen_semantic::direct,
         astc_vulkan_d2_prescreen_normalization::none, false, 8},
    };
    std::vector<astc_vulkan_d2_prescreen_score> scores;
    std::string error;
    if (!astc_vulkan_score_d2_prescreen_gpu_default(argv[1], weights, 4, 8,
                                                     energy, candidates, scores, error)) {
        std::fprintf(stderr, "D2 GPU pre-screen unavailable: %s\n", error.c_str());
        // Device absence is an optional-backend skip; numerical/API failures
        // remain visible as test failures through a non-77 status.
        return error.find("device") != std::string::npos ||
               error.find("Vulkan") != std::string::npos ? 77 : 1;
    }
    if (scores.size() != candidates.size()) return 1;
    std::printf("D2 GPU pre-screen smoke passed (%zu candidates)\n", scores.size());
    return 0;
}
