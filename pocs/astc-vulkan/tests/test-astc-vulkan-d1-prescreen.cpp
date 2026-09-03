#include "astc-vulkan-d1-prescreen.h"

#include <cassert>
#include <cmath>

int main() {
    // Two row blocks have different ranges; the per-physical-block quantizer
    // must not incorrectly share endpoints between them.
    const std::vector<float> weights{
        0.0f, 0.2f, 0.4f, 0.6f, 10.0f, 10.2f, 10.4f, 10.6f,
        0.1f, 0.3f, 0.5f, 0.7f, 10.1f, 10.3f, 10.5f, 10.7f,
        0.0f, 0.2f, 0.4f, 0.6f, 10.0f, 10.2f, 10.4f, 10.6f,
        0.1f, 0.3f, 0.5f, 0.7f, 10.1f, 10.3f, 10.5f, 10.7f};
    const std::vector<float> energy{1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    const std::vector<astc_vulkan_d1_prescreen_candidate> candidates{
        {astc_vulkan_footprint::k4x4, 16}, {astc_vulkan_footprint::k8x6, 16},
        {astc_vulkan_footprint::k10x8, 8}};
    std::vector<astc_vulkan_d1_prescreen_score> scores;
    assert(astc_vulkan_score_d1_prescreen_cpu(weights, 4, 8, energy, candidates, scores));
    assert(scores.size() == candidates.size());
    assert(std::abs(scores[0].bits_per_weight - 8.0) < 1e-12);
    assert(std::abs(scores[1].bits_per_weight - (128.0 / 48.0)) < 1e-12);
    assert(scores[2].normalized_error >= 0.0);
    std::vector<astc_vulkan_d1_prescreen_score> shortlist;
    assert(astc_vulkan_select_d1_prescreen_shortlist(scores, 2, 1.0, shortlist));
    assert(shortlist.size() == 2);
    assert(!astc_vulkan_score_d1_prescreen_cpu(weights, 4, 8, energy,
        {{astc_vulkan_footprint::k4x4, 1}}, scores));
    return 0;
}
