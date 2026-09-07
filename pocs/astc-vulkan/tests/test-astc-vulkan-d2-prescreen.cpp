#include "astc-vulkan-d2-prescreen.h"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
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
         astc_vulkan_d2_prescreen_normalization::row_absmax, true, 8},
    };
    std::vector<astc_vulkan_d2_prescreen_score> scores;
    assert(astc_vulkan_score_d2_prescreen_cpu(weights, 4, 8, energy, candidates, scores));
    assert(scores.size() == candidates.size());
    for (const auto & score : scores) {
        assert(std::isfinite(score.weighted_error));
        assert(std::isfinite(score.normalized_error));
        assert(std::isfinite(score.bits_per_weight));
    }
    // D2 reports logical model density: each physical ASTC texel carries two
    // logical paired-row weights.
    assert(std::abs(scores[0].bits_per_weight - (128.0 / 60.0)) < 1e-12);
    assert(std::abs(scores[1].bits_per_weight - 1.6) < 1e-12);
    assert(std::abs(scores[2].bits_per_weight - 1.28) < 1e-12);
    std::vector<astc_vulkan_d2_prescreen_score> shortlist;
    assert(astc_vulkan_select_d2_prescreen_shortlist(scores, 2, 0.0, shortlist));
    assert(shortlist.size() == 2);
    bool has_la = false, has_direct = false, has_scaled = false, has_unscaled = false;
    for (const auto & score : shortlist) {
        has_la |= score.candidate.semantic == astc_vulkan_d2_prescreen_semantic::luminance_alpha;
        has_direct |= score.candidate.semantic == astc_vulkan_d2_prescreen_semantic::direct;
        has_scaled |= score.candidate.normalization == astc_vulkan_d2_prescreen_normalization::row_absmax;
        has_unscaled |= score.candidate.normalization == astc_vulkan_d2_prescreen_normalization::none;
    }
    assert(has_la && has_direct && has_scaled && has_unscaled);
    const std::vector<float> trace{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f,
    };
    std::vector<float> calibration_energy;
    assert(astc_vulkan_d2_prescreen_calibration_energy(trace, 3, 8, 2, calibration_energy));
    assert(calibration_energy[0] == 5.0f);
    assert(!astc_vulkan_d2_prescreen_calibration_energy(trace, 3, 8, 4, calibration_energy));
    assert(!astc_vulkan_score_d2_prescreen_cpu(weights, 3, 8, energy, candidates, scores));
    return 0;
}
