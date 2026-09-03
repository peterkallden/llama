#include "astc-vulkan-paired-selector.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <vector>

namespace {

astc_vulkan_paired_candidate_delta candidate(std::array<uint8_t, 16> payload,
                                              std::vector<double> calibration,
                                              std::vector<double> validation) {
    return {payload, std::move(calibration), std::move(validation)};
}

} // namespace

int main() {
    // Two blocks can both look useful initially, but block 0 removes the
    // residual direction that block 1 was attempting to correct. Candidate 0
    // is the neutral paired-D2 baseline and must remain exactly zero.
    const astc_vulkan_paired_selector_config config{1, 1, 1};
    const std::vector<double> calibration_residual = {2.0};
    const std::vector<double> validation_residual = {1.0};
    std::vector<std::vector<astc_vulkan_paired_candidate_delta>> candidates(2);
    candidates[0].push_back(candidate({}, {0.0}, {0.0}));
    candidates[0].push_back(candidate({1}, {1.5}, {0.75}));
    candidates[1].push_back(candidate({}, {0.0}, {0.0}));
    candidates[1].push_back(candidate({2}, {1.0}, {0.5}));

    astc_vulkan_paired_selection_result result;
    assert(astc_vulkan_select_paired_candidates(config, calibration_residual,
                                                 validation_residual, candidates, result));
    assert(result.commits.size() == 1);
    assert(result.commits[0].block == 0 && result.commits[0].candidate == 1);
    assert(result.calibration_selected_candidates[0] == 1);
    assert(result.calibration_selected_candidates[1] == 0);
    assert(result.validation_prefix == 1);
    assert(result.validation_selected_candidates == result.calibration_selected_candidates);

    // Reject an invalid non-zero neutral baseline explicitly.
    candidates[0][0].calibration_delta[0] = 1.0;
    assert(!astc_vulkan_select_paired_candidates(config, calibration_residual,
                                                  validation_residual, candidates, result));
    std::puts("ASTC Vulkan paired selector contract passed");
    return 0;
}
