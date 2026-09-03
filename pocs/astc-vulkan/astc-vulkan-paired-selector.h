#pragma once

#include <array>
#include <cstdint>
#include <vector>

// Offline conflict-aware selection for paired-D2 ASTC candidates.
//
// Candidate producers must first run the real ASTC encoder and decoder, then
// provide each legal 16-byte payload's output delta relative to the scalar
// fallback. This keeps ASTC's fixed-function decode as the only runtime codec
// decoder: this module operates solely on offline activation-space vectors.
//
// Candidate zero in every block is mandatory and represents the exact
// scalar-compatible fallback. Selection greedily commits only positive
// calibration gains, updating the residual after every commit. An optional
// independent validation residual then chooses an exported commit prefix.
//
// Suitable for: experimental paired-D2 low-rate artifacts. It deliberately
// has no Vulkan, ggml, astcenc, or full-PV dependency, so candidate generation,
// ASTC mode search, and later weighted/YAQA shortlist ranking stay separable.
//
// References: Frantar et al., GPTQ, https://arxiv.org/abs/2210.17323;
// Tseng, Sun, De Sa, Model-Preserving Adaptive Rounding,
// https://arxiv.org/abs/2505.22988.

struct astc_vulkan_paired_candidate_delta {
    // Standard ASTC block selected offline. Its bytes are retained here so an
    // artifact writer can materialize a validation-selected prefix directly.
    std::array<uint8_t, 16> payload{};

    // Candidate output minus scalar-fallback output, in row-major
    // [sample][logical-output-row] order. The first candidate of each block
    // must be identically zero for both vectors.
    std::vector<double> calibration_delta;
    std::vector<double> validation_delta;
};

struct astc_vulkan_paired_selector_config {
    uint32_t logical_output_rows = 0;
    uint32_t calibration_samples = 0;
    uint32_t validation_samples = 0;
};

struct astc_vulkan_paired_selection_commit {
    uint32_t block = 0;
    uint32_t candidate = 0;
    double calibration_gain = 0.0;
};

struct astc_vulkan_paired_selection_result {
    // Selection after all positive calibration commits, primarily useful for
    // diagnostics. `validation_selected_candidates` is the exportable result.
    std::vector<uint32_t> calibration_selected_candidates;
    std::vector<uint32_t> validation_selected_candidates;
    std::vector<astc_vulkan_paired_selection_commit> commits;
    uint32_t validation_prefix = 0;
    double calibration_residual_loss = 0.0;
    double validation_residual_loss = 0.0;
};

// Selects from a scalar-anchored block candidate pool. `initial_*_residual`
// are reference output minus scalar-fallback output. `validation` can be
// disabled by passing an empty residual and zero validation samples; then every
// positive calibration commit is exported. Returns false on malformed shapes,
// non-finite values, or a pool without the mandatory zero fallback.
bool astc_vulkan_select_paired_candidates(
    const astc_vulkan_paired_selector_config & config,
    const std::vector<double> & initial_calibration_residual,
    const std::vector<double> & initial_validation_residual,
    const std::vector<std::vector<astc_vulkan_paired_candidate_delta>> & candidates,
    astc_vulkan_paired_selection_result & result);
