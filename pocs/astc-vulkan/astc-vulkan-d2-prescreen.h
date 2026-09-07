#pragma once

// Cheap, deterministic D2 representation screening before CPU astcenc.
//
// The screen operates on logical paired rows and estimates the error of a
// small quantized source model. It is deliberately not an ASTC oracle: exact
// legal payload construction, decode-in-the-loop selection, validation
// stopping and YAQA remain later stages. A GPU backend may mirror this
// contract, but the CPU implementation is the reference.

#include "astc-vulkan-format.h"

#include <cstdint>
#include <vector>

enum class astc_vulkan_d2_prescreen_semantic : uint8_t {
    direct = 0,
    luminance_alpha = 1,
};

enum class astc_vulkan_d2_prescreen_normalization : uint8_t {
    none = 0,
    row_absmax = 1,
};

struct astc_vulkan_d2_prescreen_candidate {
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k8x5;
    astc_vulkan_d2_prescreen_semantic semantic = astc_vulkan_d2_prescreen_semantic::luminance_alpha;
    astc_vulkan_d2_prescreen_normalization normalization =
        astc_vulkan_d2_prescreen_normalization::none;
    // Source-derived Alpha is codec steering only. It has no semantic output
    // contribution, but remains in the candidate identity for later exact
    // encode/selection and deterministic tie breaking.
    bool source_derived_alpha = false;
    uint32_t levels = 16;
};

struct astc_vulkan_d2_prescreen_score {
    astc_vulkan_d2_prescreen_candidate candidate{};
    double weighted_error = 0.0;
    double normalized_error = 0.0;
    double bits_per_weight = 0.0;
    double proxy_cost = 0.0;
};

// `weights` is row-major logical [rows][columns]. D2 consumes adjacent row
// pairs, so rows must be even. `column_energy` is the diagonal activation
// sensitivity sum and has one entry per reduction column.
bool astc_vulkan_score_d2_prescreen_cpu(
    const std::vector<float> & weights,
    uint32_t rows,
    uint32_t columns,
    const std::vector<float> & column_energy,
    const std::vector<astc_vulkan_d2_prescreen_candidate> & candidates,
    std::vector<astc_vulkan_d2_prescreen_score> & scores);

// Builds the diagonal activation-energy proxy from exactly the calibration
// prefix of a trace. Validation and holdout samples are intentionally not
// accepted here, so prescreen provenance cannot accidentally tune on them.
bool astc_vulkan_d2_prescreen_calibration_energy(
    const std::vector<float> & trace,
    uint32_t trace_samples,
    uint32_t columns,
    uint32_t calibration_samples,
    std::vector<float> & column_energy);

// Retains a bounded rate-aware shortlist. Before score fill, selection keeps
// coverage of every available semantic, normalization and source-Alpha value
// whenever the budget permits. This prevents a proxy tie from eliminating an
// exact-ASTC profile family before the CPU finisher can test it.
bool astc_vulkan_select_d2_prescreen_shortlist(
    const std::vector<astc_vulkan_d2_prescreen_score> & scores,
    uint32_t max_candidates,
    double lambda_bits,
    std::vector<astc_vulkan_d2_prescreen_score> & shortlist);
