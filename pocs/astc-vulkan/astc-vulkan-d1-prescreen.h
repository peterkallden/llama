#pragma once

// D1 pre-ASTC footprint screening.
//
// This is the broad first offline pass over an original tensor. It estimates
// whether a footprint can plausibly meet a rate/distortion budget by applying
// a simple per-physical-block scalar quantizer and weighting error by input
// activation energy. It is deliberately a *screen*, not an ASTC oracle:
// astcenc payload construction, scalar/gauge candidates, conflict-aware
// selection, and validation stopping remain the exact second pass.
//
// The metric is separable over physical blocks and therefore maps directly to
// a Vulkan compute batch without requiring sampled ASTC support. This permits
// pre-screening on an arithmetic-only GPU such as the current NVIDIA device.

#include "astc-vulkan-format.h"

#include <cstdint>
#include <vector>

struct astc_vulkan_d1_prescreen_candidate {
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k4x4;
    // Number of uniformly spaced source values. 16 approximates the scalar
    // Q4 source family and 8 the Q3 source family; this is not ASTC endpoint
    // precision and is intentionally never serialized in an artifact.
    uint32_t levels = 16;
};

struct astc_vulkan_d1_prescreen_score {
    astc_vulkan_d1_prescreen_candidate candidate{};
    double weighted_error = 0.0;
    double normalized_error = 0.0;
    double bits_per_weight = 0.0;
};

// `weights` is row-major [rows][columns]. `column_energy` holds the diagonal
// activation sensitivity sum(X[:, column]^2), and has `columns` entries.
// Returns one deterministic score per candidate in input order.
bool astc_vulkan_score_d1_prescreen_cpu(
    const std::vector<float> & weights,
    uint32_t rows,
    uint32_t columns,
    const std::vector<float> & column_energy,
    const std::vector<astc_vulkan_d1_prescreen_candidate> & candidates,
    std::vector<astc_vulkan_d1_prescreen_score> & scores);

// Retains a bounded, rate-aware shortlist for the exact CPU astcenc pass.
// `max_candidates` is a search budget, not a quality decision. The best
// sensitivity score is always retained; other entries are ordered by
// weighted_error + lambda_bits * bits_per_weight.
bool astc_vulkan_select_d1_prescreen_shortlist(
    const std::vector<astc_vulkan_d1_prescreen_score> & scores,
    uint32_t max_candidates,
    double lambda_bits,
    std::vector<astc_vulkan_d1_prescreen_score> & shortlist);
