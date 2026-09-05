#pragma once

// D1 frontend for the offline GPU-encoder experiment.
//
// v1 supports only the scalar semantic source: one logical normalized weight
// per physical texel, copied to RGB with opaque alpha. Gauge-L+A, neural
// candidate expansion, and selection remain separate later stages so the
// first Vulkan kernel can be verified against the simplest contract.

#include "astc-gpu-encoder.h"

#include <cstdint>
#include <vector>

bool astc_gpu_d1_build_scalar_source_blocks(
    astc_vulkan_footprint footprint,
    const std::vector<float> & normalized_weights,
    uint32_t rows,
    uint32_t columns,
    std::vector<astc_gpu_encoder_source_block> & blocks);

// Builds the scalar-anchored L+A gauge source family:
//
//   L = q + delta, A = q - delta, (L + A) / 2 = q.
//
// RGB carries luminance L and alpha carries A. The builder deliberately does
// not clamp: caller-selected normalization/headroom is part of the offline
// representation contract, and clipping would silently destroy the null-space
// identity. Selection and exact ASTC encode/decode remain later stages.
bool astc_gpu_d1_build_gauge_la_source_blocks(
    astc_vulkan_footprint footprint,
    const std::vector<float> & normalized_weights,
    const std::vector<float> & gauge_delta,
    uint32_t rows,
    uint32_t columns,
    std::vector<astc_gpu_encoder_source_block> & blocks);
