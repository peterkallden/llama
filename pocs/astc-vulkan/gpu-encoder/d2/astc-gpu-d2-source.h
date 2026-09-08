#pragma once

// D2 frontend for the offline GPU ASTC-encoder experiment.
//
// D2 maps two logical output rows to one physical ASTC texel. This module owns
// paired geometry, optional pair maps, Givens source transforms, and the
// per-physical-block RG/B versus R/GB interpretation. The shared proposer
// receives only resulting RGBA texels.
//
// Suitable for: isolated low-rate D2 source construction (initially 8x5).
// Not a runtime/cache producer: CPU astcenc remains the legal payload oracle.

#include "astc-gpu-encoder.h"
#include "astc-vulkan-d2-pair-transform.h"
#include "astc-vulkan-paired.h"

#include <cstdint>
#include <vector>

// Returns a deterministic direct-D2 layout map with RG/B for every physical
// ASTC block. A later offline selector may replace entries per block.
bool astc_gpu_d2_make_uniform_layout_map(astc_vulkan_footprint footprint,
                                         uint32_t logical_rows,
                                         uint32_t logical_columns,
                                         astc_vulkan_paired_layout layout,
                                         std::vector<astc_vulkan_paired_layout> & layouts);

// Expands the serialized cache pair-map into one pairing per physical ASTC
// block. Empty input selects adjacent rows. A non-empty map has one 10-byte
// pairing per five-physical-row D2 group.
bool astc_gpu_d2_expand_pair_map(astc_vulkan_footprint footprint,
                                 uint32_t logical_rows,
                                 uint32_t logical_columns,
                                 const std::vector<uint8_t> & pair_map,
                                 std::vector<astc_vulkan_d2_pairing> & pairings);

// Builds physical RGBA source blocks from normalized [0,1] logical weights.
// `layouts` has one entry per physical ASTC block in raster order. An empty
// `steering_texels` selects neutral Alpha=0.5; otherwise it contains one
// normalized steering value per unpadded paired texture texel
// `[ceil(rows/2)][columns]`. Luminance+Alpha ignores steering by definition.
// Odd logical rows and image edges use deterministic q=0.5 padding.
// `pairings`, when supplied, has one entry per physical ASTC block in raster
// order. `transform` rotates centered row pairs and maps them through a fixed
// sqrt(2) bound, so all source values remain in [0,1] without per-block scale
// metadata. Exact rank/selector helpers apply the inverse transform.
bool astc_gpu_d2_build_paired_source_blocks(
    astc_vulkan_footprint footprint,
    const std::vector<float> & normalized_weights,
    uint32_t logical_rows,
    uint32_t logical_columns,
    const std::vector<astc_vulkan_paired_layout> & layouts,
    const std::vector<float> & steering_texels,
    astc_vulkan_paired_semantic semantic,
    std::vector<astc_gpu_encoder_source_block> & blocks,
    const std::vector<astc_vulkan_d2_pairing> & pairings = {},
    astc_vulkan_d2_givens_transform transform = {});
