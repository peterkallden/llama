#pragma once

#include "astc-vulkan-format.h"
#include "astc-vulkan-paired.h"

#include <cstdint>
#include <vector>

// Packed per-physical-ASTC-block semantic layout map for paired-D2 artifacts.
//
// ASTC payload bytes do not encode whether a texel is interpreted as RG/B or
// R/GB. D2 therefore carries one layout bit per physical ASTC block in a
// separate little-endian uint32 word stream. Bit zero is the first block in
// raster order; zero denotes RG/B and one denotes R/GB. A 32-bit word stream
// is chosen over byte-at-a-time metadata so Vulkan can read it directly from a
// storage buffer without unpack/copy staging.
//
// Suitable for: standard five-row paired-D2 artifacts (8x5 and 10x5). This is offline/runtime
// metadata only; it neither changes ASTC bytes nor applies a decoder. D1 and
// mixed-footprint page metadata are intentionally separate concerns.

uint32_t astc_vulkan_paired_storage_height(uint32_t logical_height);
uint64_t astc_vulkan_paired_block_count(astc_vulkan_footprint footprint,
                                         uint32_t logical_width,
                                         uint32_t logical_height);
uint64_t astc_vulkan_paired_layout_word_count(astc_vulkan_footprint footprint,
                                               uint32_t logical_width,
                                               uint32_t logical_height);
uint64_t astc_vulkan_paired_layout_bytes(astc_vulkan_footprint footprint,
                                         uint32_t logical_width,
                                         uint32_t logical_height);

bool astc_vulkan_paired_layout_get(const std::vector<uint32_t> & words,
                                   uint64_t block_index,
                                   astc_vulkan_paired_layout & layout);
bool astc_vulkan_paired_layout_set(std::vector<uint32_t> & words,
                                   uint64_t block_index,
                                   astc_vulkan_paired_layout layout);
