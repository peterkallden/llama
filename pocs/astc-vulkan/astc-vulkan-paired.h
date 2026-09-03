#pragma once

#include "astc-vulkan-contract.h"

// Experimental D2 (two logical weights per ASTC texel) semantic contract.
//
// D2 packs a pair of logical weights from adjacent output rows at one input
// column into the RGB lanes of one ordinary RGBA ASTC texel. Alpha is a pure
// codec-steering lane: the runtime decoder deliberately ignores it. This is
// distinct from the D1 L+A gauge family, whose semantic decoder combines two
// latent values into one weight.
//
// Suitable for: offline low-rate research at roughly <= 2 nominal logical
// bits/weight, where a D1 footprint has become too coarse. D2 is experimental
// until exact ASTC roundtrip, validation-selected artifact replay, and the
// paired Vulkan matvec agree. It must retain D1 as a scheduler fallback.
//
// Reference: Khronos ASTC Data Format Specification, and the project's
// ASTC-Vulkan research note. This module only defines the semantic mapping;
// it neither encodes nor decodes ASTC blocks.

enum class astc_vulkan_semantic_density : unsigned char {
    d1_scalar = 1,
    d2_paired = 2,
};

enum class astc_vulkan_paired_layout : unsigned char {
    // q0 is duplicated in R/G; q1 is stored in B.
    rg_b = 0,
    // q0 is stored in R; q1 is duplicated in G/B.
    r_gb = 1,
};

struct astc_vulkan_rgba_texel {
    float r;
    float g;
    float b;
    float a;
};

const char * astc_vulkan_semantic_density_name(astc_vulkan_semantic_density density);
const char * astc_vulkan_paired_layout_name(astc_vulkan_paired_layout layout);

// Maps two normalized logical scalar values plus an arbitrary normalized
// steering value to an ASTC source texel. The steering value is intentionally
// not part of paired_weight reconstruction.
astc_vulkan_rgba_texel astc_vulkan_make_paired_texel(
    float q0, float q1, float steering, astc_vulkan_paired_layout layout);

// Reconstructs either member from a decoded D2 texel. This exact operation is
// shared by the CPU oracle and the later Vulkan paired shader contract.
float astc_vulkan_paired_weight(
    const astc_vulkan_rgba_texel & texel, unsigned int member,
    astc_vulkan_paired_layout layout);

// D2 maps texture row y to logical output rows 2*y and 2*y+1. The final odd
// row is padded deterministically by the packer and is excluded from loss.
constexpr unsigned int astc_vulkan_d2_logical_rows(unsigned int texture_rows) {
    return texture_rows * 2;
}

constexpr double astc_vulkan_nominal_bits_per_logical_weight(
        const ggml_vk_astc_format_contract & format,
        astc_vulkan_semantic_density density) {
    return (format.block_size_bytes * 8.0) /
           (format.texels_per_block() * static_cast<unsigned int>(density));
}
