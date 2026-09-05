#pragma once

#include "astc-vulkan-artifact.h"
#include "astc-vulkan-contract.h"

#include <vector>

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

// astc_vulkan_paired_semantic is owned by astc-vulkan-artifact.h because the
// semantic is part of a serialized cache artifact as well as this CPU/GPU
// paired-D2 contract.

// The semantic basis used before mapping a D2 pair into RGB. Direct is the
// deployed v1 contract. Common/difference is an offline research basis: it
// preserves q0/q1 before ASTC but needs additional basis metadata before it
// can be exported to the existing one-bit layout map.
enum class astc_vulkan_paired_basis : unsigned char {
    direct = 0,
    common_difference = 1,
};

// A bounded source-side Alpha codebook for D2. These perturb only the
// encoder-visible steering lane, never the runtime semantic decoder. They are
// intentionally discrete, parallel and deterministic; this is the practical
// low-rate alternative to expensive full PV iteration.
enum class astc_vulkan_paired_steering_basis : unsigned char {
    neutral,
    x_ramp,
    y_ramp,
    saddle,
    x_plus_y,
    x_minus_y,
};

struct astc_vulkan_paired_steering_factor {
    float amplitude = 0.0f;
    astc_vulkan_paired_steering_basis basis = astc_vulkan_paired_steering_basis::neutral;
};

struct astc_vulkan_rgba_texel {
    float r;
    float g;
    float b;
    float a;
};

const char * astc_vulkan_semantic_density_name(astc_vulkan_semantic_density density);
const char * astc_vulkan_paired_layout_name(astc_vulkan_paired_layout layout);
const char * astc_vulkan_paired_semantic_name(astc_vulkan_paired_semantic semantic);
const char * astc_vulkan_paired_basis_name(astc_vulkan_paired_basis basis);
const char * astc_vulkan_paired_steering_basis_name(astc_vulkan_paired_steering_basis basis);

// Returns the deterministic v1 steering order used as candidate tie-break
// input: neutral, signed ramps/saddle, then signed diagonal combinations.
std::vector<astc_vulkan_paired_steering_factor> astc_vulkan_make_paired_steering_codebook();

// x and y are normalized coordinates in [-1, 1]. Composite bases are scaled
// back to [-1, 1], so all codebook amplitudes have comparable headroom.
float astc_vulkan_paired_steering_basis_value(
    astc_vulkan_paired_steering_basis basis, float x, float y);

// Maps two normalized logical scalar values plus an arbitrary normalized
// steering value to an ASTC source texel. The steering value is intentionally
// not part of paired_weight reconstruction.
astc_vulkan_rgba_texel astc_vulkan_make_paired_texel(
    float q0, float q1, float steering, astc_vulkan_paired_layout layout,
    astc_vulkan_paired_basis basis = astc_vulkan_paired_basis::direct,
    astc_vulkan_paired_semantic semantic = astc_vulkan_paired_semantic::direct_rgb);

// Reconstructs either member from a decoded D2 texel. This exact operation is
// shared by the CPU oracle and the later Vulkan paired shader contract.
float astc_vulkan_paired_weight(
    const astc_vulkan_rgba_texel & texel, unsigned int member,
    astc_vulkan_paired_layout layout,
    astc_vulkan_paired_basis basis = astc_vulkan_paired_basis::direct,
    astc_vulkan_paired_semantic semantic = astc_vulkan_paired_semantic::direct_rgb);

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
