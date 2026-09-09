#pragma once

// Small legal ASTC subset shared by CPU tests and GPU exact kernels.
//
// The normal-block entries intentionally cover only direct endpoints and
// power-of-two (binary ISE) weight alphabets. Their physical mode/footprint
// accounting is centralized in astc-vulkan-astc-mode; broader trit/quint ISE
// modes remain a later, separately verified extension.

#include "astc-gpu-encoder.h"
#include "astc-vulkan-astc-mode.h"

#include <array>
#include <cstdint>
#include <vector>

struct astc_gpu_exact_subset_block {
    uint32_t source_block_id = 0;
    astc_gpu_exact_subset_kind mode = astc_gpu_exact_subset_kind::void_extent_unorm16;
    std::array<uint16_t, 4> unorm16_rgba{};
    std::array<uint8_t, 16> payload{};
};

// Maps every normal-block exact-subset kind to the CPU-audited physical mode
// descriptor. Refinement variants intentionally map to the same descriptor:
// they change fitting, not the ASTC runtime mode. Void-extent has no normal
// block-mode descriptor and therefore returns nullptr.
const astc_vulkan_astc_mode_descriptor * astc_gpu_exact_subset_audited_mode(
    astc_gpu_exact_subset_kind kind);

// Ensures a GPU request does not accidentally pair a legal mode with the
// wrong physical footprint before dispatch. This remains representation-free;
// D1/D2 semantics live in their own frontends.
bool astc_gpu_exact_subset_matches_audited_mode(
    const astc_gpu_encoder_request & request);

// Encodes a constant-color, void-extent ASTC payload from a UNORM16 RGBA
// value. The result is independent of the ASTC footprint.
std::array<uint8_t, 16> astc_gpu_exact_subset_pack_void_extent_unorm16(
    const std::array<uint16_t, 4> & rgba);

// Packs a normal, single-partition ASTC 6x6 LDR luminance block. It uses a
// 6x6 QUANT_2 weight grid and two direct QUANT_256 luminance endpoints.
// weights are raster ordered zero/one interpolation values.
std::array<uint8_t, 16> astc_gpu_exact_subset_pack_d1_luminance_binary_6x6(
    uint8_t endpoint_low, uint8_t endpoint_high,
    const std::array<uint8_t, 36> & weights);

// Uses the identical legal physical 6x6 form as the binary packer above.
// The distinct encoder kind selects bounded endpoint/weight refinement before
// this packer is called; it does not introduce a new ASTC runtime mode.

// Same normal LDR luminance form for 5x5. Its QUANT_2 grid is the smallest
// legal binary-weight mode for this footprint.
std::array<uint8_t, 16> astc_gpu_exact_subset_pack_d1_luminance_binary_5x5(
    uint8_t endpoint_low, uint8_t endpoint_high,
    const std::array<uint8_t, 25> & weights);

// Normal LDR 4x4 luminance block with a QUANT_4 4x4 weight grid. This mode
// remains raw-bit packed, avoiding BISE while covering the 4x4 D1 footprint.
std::array<uint8_t, 16> astc_gpu_exact_subset_pack_d1_luminance_quant4_4x4(
    uint8_t endpoint_low, uint8_t endpoint_high,
    const std::array<uint8_t, 16> & weights);

// Physical L+A 8x5 mode for the D2 frontend: four direct QUANT_256 endpoint
// values and a one-plane binary 8x5 weight grid. The shared layer knows only
// channels; D2 owns the paired-row semantic reconstruction.
std::array<uint8_t, 16> astc_gpu_exact_subset_pack_luminance_alpha_binary_8x5(
    const std::array<uint8_t, 4> & endpoints,
    const std::array<uint8_t, 40> & weights);

// A second legal D2-LA candidate: a 5x4 dual-plane grid in an 8x5 block.
// Plane 2 is Alpha, which is semantic data for D2-LA rather than steering.
std::array<uint8_t, 16> astc_gpu_exact_subset_pack_luminance_alpha_dual_binary_8x5(
    const std::array<uint8_t, 4> & endpoints,
    const std::array<uint8_t, 40> & interleaved_weights);

// Builds the first exact-subset candidate for each physical source block. All
// source channels must be finite UNORM values. The constant color is the
// arithmetic mean of the block's physical RGBA texels.
bool astc_gpu_exact_subset_encode_cpu(
    const astc_gpu_encoder_request & request,
    std::vector<astc_gpu_exact_subset_block> & blocks);
