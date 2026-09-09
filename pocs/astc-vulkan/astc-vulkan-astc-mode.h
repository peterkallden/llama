#pragma once

// CPU-side audit of the small ASTC mode subset currently used by the offline
// D1/D2 candidate banks.  This is deliberately a descriptor layer: it does
// not pack a block and it does not change the GPU shaders.  Its purpose is to
// make the physical bit budget explicit before any BISE/ISE logic is moved to
// a device backend.

#include "astc-vulkan-format.h"
#include "astc-vulkan-ise.h"

#include <cstddef>
#include <cstdint>

enum class astc_vulkan_audited_mode : uint8_t {
    d1_luminance_binary_6x6 = 0,
    d1_luminance_binary_5x5,
    d1_luminance_quant4_4x4,
    d2_luminance_alpha_binary_8x5,
    d2_luminance_alpha_dual_binary_8x5,
};

enum class astc_vulkan_endpoint_family : uint8_t {
    luminance,
    luminance_alpha,
};

struct astc_vulkan_astc_mode_descriptor {
    astc_vulkan_audited_mode mode = astc_vulkan_audited_mode::d1_luminance_binary_6x6;
    const char * name = nullptr;
    astc_vulkan_footprint footprint = astc_vulkan_footprint::k4x4;
    uint16_t block_mode = 0;
    astc_vulkan_endpoint_family endpoint_family = astc_vulkan_endpoint_family::luminance;
    uint8_t endpoint_format = 0;
    uint8_t endpoint_value_count = 0;
    uint32_t endpoint_field_bits = 0;
    uint32_t weight_value_count = 0;
    astc_vulkan_ise_range weight_range{};
    bool dual_plane = false;
    uint8_t dual_plane_component = 0;
};

struct astc_vulkan_astc_mode_budget {
    uint32_t fixed_block_bits = 0;
    uint32_t endpoint_field_bits = 0;
    uint32_t weight_bits = 0;
    uint32_t used_bits = 0;
    uint32_t remaining_bits = 0;
    bool legal = false;
};

// Returns the immutable descriptor for one currently audited physical mode.
const astc_vulkan_astc_mode_descriptor * astc_vulkan_find_audited_mode(
    astc_vulkan_audited_mode mode);

// Returns every descriptor in stable mode-bank order.
const astc_vulkan_astc_mode_descriptor * astc_vulkan_audited_modes(
    size_t * count);

// Computes the 128-bit payload accounting for a descriptor.  Endpoint fields
// are direct QUANT_256 fields in these existing modes; only the binary/trit/
// quint weight stream is evaluated by the abstract CPU ISE layer.  The
// residual fixed_block_bits is reserved for mode/header/partition/plane and
// currently-unused physical fields.  It is audited accounting, not a claim
// that all residual bits are independently reallocatable.
astc_vulkan_astc_mode_budget astc_vulkan_audit_mode_budget(
    const astc_vulkan_astc_mode_descriptor & descriptor,
    uint32_t payload_bits = 128);

