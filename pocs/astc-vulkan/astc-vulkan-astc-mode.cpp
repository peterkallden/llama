#include "astc-vulkan-astc-mode.h"

#include <array>

namespace {

using mode = astc_vulkan_audited_mode;
using family = astc_vulkan_ise_family;

constexpr astc_vulkan_ise_range binary_range(uint16_t count) {
    return {family::binary, 0, count};
}

constexpr std::array<astc_vulkan_astc_mode_descriptor, 5> kModes{{
    {mode::d1_luminance_binary_6x6, "d1_luminance_binary_6x6",
     astc_vulkan_footprint::k6x6, 0x104u,
     astc_vulkan_endpoint_family::luminance, 0u, 2u, 16u, 36u,
     binary_range(2u), false, 0u},
    {mode::d1_luminance_binary_5x5, "d1_luminance_binary_5x5",
     astc_vulkan_footprint::k5x5, 0x0e1u,
     astc_vulkan_endpoint_family::luminance, 0u, 2u, 16u, 25u,
     binary_range(2u), false, 0u},
    {mode::d1_luminance_quant4_4x4, "d1_luminance_quant4_4x4",
     astc_vulkan_footprint::k4x4, 0x042u,
     astc_vulkan_endpoint_family::luminance, 0u, 2u, 16u, 16u,
     {family::binary, 2u, 4u}, false, 0u},
    {mode::d2_luminance_alpha_binary_8x5, "d2_luminance_alpha_binary_8x5",
     astc_vulkan_footprint::k8x5, 0x065u,
     astc_vulkan_endpoint_family::luminance_alpha, 4u, 4u, 32u, 40u,
     binary_range(2u), false, 0u},
    {mode::d2_luminance_alpha_dual_binary_8x5, "d2_luminance_alpha_dual_binary_8x5",
     astc_vulkan_footprint::k8x5, 0x4c1u,
     astc_vulkan_endpoint_family::luminance_alpha, 4u, 4u, 32u, 40u,
     binary_range(2u), true, 3u},
}};

} // namespace

const astc_vulkan_astc_mode_descriptor * astc_vulkan_find_audited_mode(
    astc_vulkan_audited_mode mode) {
    for (const auto & descriptor : kModes) {
        if (descriptor.mode == mode) return &descriptor;
    }
    return nullptr;
}

const astc_vulkan_astc_mode_descriptor * astc_vulkan_audited_modes(size_t * count) {
    if (count) *count = kModes.size();
    return kModes.data();
}

astc_vulkan_astc_mode_budget astc_vulkan_audit_mode_budget(
    const astc_vulkan_astc_mode_descriptor & descriptor,
    uint32_t payload_bits) {
    astc_vulkan_astc_mode_budget result;
    result.endpoint_field_bits = descriptor.endpoint_field_bits;
    const auto weights = astc_vulkan_ise_evaluate(descriptor.weight_range,
                                                   descriptor.weight_value_count);
    if (!weights.valid || descriptor.endpoint_field_bits > payload_bits ||
        weights.total_bits > payload_bits - descriptor.endpoint_field_bits) {
        return result;
    }
    result.weight_bits = weights.total_bits;
    result.used_bits = descriptor.endpoint_field_bits + result.weight_bits;
    result.remaining_bits = payload_bits - result.used_bits;
    result.fixed_block_bits = result.remaining_bits;
    result.legal = true;
    return result;
}

