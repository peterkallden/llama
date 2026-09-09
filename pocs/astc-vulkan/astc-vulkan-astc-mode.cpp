#include "astc-vulkan-astc-mode.h"

#include <array>

namespace {

using mode = astc_vulkan_audited_mode;
using family = astc_vulkan_ise_family;

constexpr uint8_t kEndpointFormatLuminance = 0u;
constexpr uint8_t kEndpointFormatLuminanceAlpha = 4u;
constexpr uint32_t kLuminanceEndpointBits = 16u;
constexpr uint32_t kLuminanceAlphaEndpointBits = 32u;

constexpr astc_vulkan_ise_range binary_range(uint8_t bits_per_value) {
    return {family::binary, bits_per_value,
            static_cast<uint16_t>(1u << bits_per_value)};
}

constexpr auto kQuant2 = binary_range(1u);
constexpr auto kQuant4 = binary_range(2u);

constexpr std::array<astc_vulkan_astc_mode_descriptor, 5> kModes{{
    {mode::d1_luminance_binary_6x6, "d1_luminance_binary_6x6",
     astc_vulkan_footprint::k6x6, 0x104u,
     astc_vulkan_endpoint_family::luminance, kEndpointFormatLuminance, 2u,
     kLuminanceEndpointBits, 36u, kQuant2, false, 0u},
    {mode::d1_luminance_binary_5x5, "d1_luminance_binary_5x5",
     astc_vulkan_footprint::k5x5, 0x0e1u,
     astc_vulkan_endpoint_family::luminance, kEndpointFormatLuminance, 2u,
     kLuminanceEndpointBits, 25u, kQuant2, false, 0u},
    {mode::d1_luminance_quant4_4x4, "d1_luminance_quant4_4x4",
     astc_vulkan_footprint::k4x4, 0x042u,
     astc_vulkan_endpoint_family::luminance, kEndpointFormatLuminance, 2u,
     kLuminanceEndpointBits, 16u, kQuant4, false, 0u},
    {mode::d2_luminance_alpha_binary_8x5, "d2_luminance_alpha_binary_8x5",
     astc_vulkan_footprint::k8x5, 0x065u,
     astc_vulkan_endpoint_family::luminance_alpha, kEndpointFormatLuminanceAlpha, 4u,
     kLuminanceAlphaEndpointBits, 40u, kQuant2, false, 0u},
    {mode::d2_luminance_alpha_dual_binary_8x5, "d2_luminance_alpha_dual_binary_8x5",
     astc_vulkan_footprint::k8x5, 0x4c1u,
     astc_vulkan_endpoint_family::luminance_alpha, kEndpointFormatLuminanceAlpha, 4u,
     kLuminanceAlphaEndpointBits, 40u, kQuant2, true, 3u},
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
