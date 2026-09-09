#include "astc-vulkan-astc-mode.h"

#include <cstddef>
#include <cstdlib>

namespace {

void expect_mode(astc_vulkan_audited_mode mode,
                 astc_vulkan_footprint footprint,
                 uint16_t block_mode,
                 uint32_t endpoint_bits,
                 uint32_t weight_bits,
                 bool dual_plane) {
    const auto require = [](bool condition) {
        if (!condition) std::abort();
    };
    const auto * descriptor = astc_vulkan_find_audited_mode(mode);
    require(descriptor != nullptr);
    require(descriptor->footprint == footprint);
    require(descriptor->block_mode == block_mode);
    require(descriptor->endpoint_field_bits == endpoint_bits);
    require(descriptor->dual_plane == dual_plane);
    const auto budget = astc_vulkan_audit_mode_budget(*descriptor);
    require(budget.legal);
    require(budget.endpoint_field_bits == endpoint_bits);
    require(budget.weight_bits == weight_bits);
    require(budget.used_bits == endpoint_bits + weight_bits);
    require(budget.fixed_block_bits + budget.used_bits == 128u);
    require(budget.remaining_bits == 128u - budget.used_bits);
}

} // namespace

int main() {
    const auto require = [](bool condition) {
        if (!condition) std::abort();
    };
    size_t count = 0;
    const auto * modes = astc_vulkan_audited_modes(&count);
    require(modes != nullptr && count == 9u);

    expect_mode(astc_vulkan_audited_mode::d1_luminance_binary_6x6,
                astc_vulkan_footprint::k6x6, 0x104u, 16u, 36u, false);
    expect_mode(astc_vulkan_audited_mode::d1_luminance_binary_5x5,
                astc_vulkan_footprint::k5x5, 0x0e1u, 16u, 25u, false);
    expect_mode(astc_vulkan_audited_mode::d1_luminance_quant4_4x4,
                astc_vulkan_footprint::k4x4, 0x042u, 16u, 32u, false);
    expect_mode(astc_vulkan_audited_mode::d2_luminance_alpha_binary_6x5,
                astc_vulkan_footprint::k6x5, 0x161u, 32u, 30u, false);
    expect_mode(astc_vulkan_audited_mode::d2_luminance_alpha_dual_binary_6x5,
                astc_vulkan_footprint::k6x5, 0x4a1u, 32u, 30u, true);
    expect_mode(astc_vulkan_audited_mode::d2_luminance_alpha_binary_8x5,
                astc_vulkan_footprint::k8x5, 0x065u, 32u, 40u, false);
    expect_mode(astc_vulkan_audited_mode::d2_luminance_alpha_dual_binary_8x5,
                astc_vulkan_footprint::k8x5, 0x4c1u, 32u, 40u, true);
    expect_mode(astc_vulkan_audited_mode::d2_luminance_alpha_binary_10x5,
                astc_vulkan_footprint::k10x5, 0x165u, 32u, 50u, false);
    expect_mode(astc_vulkan_audited_mode::d2_luminance_alpha_dual_binary_10x5,
                astc_vulkan_footprint::k10x5, 0x4e1u, 32u, 50u, true);

    const auto * d2 = astc_vulkan_find_audited_mode(
        astc_vulkan_audited_mode::d2_luminance_alpha_dual_binary_8x5);
    require(d2->dual_plane_component == 3u);
    const auto invalid = astc_vulkan_audit_mode_budget(*d2, 64u);
    require(!invalid.legal);
    return 0;
}
