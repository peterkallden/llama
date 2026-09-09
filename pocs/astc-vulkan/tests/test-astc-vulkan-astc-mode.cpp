#include "astc-vulkan-astc-mode.h"

#include <cassert>
#include <cstddef>

namespace {

void expect_mode(astc_vulkan_audited_mode mode,
                 astc_vulkan_footprint footprint,
                 uint16_t block_mode,
                 uint32_t endpoint_bits,
                 uint32_t weight_bits,
                 bool dual_plane) {
    const auto * descriptor = astc_vulkan_find_audited_mode(mode);
    assert(descriptor != nullptr);
    assert(descriptor->footprint == footprint);
    assert(descriptor->block_mode == block_mode);
    assert(descriptor->endpoint_field_bits == endpoint_bits);
    assert(descriptor->dual_plane == dual_plane);
    const auto budget = astc_vulkan_audit_mode_budget(*descriptor);
    assert(budget.legal);
    assert(budget.endpoint_field_bits == endpoint_bits);
    assert(budget.weight_bits == weight_bits);
    assert(budget.used_bits == endpoint_bits + weight_bits);
    assert(budget.fixed_block_bits + budget.used_bits == 128u);
    assert(budget.remaining_bits == 128u - budget.used_bits);
}

} // namespace

int main() {
    size_t count = 0;
    const auto * modes = astc_vulkan_audited_modes(&count);
    assert(modes != nullptr && count == 5u);

    expect_mode(astc_vulkan_audited_mode::d1_luminance_binary_6x6,
                astc_vulkan_footprint::k6x6, 0x104u, 16u, 36u, false);
    expect_mode(astc_vulkan_audited_mode::d1_luminance_binary_5x5,
                astc_vulkan_footprint::k5x5, 0x0e1u, 16u, 25u, false);
    expect_mode(astc_vulkan_audited_mode::d1_luminance_quant4_4x4,
                astc_vulkan_footprint::k4x4, 0x042u, 16u, 32u, false);
    expect_mode(astc_vulkan_audited_mode::d2_luminance_alpha_binary_8x5,
                astc_vulkan_footprint::k8x5, 0x065u, 32u, 40u, false);
    expect_mode(astc_vulkan_audited_mode::d2_luminance_alpha_dual_binary_8x5,
                astc_vulkan_footprint::k8x5, 0x4c1u, 32u, 40u, true);

    const auto * d2 = astc_vulkan_find_audited_mode(
        astc_vulkan_audited_mode::d2_luminance_alpha_dual_binary_8x5);
    assert(d2->dual_plane_component == 3u);
    const auto invalid = astc_vulkan_audit_mode_budget(*d2, 64u);
    assert(!invalid.legal);
    return 0;
}

