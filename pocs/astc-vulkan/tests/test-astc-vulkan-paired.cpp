#include "astc-vulkan-paired.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace {

void assert_close(float actual, float expected) {
    assert(std::fabs(actual - expected) < 1e-6f);
}

} // namespace

int main() {
    static_assert(astc_vulkan_d2_logical_rows(5) == 10, "D2 row mapping changed");
    static_assert(astc_vulkan_nominal_bits_per_logical_weight(
                      ggml_vk_astc_8x5_unorm_rgba,
                      astc_vulkan_semantic_density::d2_paired) == 1.6,
                  "D2 8x5 must be the 1.6 b/w iso-rate test point");
    static_assert(astc_vulkan_nominal_bits_per_logical_weight(
                      ggml_vk_astc_6x5_unorm_rgba,
                      astc_vulkan_semantic_density::d2_paired) == 128.0 / 60.0,
                  "D2 6x5 must retain the five-row paired mapping");

    assert(std::strcmp(astc_vulkan_semantic_density_name(
               astc_vulkan_semantic_density::d2_paired), "paired-d2") == 0);
    assert(std::strcmp(astc_vulkan_paired_layout_name(
               astc_vulkan_paired_layout::rg_b), "rg-b") == 0);
    assert(std::strcmp(astc_vulkan_paired_basis_name(
               astc_vulkan_paired_basis::common_difference), "common-difference") == 0);

    const auto rg_b = astc_vulkan_make_paired_texel(0.25f, 0.75f, 0.5f,
                                                     astc_vulkan_paired_layout::rg_b);
    assert_close(rg_b.r, 0.25f);
    assert_close(rg_b.g, 0.25f);
    assert_close(rg_b.b, 0.75f);
    assert_close(rg_b.a, 0.5f);
    assert_close(astc_vulkan_paired_weight(rg_b, 0, astc_vulkan_paired_layout::rg_b), 0.25f);
    assert_close(astc_vulkan_paired_weight(rg_b, 1, astc_vulkan_paired_layout::rg_b), 0.75f);

    const auto r_gb = astc_vulkan_make_paired_texel(0.25f, 0.75f, 0.5f,
                                                      astc_vulkan_paired_layout::r_gb);
    assert_close(astc_vulkan_paired_weight(r_gb, 0, astc_vulkan_paired_layout::r_gb), 0.25f);
    assert_close(astc_vulkan_paired_weight(r_gb, 1, astc_vulkan_paired_layout::r_gb), 0.75f);

    // Common/difference is exactly invertible before ASTC. It remains an
    // offline test basis until a versioned two-bit basis map is deployed.
    const auto common_diff_rg_b = astc_vulkan_make_paired_texel(
        0.25f, 0.75f, 0.5f, astc_vulkan_paired_layout::rg_b,
        astc_vulkan_paired_basis::common_difference);
    assert_close(common_diff_rg_b.r, 0.50f);
    assert_close(common_diff_rg_b.g, 0.50f);
    assert_close(common_diff_rg_b.b, 0.25f);
    assert_close(astc_vulkan_paired_weight(common_diff_rg_b, 0,
                                            astc_vulkan_paired_layout::rg_b,
                                            astc_vulkan_paired_basis::common_difference), 0.25f);
    assert_close(astc_vulkan_paired_weight(common_diff_rg_b, 1,
                                            astc_vulkan_paired_layout::rg_b,
                                            astc_vulkan_paired_basis::common_difference), 0.75f);

    // Alpha is an encoder-only steering lane in D2; changing it cannot alter
    // the semantic runtime reconstruction before ASTC compression.
    const auto steering_changed = astc_vulkan_make_paired_texel(
        0.25f, 0.75f, 1.0f, astc_vulkan_paired_layout::rg_b);
    assert_close(astc_vulkan_paired_weight(steering_changed, 0,
                                            astc_vulkan_paired_layout::rg_b), 0.25f);
    assert_close(astc_vulkan_paired_weight(steering_changed, 1,
                                            astc_vulkan_paired_layout::rg_b), 0.75f);

    // D2-LA is a distinct semantic decoder: Alpha carries a real second
    // logical lane, so the orientation bit selects luminance-versus-Alpha.
    const auto la_rg_b = astc_vulkan_make_paired_texel(
        0.25f, 0.75f, 0.0f, astc_vulkan_paired_layout::rg_b,
        astc_vulkan_paired_basis::direct,
        astc_vulkan_paired_semantic::luminance_alpha);
    assert_close(la_rg_b.r, 0.25f);
    assert_close(la_rg_b.g, 0.25f);
    assert_close(la_rg_b.b, 0.25f);
    assert_close(la_rg_b.a, 0.75f);
    assert_close(astc_vulkan_paired_weight(la_rg_b, 0, astc_vulkan_paired_layout::rg_b,
                                            astc_vulkan_paired_basis::direct,
                                            astc_vulkan_paired_semantic::luminance_alpha), 0.25f);
    assert_close(astc_vulkan_paired_weight(la_rg_b, 1, astc_vulkan_paired_layout::rg_b,
                                            astc_vulkan_paired_basis::direct,
                                            astc_vulkan_paired_semantic::luminance_alpha), 0.75f);
    const auto la_r_gb = astc_vulkan_make_paired_texel(
        0.25f, 0.75f, 0.0f, astc_vulkan_paired_layout::r_gb,
        astc_vulkan_paired_basis::direct,
        astc_vulkan_paired_semantic::luminance_alpha);
    assert_close(astc_vulkan_paired_weight(la_r_gb, 0, astc_vulkan_paired_layout::r_gb,
                                            astc_vulkan_paired_basis::direct,
                                            astc_vulkan_paired_semantic::luminance_alpha), 0.25f);
    assert_close(astc_vulkan_paired_weight(la_r_gb, 1, astc_vulkan_paired_layout::r_gb,
                                            astc_vulkan_paired_basis::direct,
                                            astc_vulkan_paired_semantic::luminance_alpha), 0.75f);

    assert_close(static_cast<float>(astc_vulkan_nominal_bits_per_logical_weight(
                     ggml_vk_astc_10x8_unorm_rgba,
                     astc_vulkan_semantic_density::d1_scalar)), 1.6f);
    const auto codebook = astc_vulkan_make_paired_steering_codebook();
    assert(codebook.size() == 11);
    assert(codebook.front().amplitude == 0.0f);
    assert_close(astc_vulkan_paired_steering_basis_value(
                     astc_vulkan_paired_steering_basis::x_plus_y, 1.0f, 1.0f), 1.0f);
    assert_close(astc_vulkan_paired_steering_basis_value(
                     astc_vulkan_paired_steering_basis::x_minus_y, 1.0f, -1.0f), 1.0f);
    return 0;
}
