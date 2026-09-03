#include "astc-vulkan-contract.h"

#include <cassert>
#include <cmath>

static void assert_close(double actual, double expected) {
    assert(std::fabs(actual - expected) < 1e-12);
}

int main() {
    constexpr auto format_4x4 = ggml_vk_astc_4x4_unorm_rgba;
    constexpr auto format_5x5 = ggml_vk_astc_5x5_unorm_rgba;
    constexpr auto format_6x6 = ggml_vk_astc_6x6_unorm_rgba;
    constexpr auto format_8x5 = ggml_vk_astc_8x5_unorm_rgba;
    constexpr auto format_8x6 = ggml_vk_astc_8x6_unorm_rgba;
    constexpr auto format_10x6 = ggml_vk_astc_10x6_unorm_rgba;
    constexpr auto format_8x8 = ggml_vk_astc_8x8_unorm_rgba;
    constexpr auto format_10x8 = ggml_vk_astc_10x8_unorm_rgba;

    static_assert(format_4x4.block_size_bytes == 16, "ASTC blocks must be 128 bits");
    static_assert(format_5x5.block_size_bytes == 16, "ASTC blocks must be 128 bits");
    static_assert(format_6x6.block_size_bytes == 16, "ASTC blocks must be 128 bits");
    static_assert(format_8x5.block_size_bytes == 16, "ASTC blocks must be 128 bits");
    static_assert(format_8x6.block_size_bytes == 16, "ASTC blocks must be 128 bits");
    static_assert(format_10x6.block_size_bytes == 16, "ASTC blocks must be 128 bits");
    static_assert(format_8x8.block_size_bytes == 16, "ASTC blocks must be 128 bits");
    static_assert(format_10x8.block_size_bytes == 16, "ASTC blocks must be 128 bits");
    static_assert(format_4x4.texels_per_block() == 16, "ASTC 4x4 texel count changed");
    static_assert(format_5x5.texels_per_block() == 25, "ASTC 5x5 texel count changed");
    static_assert(format_6x6.texels_per_block() == 36, "ASTC 6x6 texel count changed");
    static_assert(format_8x5.texels_per_block() == 40, "ASTC 8x5 texel count changed");
    static_assert(format_8x6.texels_per_block() == 48, "ASTC 8x6 texel count changed");
    static_assert(format_10x6.texels_per_block() == 60, "ASTC 10x6 texel count changed");
    static_assert(format_8x8.texels_per_block() == 64, "ASTC 8x8 texel count changed");
    static_assert(format_10x8.texels_per_block() == 80, "ASTC 10x8 texel count changed");
    static_assert(format_4x4.channels == 4, "The PoC assumes RGBA texels");
    static_assert(format_6x6.channels == 4, "The PoC assumes RGBA texels");

    assert_close(format_4x4.nominal_bits_per_texel(), 8.0);
    assert_close(format_5x5.nominal_bits_per_texel(), 128.0 / 25.0);
    assert_close(format_6x6.nominal_bits_per_texel(), 128.0 / 36.0);
    assert_close(format_8x5.nominal_bits_per_texel(), 3.2);
    assert_close(format_8x6.nominal_bits_per_texel(), 128.0 / 48.0);
    assert_close(format_10x6.nominal_bits_per_texel(), 128.0 / 60.0);
    assert_close(format_8x8.nominal_bits_per_texel(), 2.0);
    assert_close(format_10x8.nominal_bits_per_texel(), 128.0 / 80.0);
    assert_close(format_4x4.nominal_bits_per_channel(), 2.0);
    assert_close(format_6x6.nominal_bits_per_channel(), 128.0 / 144.0);

    assert_close(128.0 / format_4x4.texels_per_block(), 8.0);
    assert_close(128.0 / format_5x5.texels_per_block(), 128.0 / 25.0);
    assert_close(128.0 / format_6x6.texels_per_block(), 128.0 / 36.0);
    assert_close(128.0 / format_4x4.texels_per_block() / format_4x4.channels, 2.0);
    assert_close(128.0 / format_6x6.texels_per_block() / format_6x6.channels, 128.0 / 144.0);

    assert(ggml_vk_astc_block_count(16, 4) == 4);
    assert(ggml_vk_astc_block_count(17, 4) == 5);
    assert(ggml_vk_astc_block_count(25, 5) == 5);
    assert(ggml_vk_astc_block_count(26, 5) == 6);
    assert(ggml_vk_astc_block_count(36, 6) == 6);
    assert(ggml_vk_astc_block_count(37, 6) == 7);
    assert(ggml_vk_astc_block_count(49, 8) == 7);
    assert(ggml_vk_astc_image_texel_count(6, 6) == 36);

    constexpr uint32_t large_extent = 4096;
    static_assert(ggml_vk_astc_image_block_count(
                      format_4x4, large_extent, large_extent) == 1024u * 1024u,
                  "ASTC 4x4 block rounding changed");
    static_assert(ggml_vk_astc_image_block_count(
                      format_5x5, large_extent, large_extent) == 820u * 820u,
                  "ASTC 5x5 block rounding changed");
    static_assert(ggml_vk_astc_image_block_count(
                      format_6x6, large_extent, large_extent) == 683u * 683u,
                  "ASTC 6x6 block rounding changed");
    static_assert(ggml_vk_astc_image_block_count(
                      format_8x6, large_extent, large_extent) == 512u * 683u,
                  "ASTC 8x6 block rounding changed");
    static_assert(ggml_vk_astc_image_block_count(
                      format_10x6, large_extent, large_extent) == 410u * 683u,
                  "ASTC 10x6 block rounding changed");
    static_assert(ggml_vk_astc_image_block_count(
                      format_8x8, large_extent, large_extent) == 512u * 512u,
                  "ASTC 8x8 block rounding changed");
    static_assert(ggml_vk_astc_image_block_count(
                      format_10x8, large_extent, large_extent) == 410u * 512u,
                  "ASTC 10x8 block rounding changed");
    assert(ggml_vk_astc_image_storage_bytes(format_4x4, large_extent, large_extent) ==
           67108864u);
    assert(ggml_vk_astc_image_storage_bytes(format_5x5, large_extent, large_extent) ==
           10758400u);
    assert(ggml_vk_astc_image_storage_bytes(format_6x6, large_extent, large_extent) ==
           7463824u);
    assert(ggml_vk_astc_image_storage_bytes(format_8x6, large_extent, large_extent) ==
           5595136u);
    assert(ggml_vk_astc_image_storage_bytes(format_10x6, large_extent, large_extent) ==
           4480480u);
    assert(ggml_vk_astc_image_storage_bytes(format_8x8, large_extent, large_extent) ==
           4194304u);
    assert(ggml_vk_astc_image_storage_bytes(format_10x8, large_extent, large_extent) ==
           3358720u);
    assert(ggml_vk_astc_image_storage_bytes(format_4x4, 5, 5) == 64u);
    assert(ggml_vk_astc_image_storage_bytes(format_6x6, 5, 5) == 16u);

    return 0;
}
