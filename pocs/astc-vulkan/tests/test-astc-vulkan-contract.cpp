#include "astc-vulkan-contract.h"

#include <cassert>
#include <cmath>

static void assert_close(double actual, double expected) {
    assert(std::fabs(actual - expected) < 1e-12);
}

int main() {
    constexpr auto format_4x4 = ggml_vk_astc_4x4_unorm_rgba;
    constexpr auto format_6x6 = ggml_vk_astc_6x6_unorm_rgba;

    static_assert(format_4x4.block_size_bytes == 16, "ASTC blocks must be 128 bits");
    static_assert(format_6x6.block_size_bytes == 16, "ASTC blocks must be 128 bits");
    static_assert(format_4x4.texels_per_block() == 16, "ASTC 4x4 texel count changed");
    static_assert(format_6x6.texels_per_block() == 36, "ASTC 6x6 texel count changed");
    static_assert(format_4x4.channels == 4, "The PoC assumes RGBA texels");
    static_assert(format_6x6.channels == 4, "The PoC assumes RGBA texels");

    assert_close(format_4x4.nominal_bits_per_texel(), 8.0);
    assert_close(format_6x6.nominal_bits_per_texel(), 128.0 / 36.0);
    assert_close(format_4x4.nominal_bits_per_channel(), 2.0);
    assert_close(format_6x6.nominal_bits_per_channel(), 128.0 / 144.0);

    assert_close(128.0 / format_4x4.texels_per_block(), 8.0);
    assert_close(128.0 / format_6x6.texels_per_block(), 128.0 / 36.0);
    assert_close(128.0 / format_4x4.texels_per_block() / format_4x4.channels, 2.0);
    assert_close(128.0 / format_6x6.texels_per_block() / format_6x6.channels, 128.0 / 144.0);

    assert(ggml_vk_astc_block_count(16, 4) == 4);
    assert(ggml_vk_astc_block_count(17, 4) == 5);
    assert(ggml_vk_astc_block_count(36, 6) == 6);
    assert(ggml_vk_astc_block_count(37, 6) == 7);
    assert(ggml_vk_astc_image_texel_count(6, 6) == 36);

    return 0;
}
