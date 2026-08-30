#include "astc-vulkan-tensor-contract.h"

#include <cassert>

int main() {
    constexpr ggml_vk_astc_weight_layout small{ 3, 10, 4 };
    static_assert(small.is_valid(), "small layout should be valid");
    static_assert(small.texel_columns() == 3, "four scalar channels per texel");
    static_assert(small.texel_count() == 9, "row-major texel count changed");
    static_assert(small.padded_scalar_count() == 36, "channel padding changed");
    static_assert(small.storage_bytes(ggml_vk_astc_4x4_unorm_rgba) == 16,
                  "small 4x4 image should occupy one block");
    static_assert(small.storage_bytes(ggml_vk_astc_6x6_unorm_rgba) == 16,
                  "small 6x6 image should occupy one block");

    constexpr ggml_vk_astc_weight_layout wide{ 100, 4096, 4 };
    assert(wide.is_valid());
    assert(wide.texel_columns() == 1024);
    assert(wide.storage_bytes(ggml_vk_astc_4x4_unorm_rgba) == 102400);
    assert(wide.storage_bytes(ggml_vk_astc_6x6_unorm_rgba) == 46512);

    constexpr ggml_vk_astc_weight_layout invalid{ 0, 10, 4 };
    static_assert(!invalid.is_valid(), "zero-row layout must be rejected");
    constexpr ggml_vk_astc_weight_layout wrong_channels{ 2, 8, 3 };
    static_assert(!wrong_channels.is_valid(), "non-RGBA layout must be rejected");

    const ggml_vk_astc_weight_metadata metadata{ 100, 4096, 1.0f, 0.0f };
    assert(metadata.logical_rows == wide.rows);
    assert(metadata.logical_columns == wide.columns);
    return 0;
}

