#include "astc-vulkan-contract.h"

#include <cassert>

int main() {
    // Host smoke only: this deliberately requires no Vulkan loader or GPU.
    // Device execution becomes a separate smoke once the sampled-image shader
    // and runtime resource path exist.
    assert(ggml_vk_astc_4x4_unorm_rgba.block_size_bytes == 16);
    assert(ggml_vk_astc_6x6_unorm_rgba.block_size_bytes == 16);
    assert(ggml_vk_astc_8x6_unorm_rgba.block_size_bytes == 16);
    assert(ggml_vk_astc_8x8_unorm_rgba.block_size_bytes == 16);
    assert(ggml_vk_astc_10x8_unorm_rgba.block_size_bytes == 16);
    assert(ggml_vk_astc_4x4_unorm_rgba.texels_per_block() <
           ggml_vk_astc_6x6_unorm_rgba.texels_per_block());
    assert(ggml_vk_astc_6x6_unorm_rgba.texels_per_block() <
           ggml_vk_astc_8x6_unorm_rgba.texels_per_block());
    assert(ggml_vk_astc_4x4_unorm_rgba.block_size_bytes ==
           ggml_vk_astc_6x6_unorm_rgba.block_size_bytes);
    return 0;
}
