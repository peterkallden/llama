#pragma once

// Experimental, host-neutral description of the ASTC formats used by the
// Vulkan weight-storage proof of concept. Keep this header independent of
// Vulkan handles so contract tests can run without a GPU or Vulkan loader.

#include <cstdint>

struct ggml_vk_astc_format_contract {
    const char * name;
    uint32_t block_width;
    uint32_t block_height;
    uint32_t block_size_bytes;
    uint32_t channels;

    constexpr uint32_t texels_per_block() const {
        return block_width * block_height;
    }

    constexpr double nominal_bits_per_texel() const {
        return (block_size_bytes * 8.0) / texels_per_block();
    }

    constexpr double nominal_bits_per_channel() const {
        return nominal_bits_per_texel() / channels;
    }
};

// Vulkan ASTC LDR blocks are 128 bits. These values describe storage density,
// not independently exact bits per weight: ASTC shares coding parameters in a
// block.
inline constexpr ggml_vk_astc_format_contract ggml_vk_astc_4x4_unorm_rgba = {
    "ASTC 4x4 UNORM RGBA", 4, 4, 16, 4,
};

inline constexpr ggml_vk_astc_format_contract ggml_vk_astc_5x5_unorm_rgba = {
    "ASTC 5x5 UNORM RGBA", 5, 5, 16, 4,
};

inline constexpr ggml_vk_astc_format_contract ggml_vk_astc_6x6_unorm_rgba = {
    "ASTC 6x6 UNORM RGBA", 6, 6, 16, 4,
};

inline constexpr ggml_vk_astc_format_contract ggml_vk_astc_8x6_unorm_rgba = {
    "ASTC 8x6 UNORM RGBA", 8, 6, 16, 4,
};

inline constexpr ggml_vk_astc_format_contract ggml_vk_astc_8x8_unorm_rgba = {
    "ASTC 8x8 UNORM RGBA", 8, 8, 16, 4,
};

constexpr uint32_t ggml_vk_astc_block_count(uint32_t extent, uint32_t block_extent) {
    return (extent + block_extent - 1) / block_extent;
}

constexpr uint64_t ggml_vk_astc_image_block_count(
        const ggml_vk_astc_format_contract & format, uint32_t width, uint32_t height) {
    return static_cast<uint64_t>(ggml_vk_astc_block_count(width, format.block_width)) *
           ggml_vk_astc_block_count(height, format.block_height);
}

constexpr uint64_t ggml_vk_astc_image_storage_bytes(
        const ggml_vk_astc_format_contract & format, uint32_t width, uint32_t height) {
    return ggml_vk_astc_image_block_count(format, width, height) * format.block_size_bytes;
}

constexpr uint32_t ggml_vk_astc_image_texel_count(
        uint32_t width, uint32_t height) {
    return width * height;
}
