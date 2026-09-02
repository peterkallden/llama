#pragma once

#include <cstdint>

enum class astc_vulkan_footprint : uint8_t {
    k4x4 = 0,
    k5x5 = 1,
    k6x6 = 2,
};

struct astc_vulkan_format_info {
    uint32_t block_width;
    uint32_t block_height;
    uint32_t block_bytes;
};

astc_vulkan_format_info astc_vulkan_format(astc_vulkan_footprint footprint);
uint64_t astc_vulkan_block_count(astc_vulkan_footprint footprint,
                                 uint32_t width, uint32_t height);
uint64_t astc_vulkan_image_bytes(astc_vulkan_footprint footprint,
                                 uint32_t width, uint32_t height);
