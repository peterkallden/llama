#pragma once

#include <cstddef>
#include <cstdint>

enum class astc_vulkan_footprint : uint8_t {
    k4x4 = 0,
    k5x5 = 1,
    k6x6 = 2,
    k8x6 = 3,
    // Keep values serialized by existing manifests stable; append new formats.
    k8x8 = 4,
    k10x6 = 5,
    k10x8 = 6,
    // D2's first iso-rate format. Appending preserves serialized manifests.
    k8x5 = 7,
    // D2's next lower-rate candidate. Appending preserves serialized manifests.
    k10x5 = 8,
    // D2's higher-rate five-row candidate. Appending preserves serialized manifests.
    k6x5 = 9,
};

struct astc_vulkan_format_info {
    uint32_t block_width;
    uint32_t block_height;
    uint32_t block_bytes;
};

inline constexpr size_t astc_vulkan_footprint_count = 10;

astc_vulkan_format_info astc_vulkan_format(astc_vulkan_footprint footprint);
bool astc_vulkan_footprint_is_valid(astc_vulkan_footprint footprint);
// 6x5, 8x5, 8x6, 10x5, 10x6, 8x8, and 10x8 are standard Vulkan formats, but remain opt-in in this PoC
// until broader device coverage and quality data are available.
bool astc_vulkan_footprint_is_experimental(astc_vulkan_footprint footprint);
uint64_t astc_vulkan_block_count(astc_vulkan_footprint footprint,
                                 uint32_t width, uint32_t height);
uint64_t astc_vulkan_image_bytes(astc_vulkan_footprint footprint,
                                 uint32_t width, uint32_t height);
